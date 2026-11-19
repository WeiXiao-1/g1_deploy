"""Run the C++ GetUpMimic policy against the training bundle's G1 physics."""

from contextlib import ExitStack, contextmanager, nullcontext, suppress
import json
import math
from queue import Empty, SimpleQueue
import subprocess
import time

import mujoco as mj
import numpy as np

from viewer_camera import close_viewer


EPISODE_SECONDS = 15.0
STABLE_SECONDS = 1.0
STANDING_HEIGHT_M = 0.65
STANDING_UPRIGHT_COS = 0.9
STANDBY_DAMPING = 1.0  # Simulated standby Kd, with zero position gain.
GETUP_RAMP_SECONDS = 0.18
AUTO_LIMP_SECONDS = 2.0
AUTO_STANDBY_SECONDS = 0.5


class PolicyStream:
    def __init__(self, process, hardware=False):
        self.process = process
        header = self._read().split()
        name = header[0] if header else ""
        self.drive_pd = hardware and name == "g1-getup-mimic-drive-pd-v1"
        expected = ({"g1-getup-mimic-hardware-v1", "g1-getup-mimic-drive-pd-v1"} if hardware
                    else {"g1-getup-mimic-v1"})
        joint_start = 6 if hardware else 2
        if self.drive_pd:
            joint_start += 29  # Drive effort limits, followed by joint names in DDS order.
        if len(header) != joint_start + 29 or name not in expected or len(set(header[joint_start:])) != 29:
            raise RuntimeError("Unexpected C++ policy handshake. Rebuild g1_getup_mimic")
        self.period = int(header[1]) / 1000
        if self.period != 0.01:
            raise RuntimeError("The GetUpMimic benchmark requires a 10 ms policy period")
        self.joint_names = header[joint_start:]
        self.control_period = int(header[2]) / 1000 if hardware else None
        if hardware:
            self.timeout, self.stop_kd, self.damping_seconds = map(float, header[3:6])
        if self.drive_pd:
            self.drive_effort = np.asarray(header[6:joint_start], dtype=np.float64)
            if not np.isfinite(self.drive_effort).all() or np.any(self.drive_effort <= 0):
                raise RuntimeError("Invalid simulated drive effort limits")

    def _read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("C++ policy exited before replying. Check its output above")
        return line

    def _request(self, line):
        try:
            self.process.stdin.write(line + "\n")
            self.process.stdin.flush()
        except BrokenPipeError as error:
            raise RuntimeError("C++ policy closed its input. Check its output above") from error
        return self._read()

    def reset(self):
        if self._request("reset").strip() != "ok":
            raise RuntimeError("C++ policy reset was not acknowledged")

    def step(self, q, dq, quaternion, omega):
        reply = self._send_state("step", (q, dq, quaternion, omega))
        values = np.asarray(reply.split(), dtype=np.float64)
        if values.size != 3 * len(self.joint_names) or not np.isfinite(values).all():
            raise RuntimeError("C++ policy returned invalid motor targets")
        return values.reshape(3, -1)

    def _send_state(self, command, observation):
        state = np.concatenate(observation)
        return self._request(command + " " + " ".join(map(str, state))).strip()

    def check_entry(self, observation):
        reply = self._send_state("check", observation)
        if reply == "ready":
            return None
        if reply == "unready":
            return "Wait for the robot to settle"
        if reply.startswith("fault "):
            return reply.removeprefix("fault ")
        raise RuntimeError(f"Unexpected hardware entry reply: {reply}")

    def motor_tick(self, command, observation):
        phase, _, payload = self._send_state(command, observation).partition(" ")
        if phase == "fault":
            return phase, payload
        values = np.asarray(payload.split(), dtype=np.float64)
        count = len(self.joint_names) * (3 if self.drive_pd else 1)
        if (phase not in ("damping", "rising", "standing", "timeout")
                or values.size != count or not np.isfinite(values).all()):
            raise RuntimeError("Invalid GetUpMimic hardware simulation output")
        return phase, values.reshape(3, -1) if self.drive_pd else values

    def telemetry(self):
        return self._request("telemetry").strip()


@contextmanager
def open_policy(project_root, config, hardware=False, expert=None, *, actuation=None):
    binary = project_root / "build/g1_getup_mimic"
    if not binary.is_file():
        raise FileNotFoundError(f"Missing {binary}. Build the g1_getup_mimic CMake target first")
    stream = "sim-hardware-stream" if hardware else "sim-stream"
    command = [binary, stream, "--config", config]
    if expert is not None:
        command.extend(("--expert", expert))
    if actuation is not None:
        command.extend(("--actuation", actuation))
    with subprocess.Popen(command, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, encoding="ascii", bufsize=1) as process:
        try:
            yield PolicyStream(process, hardware)
        finally:
            with suppress(BrokenPipeError):
                process.stdin.close()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


class GetUpSimulation:
    def __init__(self, mjcf, settings, policy):
        self.model = model = mj.MjModel.from_xml_path(str(mjcf.resolve()))
        self.policy = policy
        if model.nu:
            raise ValueError("Use the training bundle's G1 MJCF, which takes applied joint forces")
        columns = {name: index for index, name in enumerate(settings["joint_names"])}
        if len(settings["joint_names"]) != 29 or set(columns) != set(policy.joint_names):
            raise ValueError("Simulation settings must cover the 29 C++ policy motor names")
        self.pose_columns = [columns[name] for name in policy.joint_names]
        joints = [model.joint(name).id for name in policy.joint_names]
        if not np.all(model.jnt_type[joints] == mj.mjtJoint.mjJNT_HINGE):
            raise ValueError("The 29 policy motors require hinge joints in the MJCF")
        self.joint_q = model.jnt_qposadr[joints]
        self.joint_dq = model.jnt_dofadr[joints]

        pelvis = model.body("pelvis").id
        root_joint = model.body_jntadr[pelvis]
        if model.body_jntnum[pelvis] != 1 or model.jnt_type[root_joint] != mj.mjtJoint.mjJNT_FREE:
            raise ValueError("The MJCF pelvis requires a free joint")
        root_q, root_dq = model.jnt_qposadr[root_joint], model.jnt_dofadr[root_joint]
        self.root_position = slice(root_q, root_q + 3)
        self.root_quaternion = slice(root_q + 3, root_q + 7)
        # MuJoCo's free-joint angular velocity is already in the pelvis frame.
        self.root_omega = slice(root_dq + 3, root_dq + 6)

        model.opt.timestep = policy.control_period or settings["physics_dt"]
        self.substeps = round(policy.period / model.opt.timestep)
        if not math.isclose(self.substeps * model.opt.timestep, policy.period):
            raise ValueError("The physics step must divide the GetUpMimic policy period")
        model.opt.gravity[:] = (0, 0, -9.81)
        model.opt.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
        model.dof_armature[self.joint_dq] = np.asarray(settings["armature"])[self.pose_columns]
        model.dof_damping[self.joint_dq] = 0
        self.effort_limit = np.asarray(settings["effort_limit"])[self.pose_columns]
        self.data = mj.MjData(model)

    def reset(self, pose):
        self.policy.reset()
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[self.root_position] = pose["root_position"]
        self.data.qpos[self.root_quaternion] = pose["root_quaternion"]
        self.data.qpos[self.joint_q] = np.asarray(pose["joint_position"])[self.pose_columns]
        mj.mj_forward(self.model, self.data)

    def observation(self):
        data = self.data
        return (
            data.qpos[self.joint_q], data.qvel[self.joint_dq],
            data.qpos[self.root_quaternion], data.qvel[self.root_omega],
        )

    def step(self, policy_weight=1.0):
        target, kp, kd = self.policy.step(*self.observation())
        if policy_weight < 1.0:
            # Blend standby and policy torque through the gains. The model still sees
            # the live state and its own action history from the first tick.
            kp = policy_weight * kp
            kd = STANDBY_DAMPING + policy_weight * (kd - STANDBY_DAMPING)
        self.advance(target, kp, kd)
        return self.upright()

    def upright(self):
        data = self.data
        height = data.qpos[self.root_position][2]
        x, y = data.qpos[self.root_quaternion][1:3]
        return height, 1 - 2 * (x * x + y * y)

    def advance(self, target=0.0, kp=0.0, kd=0.0):
        expected_time = self.data.time + self.policy.period
        for _ in range(self.substeps):
            self.apply_forces(self.pd_torque(target, kp, kd, self.effort_limit))
        self.check_step(expected_time)

    def pd_torque(self, target, kp, kd, effort_limit):
        torque = kp * (target - self.data.qpos[self.joint_q]) - kd * self.data.qvel[self.joint_dq]
        return np.clip(torque, -effort_limit, effort_limit)

    def apply_forces(self, torques):
        self.data.qfrc_applied.fill(0)
        self.data.qfrc_applied[self.joint_dq] = torques
        mj.mj_step(self.model, self.data)

    def check_step(self, expected_time):
        data = self.data
        if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
                or not math.isclose(data.time, expected_time, rel_tol=0, abs_tol=1e-8)):
            raise RuntimeError(f"Invalid MuJoCo state at simulation time {expected_time:.2f} s")


class RecoveryRun:
    def __init__(self, simulation, pose, ramp_seconds=0.0):
        self.simulation = simulation
        self.label = f"{pose['kind']} row={pose['row']}"
        self.period = simulation.policy.period
        self.ramp_ticks = round(ramp_seconds / self.period)
        self.stable_ticks = round(STABLE_SECONDS / self.period)
        self.tick = self.held_ticks = 0
        self.first_stable = None
        self.height = simulation.data.qpos[simulation.root_position][2]

    @property
    def elapsed(self):
        return self.tick * self.period

    def step(self):
        progress = min(self.tick / self.ramp_ticks, 1.0) if self.ramp_ticks else 1.0
        policy_weight = progress * progress * (3.0 - 2.0 * progress)
        self.height, upright = self.simulation.step(policy_weight)
        self.tick += 1
        standing = self.height > STANDING_HEIGHT_M and upright > STANDING_UPRIGHT_COS
        self.held_ticks = self.held_ticks + 1 if standing else 0
        if self.held_ticks == self.stable_ticks and self.first_stable is None:
            self.first_stable = self.elapsed


class RecoverySession:
    def __init__(self, simulation, pose):
        self.simulation = simulation
        self.policy = simulation.policy
        self.pose = pose
        self.hardware = self.policy.control_period is not None
        self.period = self.policy.control_period if self.hardware else self.policy.period
        self.phase = "off"
        self.recovery = None
        self.started_at = None
        self.first_stable = None
        self.failure = None
        self.damping_until = 0.0

    def take_over(self):
        if self.phase != "off":
            return None
        if self.hardware:
            reason = self.policy.check_entry(self.simulation.observation())
            if reason:
                return f"Takeover blocked. {reason}"
        self.phase = "ready"
        return "Control enabled. Damping only (P to get up)"

    def begin(self):
        if self.phase != "ready":
            return None
        if self.hardware:
            reason = self.policy.check_entry(self.simulation.observation())
            if reason:
                return f"Get-up blocked. {reason}"
        self.policy.reset()
        self.phase = "running"
        self.started_at = self.simulation.data.time
        self.first_stable = self.failure = None
        if not self.hardware:
            self.recovery = RecoveryRun(self.simulation, self.pose, ramp_seconds=GETUP_RAMP_SECONDS)
        return f"Getting up from live pose at {self.started_at:.2f} s"

    def stop(self):
        self.recovery = None
        if self.phase == "fault":
            self.damping_until = self.simulation.data.time
        else:
            self.phase = "off"

    def advance(self):
        if self.hardware:
            return self.advance_hardware()
        if self.recovery is None:
            self.simulation.advance(kd=STANDBY_DAMPING if self.phase == "ready" else 0.0)
        else:
            self.recovery.step()
            if self.phase == "running":
                if self.recovery.first_stable is not None:
                    self.phase = "standing"
                    return f"Standing after {self.recovery.first_stable:.2f} s"
                if self.recovery.elapsed >= EPISODE_SECONDS:
                    self.recovery = None
                    self.phase = "ready"
                    return "Get-up timed out. Damping only (P to retry)"
        return None

    def advance_hardware(self):
        sim = self.simulation
        now = sim.data.time
        torque = 0.0
        message = None
        if self.phase in ("ready", "running", "standing"):
            command = "damping" if self.phase == "ready" else "getup"
            status, output = self.policy.motor_tick(command, sim.observation())
            if status == "fault":
                self.failure = output
                self.phase = "fault"
                self.damping_until = now + self.policy.damping_seconds
                message = f"Protection stop: {self.failure}. R/N resets the scene"
            else:
                torque = output
                if self.policy.drive_pd:
                    torque = sim.pd_torque(*output, effort_limit=self.policy.drive_effort)
                if status == "timeout":
                    self.phase = "ready"
                    self.failure = "get-up timed out"
                    message = "Get-up timed out. Damping only (P to retry)"
                elif status == "standing" and self.phase == "running":
                    self.phase = "standing"
                    self.first_stable = now - self.started_at
                    message = f"Standing after {self.first_stable:.2f} s"
        if self.phase == "fault":
            # Approximate the short drive-side damping burst with the simulator's joint velocity.
            torque = -self.policy.stop_kd * sim.data.qvel[sim.joint_dq] if now < self.damping_until else 0.0
        sim.apply_forces(torque)
        sim.check_step(now + self.period)
        return message


def pace_viewer(viewer, tick, deadline, sync_every=2):
    if tick % sync_every == 0:
        viewer.sync()
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


def run_episode(simulation, pose, viewer):
    recovery = RecoveryRun(simulation, pose)
    started = time.monotonic()
    for _ in range(round(EPISODE_SECONDS / recovery.period)):
        if viewer is not None and not viewer.is_running():
            raise RuntimeError(f"Viewer closed before {recovery.label} completed")
        with viewer.lock() if viewer is not None else nullcontext():
            recovery.step()
        if viewer is not None:
            pace_viewer(viewer, recovery.tick, started + recovery.elapsed)

    passed = recovery.held_ticks >= recovery.stable_ticks
    status = "PASS" if passed else "FAIL"
    first = f"{recovery.first_stable:.2f} s" if recovery.first_stable is not None else "never"
    print(f"{status} {recovery.label} | first stable: {first}"
          f" | final stand: {recovery.held_ticks * recovery.period:.2f} s"
          f" | final pelvis: {recovery.height:.3f} m", flush=True)
    return passed


def run_hardware_episode(simulation, pose, viewer):
    session = RecoverySession(simulation, pose)
    label = f"{pose['kind']} row={pose['row']}"
    period = session.period
    started = time.monotonic()
    settle_ticks = round(AUTO_LIMP_SECONDS / period)
    standby_ticks = round(AUTO_STANDBY_SECONDS / period)
    start_tick = settle_ticks + standby_ticks
    final_tick = start_tick + round(session.policy.timeout / period)
    sync_every = round(0.02 / period)
    held = 0
    for tick in range(final_tick + 1):
        if viewer is not None and not viewer.is_running():
            raise RuntimeError(f"Viewer closed before {label} completed")
        with viewer.lock() if viewer is not None else nullcontext():
            if tick == settle_ticks:
                message = session.take_over()
                if session.phase != "ready":
                    session.failure = message
                    break
            elif tick == start_tick:
                message = session.begin()
                if session.phase != "running":
                    session.failure = message
                    break
            session.advance()
            height, upright = simulation.upright()
            held = held + 1 if height > STANDING_HEIGHT_M and upright > STANDING_UPRIGHT_COS else 0
            if session.failure:
                break
        if viewer is not None:
            pace_viewer(viewer, tick, started + simulation.data.time, sync_every)

    height, _ = simulation.upright()
    passed = session.phase == "standing" and held * period >= STABLE_SECONDS
    status = "PASS" if passed else "FAIL"
    first = f"{session.first_stable:.2f} s" if session.first_stable is not None else "never"
    reason = session.failure or ("MuJoCo standing check failed" if not passed else "")
    if session.failure and session.started_at is not None:
        reason += f" at {simulation.data.time - session.started_at - period:.2f} s"
    print(f"{status} {label} | controller standing: {first} | final stand: {held * period:.2f} s"
          f" | final pelvis: {height:.3f} m" + (f" | {reason}" if reason else ""), flush=True)
    return passed


def open_viewer(stack, simulation, follow, key_callback=None):
    from mujoco import viewer as mj_viewer

    if follow:
        from viewer_camera import follow_robot

        stack.enter_context(follow_robot())
    viewer = mj_viewer.launch_passive(simulation.model, simulation.data, key_callback=key_callback)
    stack.callback(close_viewer, viewer)
    if not follow:
        with viewer.lock():
            viewer.cam.lookat[:] = (0, 0, 0.4)
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
    return viewer


def run_interactive(simulation, poses, viewer, keys, monitor=None, *, label="GetUpMimic"):
    import glfw

    pose_index = 0
    session = RecoverySession(simulation, poses[pose_index])
    scene_started = time.monotonic()
    tick = 0
    instructions = ("9 Take over | P Get up", "Space Stop | R Reset scene | N Next pose")
    limp = "Limp (9 to take over)"

    def show(message):
        pose = poses[pose_index]
        status = f"{pose['kind']} row={pose['row']} | {message}"
        print(status, flush=True)
        if hasattr(viewer, "set_texts"):
            text = "\n".join((f"{label} | {status}", *instructions))
            viewer.set_texts((None, None, text, None))

    print(f"{label} manual controls | " + " | ".join(instructions), flush=True)
    show(limp)
    while viewer.is_running():
        pressed = []
        with suppress(Empty):
            while True:
                pressed.append(keys.get_nowait())
        # Discard other keys queued for this tick when the operator requests a stop.
        if glfw.KEY_SPACE in pressed:
            pressed = [glfw.KEY_SPACE]

        for key in pressed:
            if key in (glfw.KEY_R, glfw.KEY_N):
                if key == glfw.KEY_N:
                    pose_index = (pose_index + 1) % len(poses)
                with viewer.lock():
                    simulation.reset(poses[pose_index])
                session = RecoverySession(simulation, poses[pose_index])
                scene_started = time.monotonic()
                tick = 0
                show(limp)
            elif key == glfw.KEY_SPACE:
                session.stop()
                message = limp if session.phase == "off" else "Limp (R/N resets the scene)"
                show(f"Stopped at {simulation.data.time:.2f} s. {message}")
            elif key in (glfw.KEY_9, glfw.KEY_KP_9):
                with viewer.lock():
                    message = session.take_over()
                if message:
                    show(message)
                if session.hardware and message:
                    break
            elif key == glfw.KEY_P:
                with viewer.lock():
                    message = session.begin()
                if message:
                    show(message)
                elif session.phase == "off":
                    show(limp)

        # Gravity and contact advance in every phase, including with zero motor torque.
        with viewer.lock():
            message = session.advance()
            if monitor is not None:
                phase = session.phase
                if phase == "fault" and simulation.data.time < session.damping_until:
                    phase = "fault-damping"
                monitor.update(simulation.data, phase, session.failure if session.phase == "fault" else None)
        if message:
            show(message)
        tick += 1
        pace_viewer(viewer, tick, scene_started + simulation.data.time, round(0.02 / session.period))
    return 0


def run(project_root, mjcf, kind, headless, follow, interactive=False, hardware=False, monitor_port=None,
        *, expert=None, actuation=None):
    if not mjcf.is_file():
        raise FileNotFoundError(f"MJCF file not found: {mjcf}")
    config = project_root / "apps/getup_mimic/config/g1_getup_mimic.json"
    label = "GetUpMimic"
    if expert is not None:
        profiles = json.loads(config.read_text())["model_test"]
        if expert not in profiles:
            raise ValueError(f"Unknown model_test expert {expert}. Choose from {', '.join(profiles)}")
        kind = kind or profiles[expert]["pose"]
        label = f"model_test {expert}"
        print(label, flush=True)
    kind = kind or "all"
    settings_path = config.with_name("g1_getup_mimic_sim.json")
    settings = json.loads(settings_path.read_text())
    poses = [pose for pose in settings["poses"] if kind == "all" or pose["kind"] == kind]
    if not poses:
        raise ValueError(f"No GetUpMimic presets for {kind}")

    passed = 0
    viewer = None
    with open_policy(project_root, config, hardware, expert, actuation=actuation) as policy, ExitStack() as stack:
        simulation = GetUpSimulation(mjcf, settings, policy)
        if hardware:
            mode = "drive_pd" if policy.drive_pd else "host_torque"
            label = f"{label} · {mode}"
            print(f"GetUpMimic MuJoCo hardware config | {mode} | policy {policy.period * 1000:g} ms"
                  f" / control and physics {policy.control_period * 1000:g} ms", flush=True)
        if interactive:
            monitor = None
            if monitor_port is not None:
                from joint_web_monitor import SimulationMonitor

                monitor = stack.enter_context(SimulationMonitor(
                    monitor_port, label, simulation.model, policy.joint_names,
                    simulation.joint_q, simulation.joint_dq, "qfrc_applied", policy.telemetry))
            simulation.reset(poses[0])
            keys = SimpleQueue()
            viewer = open_viewer(stack, simulation, follow, keys.put)
            return run_interactive(simulation, poses, viewer, keys, monitor, label=label)
        if hardware:
            print(f"{len(poses)} cases | limp {AUTO_LIMP_SECONDS:g} s, standby {AUTO_STANDBY_SECONDS:g} s,"
                  f" recovery {policy.timeout:g} s", flush=True)
        else:
            print(f"GetUpMimic MuJoCo: {len(poses)} cases, {EPISODE_SECONDS:g} s each, "
                  f"policy {policy.period * 1000:g} ms / physics {simulation.model.opt.timestep * 1000:g} ms", flush=True)
        run_case = run_hardware_episode if hardware else run_episode
        for pose in poses:
            with viewer.lock() if viewer is not None else nullcontext():
                simulation.reset(pose)
            if not headless and viewer is None:
                viewer = open_viewer(stack, simulation, follow)
            passed += run_case(simulation, pose, viewer)

    print(f"GetUpMimic recovery: {passed}/{len(poses)} passed", flush=True)
    return 0 if passed == len(poses) else 1
