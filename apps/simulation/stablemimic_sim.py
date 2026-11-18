"""Run StableMimic in MuJoCo with manual takeover or an automatic parity check."""

import argparse
from contextlib import ExitStack, suppress
import json
import math
from pathlib import Path
from queue import Empty, SimpleQueue
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import mujoco as mj
import numpy as np

from viewer_camera import close_viewer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "apps/stablemimic/config/g1_stablemimic.json"


def read_sample(sim, order, rotation_xyzw):
    """Raw world-frame state and the already aligned reference, in DDS joint order."""
    data, ref, frame = sim.data, sim.motion, sim.frame
    velocity = np.empty(6)
    mj.mj_objectVelocity(sim.model, data, mj.mjtObj.mjOBJ_BODY, sim.pelvis, velocity, 0)
    return {
        "q": data.qpos[sim.qpos_ids][order],
        "dq": data.qvel[sim.qvel_ids][order],
        "pelvis_rotation": data.xmat[sim.pelvis],
        "pelvis_omega_world": velocity[:3],
        "pelvis_velocity_world": velocity[3:],
        "torso_position_world": data.xpos[sim.torso],
        "torso_rotation": data.xmat[sim.torso],
        "reference_q": ref["joint_pos"][frame][order],
        "reference_dq": ref["joint_vel"][frame][order],
        "reference_torso_position_world": ref["body_pos_w"][frame, 1] + sim.xy_offset,
        "reference_torso_rotation": rotation_xyzw(ref["body_quat_w"][frame, 1]).reshape(-1),
    }


def place_fallen_robot(sim, library, index, order):
    # Keep the dance reference cursor. Only the simulated robot is reset.
    mj.mj_resetData(sim.model, sim.data)
    sim.data.qpos[:3] = [0.0, 0.0, library["root_heights"][index]]
    quaternion = library["root_quats"][index].astype(float)
    sim.data.qpos[3:7] = quaternion / np.linalg.norm(quaternion)
    sim.data.qpos[sim.qpos_ids] = library["poses"][index, order]
    sim.xy_offset[:2] = -sim.motion["body_pos_w"][sim.frame, 0, :2]
    sim.fallen = True
    mj.mj_forward(sim.model, sim.data)


class CheckedSession:
    def __init__(self, process, sim, policy_names, rotation_xyzw):
        self.process = process
        self.sim = sim
        self.python_session = sim.session
        self.rotation_xyzw = rotation_xyzw
        header = process.stdout.readline().split()
        names = header[2:]
        if header[:2] != ["g1-stablemimic-v1", "20"] or len(names) != 29 or set(names) != set(policy_names):
            raise RuntimeError("Unexpected StableMimic DDS joint order or policy period")
        self.dds_order = [policy_names.index(name) for name in names]
        self.policy_order = np.argsort(self.dds_order)
        self.errors = dict(obs=0.0, action=0.0, gate=0.0, target_rad=0.0)

    def reset(self):
        self.process.stdin.write("reset\n")
        self.process.stdin.flush()
        if self.process.stdout.readline().strip() != "ok":
            raise RuntimeError("C++ StableMimic did not acknowledge reset")

    def compare(self, name, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=3e-4,
                                   err_msg=f"C++ StableMimic {name} at frame {self.sim.frame}")
        self.errors[name] = max(self.errors[name], float(np.max(np.abs(actual - expected))))

    def run(self, outputs, inputs):
        # Sim2Sim prepares an independent Python reference before calling here.
        # Return C++ actions to its original PD and physics path.
        if list(outputs) != ["action", "gate_weights"]:
            raise RuntimeError("Unexpected StableMimic playback outputs")
        sample = read_sample(self.sim, self.dds_order, self.rotation_xyzw)
        line = "inspect " + " ".join(str(float(value)) for field in sample.values() for value in field)
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()
        values = np.asarray(self.process.stdout.readline().split(), dtype=np.float64)
        joint_count = len(self.dds_order)
        expected_policy = inputs["policy"][0]
        expected_gate_obs = inputs["gate"][0]
        action_end = joint_count
        gate_end = action_end + 2
        target_end = gate_end + joint_count
        policy_end = target_end + expected_policy.size
        if values.size != policy_end + expected_gate_obs.size or not np.isfinite(values).all():
            raise RuntimeError("C++ StableMimic returned invalid inference data")
        action = values[:action_end].astype(np.float32)[None]
        gate = values[action_end:gate_end].astype(np.float32)[None]
        target = values[gate_end:target_end][self.policy_order]
        self.compare("obs", values[target_end:policy_end], expected_policy)
        self.compare("obs", values[policy_end:], expected_gate_obs)
        expected_action, expected_gate = self.python_session.run(outputs, inputs)
        self.compare("action", action, expected_action)
        self.compare("gate", gate, expected_gate)
        clip = self.sim.action_clip
        clipped = expected_action[0] if clip is None else np.clip(expected_action[0], -clip, clip)
        self.compare("target_rad", target, self.sim.nominal + self.sim.scale * clipped)
        return action, gate


class OperatorControl:
    def __init__(self, sim, settings, substeps):
        self.sim = sim
        self.reference_start = sim.frame
        self.standby_kd = float(settings["standby_kd"])
        self.ramp_seconds = float(settings["ramp_seconds"])
        if any(not math.isfinite(value) or value < 0 for value in (self.standby_kd, self.ramp_seconds)):
            raise ValueError("StableMimic standby_kd and ramp_seconds must be finite and nonnegative")
        self.substeps = substeps
        self.phase = "limp"
        self.started_at = 0.0
        self.active_steps = 0

    def take_over(self):
        self.phase = "ready"
        return "Control enabled. Damping only"

    def reference_complete(self):
        return self.sim.frame >= self.sim.end

    def begin(self):
        sim = self.sim
        sim.session.reset()
        sim.action.fill(0.0)
        for history in sim.history.values():
            history.clear()
        sim.frame = self.reference_start
        sim.xy_offset.fill(0.0)
        sim.xy_offset[:2] = sim.data.xpos[sim.pelvis, :2] - sim.motion["body_pos_w"][sim.frame, 0, :2]
        sim.fallen = False
        sim.upright_ticks = 0
        self.started_at = sim.data.time
        self.active_steps = 0
        self.phase = "running"
        return f"Policy started from current pose at {sim.data.time:.2f}s"

    def stop(self):
        self.phase = "limp"
        self.sim.data.ctrl.fill(0.0)

    def advance(self):
        sim = self.sim
        if self.phase == "running":
            drive = sim.drive
            kp, kd = drive["stiffness"], drive["damping"]
            elapsed = sim.data.time - self.started_at
            progress = min(1.0, elapsed / self.ramp_seconds) if self.ramp_seconds else 1.0
            weight = progress * progress * (3.0 - 2.0 * progress)
            drive["stiffness"] = weight * kp
            drive["damping"] = self.standby_kd + weight * (kd - self.standby_kd)
            try:
                record = sim.step()
            finally:
                drive["stiffness"], drive["damping"] = kp, kd
            self.active_steps += 1
            return record

        # Continue floor contact and gravity in both passive phases. The MJCF
        # applies the StableMimic drive's actuator force limits to damping.
        kd = self.standby_kd if self.phase == "ready" else 0.0
        sim.data.ctrl.fill(0.0)
        expected_time = sim.data.time + self.substeps * sim.model.opt.timestep
        for _ in range(self.substeps):
            if kd:
                sim.data.ctrl[sim.motor_ids] = -kd * sim.data.qvel[sim.qvel_ids]
            mj.mj_step(sim.model, sim.data)
        if (not math.isclose(sim.data.time, expected_time, abs_tol=1e-9, rel_tol=0)
                or not np.isfinite(sim.data.qpos).all() or not np.isfinite(sim.data.qvel).all()):
            raise RuntimeError("Invalid StableMimic MuJoCo integration")
        mj.mj_forward(sim.model, sim.data)
        return None


class HardwareOperatorControl:
    def __init__(self, process, sim, policy_names, period):
        self.process = process
        self.sim = sim
        header = process.stdout.readline().split()
        names = header[3:]
        if header[:3] != ["g1-stablemimic-hardware-v1", "20", "2"] or len(names) != 29 or set(names) != set(policy_names):
            raise RuntimeError("Unexpected StableMimic hardware simulation handshake")
        self.joint_names = names
        self.dds_order = [policy_names.index(name) for name in names]
        self.policy_order = np.argsort(self.dds_order)
        sim.model.opt.timestep = int(header[2]) / 1000
        self.substeps = round(period / sim.model.opt.timestep)
        self.phase = "limp"
        self.active_steps = 0
        self.failure = None

    def request(self, command):
        if command in ("reset", "telemetry"):
            line = command
        else:
            sim, order = self.sim, self.dds_order
            data = sim.data
            velocity = np.empty(6)
            mj.mj_objectVelocity(sim.model, data, mj.mjtObj.mjOBJ_BODY, sim.pelvis, velocity, 0)
            state = np.concatenate((data.qpos[sim.qpos_ids][order], data.qvel[sim.qvel_ids][order],
                                    data.xpos[sim.pelvis], data.xmat[sim.pelvis], velocity))
            line = command + " " + " ".join(map(str, state))
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline().strip()
        if not response:
            raise RuntimeError("StableMimic hardware simulation stream exited")
        return response

    def entry_error(self):
        reply = self.request("check")
        if reply == "ready":
            return None
        if reply == "unready":
            return "Wait for the robot to settle"
        if reply.startswith("fault "):
            return reply.removeprefix("fault ")
        raise RuntimeError("Unexpected StableMimic hardware entry response")

    def take_over(self):
        error = self.entry_error()
        if error:
            return f"Takeover blocked. {error}"
        self.phase = "ready"
        return "Control enabled. Damping only"

    def begin(self):
        error = self.entry_error()
        if error:
            return f"Start blocked. {error}"
        if self.request("reset") != "ok":
            raise RuntimeError("C++ StableMimic did not acknowledge hardware reset")
        self.phase = "running"
        self.active_steps = 0
        return f"Policy started from current pose at {self.sim.data.time:.2f}s"

    def stop(self):
        self.phase = "limp"
        self.failure = None
        self.sim.data.ctrl.fill(0)

    def reference_complete(self):
        # The real controller tracks the standing reference after the final motion frame.
        return False

    def advance(self):
        sim, record = self.sim, None
        data = sim.data
        now = data.time
        peak_torque = 0.0
        for _ in range(self.substeps):
            data.ctrl.fill(0)
            if self.phase in ("ready", "running"):
                command = "track" if self.phase == "running" else "damping"
                reply = self.request(command)
                if reply.startswith("fault "):
                    self.failure = reply.removeprefix("fault ")
                    self.phase = "fault"
                else:
                    kind, *values = reply.split()
                    if kind != command:
                        raise RuntimeError("Unexpected StableMimic hardware command response")
                    if kind == "track":
                        frame, realigned, holding, tracking, recovery, *values = values
                        if record is None:
                            record = {"time": now, "frame": int(frame), "xy_realigned": realigned == "1",
                                      "standing_reference": holding == "1", "gate": [float(tracking), float(recovery)],
                                      "pelvis_height": float(data.xpos[sim.pelvis, 2]),
                                      "pelvis_tilt_deg": math.degrees(math.acos(np.clip(data.xmat[sim.pelvis, 8], -1, 1)))}
                    torque = np.asarray(values, dtype=float)
                    if torque.shape != (29,) or not np.isfinite(torque).all():
                        raise RuntimeError("Invalid StableMimic hardware torque response")
                    data.ctrl[sim.motor_ids] = torque[self.policy_order]
            mj.mj_step(sim.model, data)
            peak_torque = max(peak_torque, float(np.abs(data.qfrc_actuator[sim.qvel_ids]).max()))
            mj.mj_forward(sim.model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("Invalid StableMimic hardware MuJoCo integration")
        if record is not None and not self.failure:
            record["peak_torque_nm"] = peak_torque
            self.active_steps += 1
            return record
        return None


def run_operator(sim, control, period, max_steps, log_step, monitor=None):
    import glfw
    from mujoco.viewer import launch_passive

    keys = SimpleQueue()
    completed, last = 0, None
    with ExitStack() as window:
        viewer = launch_passive(sim.model, sim.data, key_callback=keys.put)
        window.callback(close_viewer, viewer)
        with viewer.lock():
            viewer.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = sim.pelvis
            viewer.cam.distance = 3.0
        keys_text = "9 Take over | P Start | Space Stop"

        def show(status):
            print(f"StableMimic: {status} | {keys_text}", flush=True)
            if hasattr(viewer, "set_texts"):
                viewer.set_texts((None, None, f"StableMimic | {status}\n{keys_text}", None))

        show("Limp")
        deadline = time.monotonic()
        while viewer.is_running():
            pressed = set()
            with suppress(Empty):
                while True:
                    pressed.add(keys.get_nowait())
            message = None
            with viewer.lock():
                if glfw.KEY_SPACE in pressed:
                    control.stop()
                    message = "Stopped. Limp"
                elif control.phase == "limp" and {glfw.KEY_9, glfw.KEY_KP_9} & pressed:
                    message = control.take_over()
                elif control.phase == "ready" and glfw.KEY_P in pressed:
                    message = control.begin()
                phase = control.phase
                record = control.advance()
                if monitor is not None:
                    monitor.update(sim.data, control.phase, control.failure)
                if phase != "fault" and control.phase == "fault":
                    message = f"Protection stop: {control.failure}. Space acknowledges"
                elif record is not None:
                    completed += 1
                    last = record
                    message = log_step(record) or message
                    if control.reference_complete() or (max_steps is not None and control.active_steps >= max_steps):
                        control.phase = "ready"
                        message = "Run complete. Damping only"
            if message:
                show(message)
            viewer.sync()
            deadline += period
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                deadline = time.monotonic()
    return completed, last


def run(args):
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    settings = config["simulation"]
    stablemimic_root = args.stablemimic_root.resolve()
    scripts = stablemimic_root / "scripts"
    if not (scripts / "play_moe_mujoco.py").is_file():
        raise FileNotFoundError(f"Missing StableMimic player under {scripts}")
    sys.path.insert(0, str(scripts))
    from play_moe_mujoco import CONTROL_DT, POLICY_METADATA, SUBSTEPS, Sim2Sim, rotation_xyzw

    binary = PROJECT_ROOT / "build/g1_stablemimic"
    if not binary.is_file():
        raise FileNotFoundError(f"Missing {binary}. Build the g1_stablemimic CMake target first")
    model = args.onnx.resolve() if args.onnx is not None else (config_path.parent / config["policy_path"]).resolve()
    steps = args.steps if args.steps is not None else 1000
    if args.motion_file:
        sim = Sim2Sim(model, args.motion_file.resolve(), args.start_frame)
        if not args.hardware_config:
            steps = min(steps, sim.end - args.start_frame)
    else:
        from evaluate_getup_mujoco import StandingSim

        sim = StandingSim(model, stablemimic_root / settings["standing_command"])
    metadata = json.loads(sim.session.get_modelmeta().custom_metadata_map[POLICY_METADATA])
    if args.library:
        from evaluate_getup_mujoco import load_library

        library_path = stablemimic_root / settings["pose_libraries"][args.library]
        library, _, order = load_library(library_path, metadata["joint_names"])
        if not 0 <= args.pose_index < len(library["poses"]):
            raise ValueError(f"Invalid {args.library} pose index: {args.pose_index}")
        if args.motion_file:
            place_fallen_robot(sim, library, args.pose_index, order)
        else:
            sim.reset_pose(library, args.pose_index, order)

    realignments = []
    with ExitStack() as resources:
        trace = resources.enter_context(args.trace.open("x", encoding="utf-8", buffering=1)) if args.trace else None
        mode = "sim-hardware-stream" if args.hardware_config else "sim-stream"
        command = [binary, mode, "--config", config_path, "--model", model]
        if args.hardware_config and args.motion_file:
            sys.path.insert(0, str(PROJECT_ROOT / "apps/stablemimic/scripts"))
            from export_reference import export_reference

            temp = Path(resources.enter_context(TemporaryDirectory(prefix="stablemimic-reference-")))
            motion = temp / "motion.txt"
            export_reference(args.motion_file.resolve(), motion, standing=False)
            command += ["--motion", motion, "--start-frame", str(args.start_frame)]
        process = resources.enter_context(subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding="ascii", bufsize=1))

        def log_step(record):
            if trace is not None:
                trace.write(json.dumps(record, allow_nan=False) + "\n")
            if record["xy_realigned"]:
                realignments.append({"time": record["time"], "frame": record["frame"], "gate": record["gate"]})
                if args.motion_file:
                    return f"Upright. Dance re-aligned at frame {record['frame']}"
            return None

        try:
            if args.hardware_config:
                control = HardwareOperatorControl(process, sim, metadata["joint_names"], CONTROL_DT)
            else:
                sim.session = CheckedSession(process, sim, metadata["joint_names"], rotation_xyzw)
                if args.viewer:
                    control = OperatorControl(sim, settings, SUBSTEPS)
            last = None
            completed = 0
            if args.viewer:
                monitor = None
                if args.monitor is not None:
                    from joint_web_monitor import SimulationMonitor

                    monitor = resources.enter_context(SimulationMonitor(
                        args.monitor, "StableMimic", sim.model, control.joint_names,
                        sim.qpos_ids[control.dds_order], sim.qvel_ids[control.dds_order],
                        "qfrc_actuator", lambda: control.request("telemetry")))
                completed, last = run_operator(sim, control, CONTROL_DT, args.steps, log_step, monitor)
            elif args.hardware_config:
                for _ in range(round(2 / CONTROL_DT)):
                    control.advance()
                print(f"StableMimic: {control.take_over()}", file=sys.stderr)
                if control.phase != "ready":
                    raise RuntimeError("Hardware takeover rejected")
                for _ in range(round(0.5 / CONTROL_DT)):
                    control.advance()
                    if control.failure:
                        raise RuntimeError(f"Protection stop: {control.failure}")
                print(f"StableMimic: {control.begin()}", file=sys.stderr)
                if control.phase != "running":
                    raise RuntimeError("Hardware policy start rejected")
                for _ in range(steps):
                    last = control.advance()
                    if control.failure:
                        raise RuntimeError(f"Protection stop: {control.failure}")
                    log_step(last)
                    completed += 1
            else:
                for _ in range(steps):
                    last = sim.step()
                    log_step(last)
                    completed += 1
            report = {"control_steps": completed, "sim_time": sim.data.time, "last": last}
            if args.hardware_config:
                report.update({"motor_period_ms": 1000 * sim.model.opt.timestep,
                               "policy_period_ms": 1000 * CONTROL_DT, "fault": control.failure})
            else:
                report["max_cpp_python_error"] = sim.session.errors
            if args.library and not args.motion_file and not args.hardware_config:
                report["standing_metrics"] = sim.metrics()
            if args.motion_file:
                report["xy_realignments"] = realignments
            print(json.dumps(report, allow_nan=False))
        finally:
            with suppress(BrokenPipeError):
                process.stdin.close()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stablemimic-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--onnx", type=Path, help="Override the ONNX path in the StableMimic config")
    parser.add_argument("--motion-file", type=Path, help="50 Hz reference motion, including during recovery")
    parser.add_argument("--library", choices=("backup", "faceup"), help="Spawn in the selected fallen-pose library")
    parser.add_argument("--pose-index", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--steps", type=int, help="Maximum policy steps per run (headless default: 1000)")
    parser.add_argument("--trace", type=Path, help="Save per-step reference frames and Gate weights to a new JSONL file")
    parser.add_argument("--viewer", action="store_true", help="Open MuJoCo with manual 9, P and Space controls")
    parser.add_argument("--hardware-config", action="store_true",
                        help="Use the real StableMimic control period, motion protection and reference playback")
    parser.add_argument("--monitor", nargs="?", const=8765, type=int, metavar="PORT",
                        help="Serve the joint dashboard on localhost (default port 8765)")
    args = parser.parse_args()
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.monitor is not None and not (args.hardware_config and args.viewer and 0 <= args.monitor <= 65535):
        parser.error("--monitor requires --hardware-config and --viewer, with a valid port")
    if not (args.library or args.motion_file):
        parser.error("Specify --library, --motion-file, or both")
    if (args.start_frame and not args.motion_file) or (args.pose_index and not args.library):
        parser.error("--pose-index requires --library. --start-frame requires --motion-file")
    try:
        run(args)
    except (AssertionError, KeyError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"StableMimic simulation: {error}\n")


if __name__ == "__main__":
    main()
