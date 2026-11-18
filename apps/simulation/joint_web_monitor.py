"""Serve the existing read-only joint dashboard from a MuJoCo control loop."""

from pathlib import Path
import sys
import time
from types import SimpleNamespace

import mujoco as mj
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "monitor"))
from joint_monitor import JointMonitor, read_imu_state


class SimulationMonitor:
    def __init__(self, port, controller, model, names, q_ids, dq_ids, force_field, read_targets):
        self.controller = controller
        self.model = model
        self.pelvis = model.body("pelvis").id
        self.torso = model.body("torso_link").id
        self.force_field = force_field
        self.read_targets = read_targets
        self.server = JointMonitor("127.0.0.1", port)
        if tuple(names) != tuple(self.server.names):
            self.server.close()
            raise RuntimeError("Unexpected DDS joint order for the MuJoCo web monitor")
        self.q_ids, self.dq_ids = q_ids, dq_ids
        self.next_sample = 0.0
        self.last_phase = None
        self.body_velocity = np.zeros(6)

    def __enter__(self):
        self.server.start()
        return self

    def __exit__(self, *_):
        self.server.close()

    def imu(self, data, body):
        mj.mj_objectVelocity(self.model, data, mj.mjtObj.mjOBJ_BODY, body, self.body_velocity, 1)
        return SimpleNamespace(quaternion=data.xquat[body], gyroscope=self.body_velocity[:3].copy())

    def update(self, data, phase, fault=None):
        now = time.monotonic()
        if now < self.next_sample and phase == self.last_phase:
            return
        enabled = phase in {"ready", "running", "standing", "fault-damping"}
        target = None
        if enabled and phase != "fault-damping":
            reply = self.read_targets()
            if reply != "none":
                values = np.asarray(reply.split(), dtype=float)
                if values.size != 87 or not np.isfinite(values).all():
                    raise RuntimeError("Invalid MuJoCo monitor target. Rebuild the controller")
                target = values.reshape(3, -1)
        joints = []
        torque = getattr(data, self.force_field)
        for index, (name, q_id, dq_id) in enumerate(zip(self.server.names, self.q_ids, self.dq_ids)):
            q, dq, tau = float(data.qpos[q_id]), float(data.qvel[dq_id]), float(torque[dq_id])
            goal, kp, kd = (0.0, 0.0, 0.0) if target is None else map(float, target[:, index])
            joints.append(dict(index=index, name=name, q=q, dq=dq, tau=tau, q_target=goal, dq_target=0.0,
                               kp=kp, kd=kd, error=goal-q, target_available=target is not None, enabled=enabled,
                               active=enabled and (tau != 0.0 or kp != 0.0 or kd != 0.0)))
        self.server.on_torso_imu(self.imu(data, self.torso))
        self.server.publish_snapshot(dict(joints=joints, simulation_time=float(data.time), output_enabled=enabled,
                                          dry_run=False, imu=read_imu_state(self.imu(data, self.pelvis)),
                                          simulation=dict(controller=self.controller, phase=phase, fault=fault)), now)
        self.next_sample = now + self.server.period
        self.last_phase = phase
