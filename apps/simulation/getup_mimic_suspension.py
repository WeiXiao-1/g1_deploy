"""Unilateral spring/damper winch attached to the GetUpMimic torso.

Forces are applied at a body-fixed attachment through MuJoCo's point Jacobian.
The cable can pull only; it does not fix body orientation or overwrite qpos.
"""
import math

import mujoco as mj
import numpy as np


class TorsoSuspension:
    def __init__(self, model, data, *, speed=0.08):
        if not math.isfinite(speed) or not 0 < speed <= 0.2:
            raise ValueError("Suspension winch speed must be in (0, 0.2] m/s")
        self.model = model
        self.body = model.body("torso_link").id
        self.local_point = np.array([0.0, 0.0, 0.20])
        self.weight = float(model.body_subtreemass[model.body("pelvis").id]) * 9.81
        self.stiffness = 3000.0
        self.damping = 220.0
        self.speed = speed
        self.jacobian = np.zeros((3, model.nv))
        self.tension = 0.0
        self.reset(data)

    def point(self, data):
        return data.xpos[self.body] + data.xmat[self.body].reshape(3, 3) @ self.local_point

    def reset(self, data):
        self.anchor = self.point(data) + np.array([0.0, 0.0, 1.5])
        # Preload supports body weight without an initial free fall.
        self.raised_length = 1.5 - self.weight / self.stiffness
        self.length = self.target_length = self.raised_length
        self.tension = self.weight
        self.grounded_seconds = 0.0
        self.motion = "raised"

    def lower(self):
        self.target_length = max(self.length, self.target_length) + 0.10
        self.grounded_seconds = 0.0
        self.motion = "lowering"

    def raise_body(self):
        self.target_length = max(0.10, min(self.length, self.target_length) - 0.10)
        self.grounded_seconds = 0.0
        self.motion = "raising"

    def apply(self, data, dt):
        difference = self.target_length - self.length
        payout = float(np.clip(difference, -self.speed * dt, self.speed * dt))
        self.length += payout
        point = self.point(data)
        offset = point - self.anchor
        distance = float(np.linalg.norm(offset))
        if distance < 1e-9:
            self.tension = 0.0
            return
        away = offset / distance
        mj.mj_jac(self.model, data, self.jacobian, None, point, self.body)
        extension = distance - self.length
        stretch_rate = float(away @ (self.jacobian @ data.qvel)) - payout / dt
        self.tension = (float(np.clip(self.stiffness * extension + self.damping * stretch_rate,
                                     0.0, 2.5 * self.weight)) if extension > 0 else 0.0)
        mj.mj_applyFT(self.model, data, -self.tension * away, np.zeros(3), point,
                     self.body, data.qfrc_applied)

    def bearing_on_both_feet(self, data):
        feet = set()
        for contact in data.contact[:data.ncon]:
            names = [self.model.geom(i).name or "" for i in (contact.geom1, contact.geom2)]
            if "floor" in names:
                for side in ("left", "right"):
                    if any(name.startswith(side + "_foot") for name in names):
                        feet.add(side)
        return len(feet) == 2 and self.tension < 0.85 * self.weight

    def after_step(self, data, dt):
        self.grounded_seconds = (self.grounded_seconds + dt
                                 if self.bearing_on_both_feet(data) else 0.0)
        if self.motion == "lowering" and abs(self.target_length - self.length) < 1e-9:
            self.motion = "lowered"
        elif self.motion == "raising" and abs(self.target_length - self.length) < 1e-9:
            self.motion = "raised"

    def grounded(self, data):
        return self.grounded_seconds >= 0.3 and self.bearing_on_both_feet(data)

    def draw(self, viewer, data):
        scene = viewer.user_scn
        scene.ngeom = 1
        geom = scene.geoms[0]
        mj.mjv_initGeom(geom, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                        np.eye(3).ravel(), np.array([0.65, 0.35, 0.1, 1.0]))
        mj.mjv_connector(geom, mj.mjtGeom.mjGEOM_CAPSULE, 0.008, self.anchor, self.point(data))

    def status(self):
        return (f"rope {self.tension:.0f} N, length {self.length:.2f} m "
                f"(target {self.target_length:.2f} m), {self.motion}")
