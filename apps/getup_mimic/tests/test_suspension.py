"""Physics-independent suspension invariants using a tiny MuJoCo model."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'simulation'))
try:
    import mujoco as mj
    import numpy as np
    from getup_mimic_suspension import TorsoSuspension
except ImportError:
    mj = None


@unittest.skipIf(mj is None, 'MuJoCo/NumPy not installed in this Python environment')
class SuspensionChecks(unittest.TestCase):
    def setUp(self):
        self.model = mj.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="pelvis" pos="0 0 1"><freejoint/><geom type="sphere" size="0.1" mass="5"/>
            <body name="torso_link" pos="0 0 0.2"><geom type="sphere" size="0.1" mass="5"/>
            </body></body></worldbody></mujoco>''')
        self.data = mj.MjData(self.model)
        mj.mj_forward(self.model,self.data)
        self.rope = TorsoSuspension(self.model,self.data)

    def test_preload_supports_weight_and_slack_cable_does_not_push(self):
        self.rope.apply(self.data,0.002)
        self.assertAlmostEqual(self.rope.tension,98.1,places=6)
        self.assertAlmostEqual(self.data.qfrc_applied[2],98.1,places=6)
        self.rope.length = self.rope.target_length = 2.0
        self.data.qfrc_applied[:] = 0
        self.rope.apply(self.data,0.002)
        self.assertEqual(self.rope.tension,0)
        self.assertTrue(np.all(self.data.qfrc_applied==0))

    def test_winch_rate_and_force_cap(self):
        self.rope.lower()
        old=self.rope.length
        self.rope.apply(self.data,0.002)
        self.assertLessEqual(abs(self.rope.length-old),0.08*0.002+1e-12)
        self.data.qpos[0]=1.0
        mj.mj_forward(self.model,self.data)
        self.data.qfrc_applied[:]=0
        self.rope.apply(self.data,0.002)
        self.assertLess(self.data.qfrc_applied[0],0)
        self.assertLessEqual(self.rope.tension,2.5*self.rope.weight)
        self.assertFalse(self.rope.grounded(self.data))

    def test_each_press_adds_ten_centimeters_without_contact_auto_stop(self):
        initial=self.rope.target_length
        self.rope.lower()
        self.assertAlmostEqual(self.rope.target_length-initial,0.1)
        self.rope.lower()
        self.assertAlmostEqual(self.rope.target_length-initial,0.2)
        goal=self.rope.target_length
        self.rope.bearing_on_both_feet=lambda data: True
        self.rope.after_step(self.data,0.5)
        self.assertEqual(self.rope.target_length,goal)
        self.assertEqual(self.rope.motion,'lowering')
        for _ in range(1251):
            self.rope.apply(self.data,0.002)
            self.rope.after_step(self.data,0.002)
        self.assertAlmostEqual(self.rope.length,goal)
        self.assertEqual(self.rope.motion,'lowered')

    def test_lower_interrupts_raising_by_paying_out_from_current_length(self):
        self.rope.length += 0.4
        current=self.rope.length
        self.rope.raise_body()
        self.rope.lower()
        self.assertAlmostEqual(self.rope.target_length,current+0.1)

    def test_visible_rope_geometry_does_not_require_a_viewer_window(self):
        scene=mj.MjvScene(self.model,maxgeom=1)
        self.rope.draw(SimpleNamespace(user_scn=scene),self.data)
        self.assertEqual(scene.ngeom,1)
        self.assertGreater(scene.geoms[0].size[2],0)


if __name__=='__main__':
    unittest.main()
