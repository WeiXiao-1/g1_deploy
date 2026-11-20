"""Offline C++ reset state-machine checks; no SDK, DDS or robot connection.
Run after building g1_getup_mimic: python3 -m unittest discover -s apps/getup_mimic/tests -v
"""
import json
import math
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / 'apps/getup_mimic/config/g1_getup_mimic.json'
POSES = json.loads(CONFIG.with_name('g1_getup_mimic_reset.json').read_text())


class Stream:
    def __init__(self, pose='recovered', mode='drive_pd', *, delay_ms=None, binary=None):
        command = [str(binary or ROOT/'build/g1_getup_mimic'), 'sim-hardware-stream', '--config', str(CONFIG),
                   '--reset-pose', pose, '--actuation', mode]
        if delay_ms is not None:
            command.extend(('--inference-delay-ms', str(delay_ms)))
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.header = self.process.stdout.readline()
        if not self.header:
            raise RuntimeError(self.process.stderr.read())

    def send(self, command, q=None):
        if q is not None:
            command += ' ' + ' '.join(map(str, [*q, *([0]*29), 1, 0, 0, 0, 0, 0, 0]))
        self.process.stdin.write(command+'\n')
        self.process.stdin.flush()
        return self.process.stdout.readline().strip()

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=5)
        self.process.stdout.close()
        self.process.stderr.close()


class ResetChecks(unittest.TestCase):
    def complete_reset(self, selected):
        stream = Stream(selected)
        self.addCleanup(stream.close)
        q = list(POSES['profiles']['recorded' if selected == 'recovered' else 'recovered']['joint_position'])
        start = q.copy()
        goal = POSES['profiles'][selected]['joint_position']
        previous = q.copy()
        for tick in range(7000):
            reply = stream.send('reset-pose' if tick == 0 else 'hold', q).split()
            self.assertNotEqual(reply[0], 'fault', reply)
            values = list(map(float, reply[1:]))
            self.assertEqual(len(values), 87)
            q = values[:29]  # Ideal tracking plant isolates interpolation/control state.
            self.assertTrue(all(math.isfinite(v) for v in values))
            self.assertLessEqual(max(abs(a-b) for a,b in zip(q,previous)), 0.35*0.002+2e-6)
            if tick == 0:
                self.assertLess(max(abs(a-b) for a,b in zip(q,start)), 1e-6)
                self.assertEqual(max(values[29:58]), 0.0)
            previous = q.copy()
            if reply[0] == 'holding':
                self.assertLess(max(abs(a-b) for a,b in zip(q,goal)), 1e-6)
                return stream, q, values
        self.fail('Reset did not complete under ideal tracking')

    def test_all_poses_speed_limit_hold_and_policy_handover(self):
        for pose in POSES['profiles']:
            with self.subTest(pose=pose):
                stream,q,held = self.complete_reset(pose)
                self.assertEqual(stream.send('check-reset', q), 'ready')
                first = stream.send('getup', q).split()
                self.assertIn(first[0], ['rising', 'standing'])
                values = list(map(float, first[1:]))
                # At transition weight zero, preserve the full reset hold, not damping.
                self.assertLess(max(abs(a-b) for a,b in zip(values,held)), 1e-5)

    def test_preview_clear_stop_and_repeated_reset(self):
        stream = Stream()
        self.addCleanup(stream.close)
        q = POSES['profiles']['recorded']['joint_position']
        self.assertTrue(stream.send('reset-pose',q).startswith('resetting '))
        self.assertIn('interpolation to finish', stream.send('check-reset',q))
        self.assertEqual(stream.send('reset'), 'ok')
        self.assertEqual(stream.send('telemetry'), 'none')
        self.assertTrue(stream.send('reset-pose',q).startswith('resetting '))

    def test_drift_allows_policy_after_a_completed_reset(self):
        stream,q,_ = self.complete_reset('recovered')
        q[0] += 0.2
        reason = stream.send('check-reset',q)
        self.assertTrue(reason.startswith('unready Reset pose error at left_hip_pitch_joint'), reason)
        self.assertTrue(stream.send('getup',q).startswith('rising '))

    def test_timed_holding_without_tracking_does_not_timeout_and_allows_policy(self):
        stream = Stream()
        self.addCleanup(stream.close)
        q = POSES['profiles']['recorded']['joint_position']
        first_holding = None
        held = None
        for tick in range(7600):
            reply = stream.send('reset-pose' if tick == 0 else 'hold', q).split()
            self.assertNotEqual(reply[0], 'fault', reply)
            if reply[0] == 'holding':
                if first_holding is None:
                    first_holding = tick
                    held = reply[1:]
                self.assertEqual(reply[1:], held)  # Keep fixed goals/gains indefinitely.
            elif first_holding is not None:
                self.fail('Timed Holding was lost')
        self.assertEqual(first_holding, 2000)  # Four seconds, independent of actual tracking.
        self.assertTrue(all(float(v) > 0 for v in held[29:58]))
        reason = stream.send('check-reset', q)
        self.assertTrue(reason.startswith('unready Reset pose error at '), reason)
        self.assertTrue(stream.send('getup', q).startswith('rising '))

    def test_policy_interrupts_incomplete_reset_and_preserves_blend(self):
        stream = Stream()
        self.addCleanup(stream.close)
        q = POSES['profiles']['recorded']['joint_position']
        for tick in range(101):
            reply = stream.send('reset-pose' if tick == 0 else 'hold', q)
            self.assertTrue(reply.startswith('resetting '), reply)
        self.assertIn('interpolation to finish', stream.send('check-reset', q))
        dq = [0.0] * 29
        dq[4] = 2.0  # Also exceed the former joint/gyro entry speed thresholds.
        state = [*q, *dq, 1, 0, 0, 0, 0, 0, 1.2]
        first = stream.send('getup ' + ' '.join(map(str, state))).split()
        self.assertIn(first[0], ('rising', 'standing'))
        values = list(map(float, first[1:]))
        self.assertTrue(all(math.isfinite(v) for v in values))
        self.assertTrue(any(v > 0 for v in values[29:58]))

    def test_motion_check_reports_actual_cause(self):
        stream = Stream()
        self.addCleanup(stream.close)
        q = POSES['profiles']['recovered']['joint_position']
        dq = [0.0] * 29
        dq[4] = 2.0
        values = [*q, *dq, 1, 0, 0, 0, 0, 0, 0]
        reason = stream.send('check ' + ' '.join(map(str, values)))
        self.assertIn('Joint still moving: left_ankle_pitch_joint', reason)
        values = [*q, *([0]*29), 1, 0, 0, 0, 0, 0, 1.2]
        reason = stream.send('check ' + ' '.join(map(str, values)))
        self.assertIn('Pelvis angular speed=', reason)

    def test_invalid_profile_is_rejected_before_stream_start(self):
        process = subprocess.run(
            [str(ROOT/'build/g1_getup_mimic'),'sim-hardware-stream','--config',str(CONFIG),
             '--reset-pose','unknown'],capture_output=True,text=True,timeout=5)
        self.assertNotEqual(process.returncode,0)
        self.assertIn('Unknown reset pose',process.stderr)


if __name__ == '__main__':
    unittest.main()
