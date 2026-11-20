"""Offline checks for simulated inference completion, feedback and cancellation."""
import math
import unittest

from test_reset import POSES, Stream


class InferenceDelayChecks(unittest.TestCase):
    def stream(self, **kwargs):
        stream = Stream(**kwargs)
        self.addCleanup(stream.close)
        return stream

    def tick(self, stream, q):
        reply = stream.send('getup', q).split()
        self.assertIn(reply[0], ('rising', 'standing'), reply)
        return list(map(float, reply[1:]))

    def near(self, a, b, tolerance=2e-5):
        self.assertEqual(len(a), len(b))
        self.assertLess(max(abs(x-y) for x,y in zip(a,b)), tolerance)

    def test_explicit_zero_matches_default(self):
        for mode in ('drive_pd', 'host_torque'):
            implicit = self.stream(mode=mode)
            explicit = self.stream(mode=mode, delay_ms=0)
            for tick in range(120):
                q = list(POSES['profiles']['recorded']['joint_position'])
                q[0] += .02*math.sin(tick*.1)
                self.assertEqual(implicit.send('getup', q), explicit.send('getup', q))

    def test_nine_ms_delays_results_not_observations_or_physics_feedback(self):
        delayed = self.stream(delay_ms=9)
        oracle = self.stream(delay_ms=0)
        host = self.stream(delay_ms=9, mode='host_torque')
        first = second = None
        effort = list(map(float, delayed.header.split()[6:35]))
        for tick in range(11):
            q = list(POSES['profiles']['recorded']['joint_position'])
            q[0] += .03*tick  # Different state while the first inference waits.
            expected = self.tick(oracle, q)
            if tick == 0:
                first = expected[:29]
            if tick == 5:
                second = expected[:29]
            result = self.tick(delayed, q)
            host_torque = self.tick(host, q)
            if tick < 5:  # 0, 2, 4, 6, 8 ms: keep standby damping.
                self.assertEqual(result[29:58], [0.0]*29)
                self.near(result[:29], q)
            else:
                self.near(result[:29], first if tick < 10 else second)
                if tick == 5:  # 9 ms completes; first consumed at 10 ms.
                    self.assertEqual(result[29:58], [0.0]*29)
                else:
                    self.assertGreater(max(result[29:58]), 0)
            # Host torque must use current q, not the inference launch-time q.
            computed = [max(-limit, min(limit, kp*(goal-actual)))
                        for goal,kp,actual,limit in zip(result[:29],result[29:58],q,effort)]
            self.near(host_torque, computed, 3e-4)
        self.assertGreater(max(abs(a-b) for a,b in zip(first,second)), 1e-3)

    def test_slow_inference_skips_missed_policy_deadline(self):
        delayed = self.stream(delay_ms=11)
        oracle = self.stream(delay_ms=0)
        q = POSES['profiles']['recorded']['joint_position']
        targets = [self.tick(oracle,q)[:29] for _ in range(6)]
        first,second = targets[0],targets[5]
        for tick in range(17):
            result = self.tick(delayed,q)
            if tick < 6:
                self.assertEqual(result[29:58], [0.0]*29)
            else:
                # Starts at 0/20 ms, completes at 11/31, consumed at 12/32.
                self.near(result[:29], first if tick < 16 else second)

    def test_reset_discards_pending_inference_and_restarts_history(self):
        delayed = self.stream(delay_ms=9)
        q_old = POSES['profiles']['recovered']['joint_position']
        q_new = POSES['profiles']['recorded']['joint_position']
        self.tick(delayed,q_old)
        self.assertEqual(delayed.send('reset'), 'ok')
        expected = self.tick(self.stream(),q_new)[:29]
        for tick in range(6):
            result = self.tick(delayed,q_new)
            if tick < 5:
                self.assertEqual(result[29:58], [0.0]*29)
            else:
                self.near(result[:29], expected)

    def test_reset_hold_continues_until_first_delayed_result(self):
        delayed = self.stream(delay_ms=9)
        baseline = self.stream()
        q = POSES['profiles']['recovered']['joint_position']
        held = None
        for tick in range(2001):
            command = 'reset-pose' if tick == 0 else 'hold'
            held = delayed.send(command,q)
            self.assertEqual(held, baseline.send(command,q))
        self.assertTrue(held.startswith('holding '))
        values = list(map(float,held.split()[1:]))
        for _ in range(6):
            self.near(self.tick(delayed,q),values)
        self.assertGreater(max(self.tick(delayed,q)[29:58]),0)


if __name__ == '__main__':
    unittest.main()
