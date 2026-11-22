"""End-to-end real-physics coverage for persistent five-object drawer storage."""

import unittest

import numpy as np

from serve_grasp_study import StudyViewer


class ContinuousStorageTests(unittest.TestCase):
    def test_all_five_are_released_then_physically_closed_without_scene_reset(self):
        viewer = StudyViewer()
        try:
            initial = viewer.state()
            initial_positions = {item['target']: np.array(item['position_m'])
                                 for item in initial['objects']}
            initial_qpos = viewer.sim.data.qpos.copy()
            viewer.command({'action': 'pause'})
            viewer.advance()
            self.assertEqual(viewer.sim.data.time, initial['time_s'])
            np.testing.assert_array_equal(viewer.sim.data.qpos, initial_qpos)
            viewer.command({'action': 'pause'})
            previous_time = float(viewer.sim.data.time)
            checked = set()
            entered_push = False
            while viewer.session.status == 'running':
                viewer.advance()
                now = float(viewer.sim.data.time)
                self.assertGreater(now, previous_time)
                self.assertLess(now, 481.)
                previous_time = now
                for target in viewer.session.completed_targets:
                    if target in checked:
                        continue
                    item = viewer.session.facts.measure(target)
                    self.assertTrue(item['stored'], item)
                    self.assertGreater(np.linalg.norm(np.array(item['position_m'])
                                                     - initial_positions[target]), .1)
                    checked.add(target)
                if viewer.session.phase == 'push_drawer' and not entered_push:
                    self.assertEqual(checked, set(initial_positions))
                    self.assertTrue(all(item['inside'] and item['detached']
                                        for item in viewer.session.facts.objects()))
                    entered_push = True
            state = viewer.state()
            self.assertTrue(state['outcome'], state['failure_reason'])
            self.assertEqual(state['scene_resets_during_mission'], 0)
            self.assertTrue(entered_push)
            self.assertTrue(state['all_stored'], state['objects'])
            self.assertTrue(state['drawer_closed'])
            self.assertTrue(state['hand_empty'])
            self.assertGreaterEqual(state['verified_hold_s'], 3.)
            self.assertGreaterEqual(state['push_contact_peak_n'], .1)
            self.assertLess(np.max(np.abs(viewer.sim.data.qpos[viewer.sim.arm_qpos]
                                          - viewer.sim.arm_home)), .03)
            self.assertFalse(state['warnings'])
            # Completion must remain a closed, occupied drawer, not auto-replay.
            for _ in range(20):
                viewer.advance()
            state = viewer.state()
            self.assertEqual(state['controller_status'], 'succeeded')
            self.assertTrue(state['all_stored'] and state['drawer_closed'] and state['hand_empty'])
        finally:
            viewer.close()


if __name__ == '__main__':
    unittest.main()
