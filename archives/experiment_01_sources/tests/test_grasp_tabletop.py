"""Real-model regressions for the independent five-object tabletop scene."""

from itertools import combinations
import unittest

import mujoco
import numpy as np

from collisions import collision_bounds
from grasp_study_objects import STUDY_OBJECTS
from grasp_tabletop import FiveObjectSimulation, LAYOUT_XY_M, MASSES_KG


class FiveObjectSimulationTests(unittest.TestCase):
    def test_every_target_has_all_five_dynamic_visible_colliding_objects(self):
        for target in STUDY_OBJECTS:
            with self.subTest(target=target):
                sim = FiveObjectSimulation(target)
                try:
                    self.assertEqual(sim.skill.target.name, target)
                    self.assertTrue(sim.model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_MULTICCD)
                    collidable = (sim.model.geom_contype | sim.model.geom_conaffinity) != 0
                    for name, mass in MASSES_KG.items():
                        body = sim.model.body(f'{name}_object').id
                        joint = sim.model.joint(f'{name}_free')
                        self.assertEqual(joint.type[0], mujoco.mjtJoint.mjJNT_FREE)
                        self.assertEqual(sim.model.body_rootid[body], joint.bodyid[0])
                        self.assertAlmostEqual(sim.model.body_mass[body], mass)
                        body_geoms = sim.model.geom_bodyid == body
                        contact_geoms = np.flatnonzero(body_geoms & collidable)
                        self.assertGreater(len(contact_geoms), 0)
                        np.testing.assert_allclose(
                            sim.model.geom_friction[contact_geoms],
                            np.tile([.8, .005, .0001], (len(contact_geoms), 1)))
                        visible_geoms = np.flatnonzero(
                            body_geoms & (sim.options.geomgroup[sim.model.geom_group] != 0)
                            & (sim.model.geom_rgba[:, 3] > 0))
                        self.assertGreater(len(visible_geoms), 0)
                    finger_geoms = np.flatnonzero(
                        np.isin(sim.model.geom_bodyid, list(sim.skill.fingers)) & collidable)
                    np.testing.assert_allclose(
                        sim.model.geom_friction[finger_geoms],
                        np.tile([.8, .005, .0001], (len(finger_geoms), 1)))
                finally:
                    sim.close()

    def test_settled_collision_bounds_are_disjoint_and_inside_tabletop(self):
        sim = FiveObjectSimulation('bowl')
        try:
            table_low, table_high = collision_bounds(
                sim.model, sim.data, np.array([sim.model.geom('tabletop').id]))
            bounds = {}
            for state in sim.object_states():
                name = state['target']
                body = sim.model.body(state['name']).id
                geoms = np.flatnonzero(
                    (sim.model.geom_bodyid == body)
                    & ((sim.model.geom_contype | sim.model.geom_conaffinity) != 0))
                low, high = collision_bounds(sim.model, sim.data, geoms)
                bounds[name] = low, high
                self.assertTrue(np.all(low[:2] >= table_low[:2]), name)
                self.assertTrue(np.all(high[:2] <= table_high[:2]), name)
                self.assertLess(abs(low[2] - table_high[2]), .001, name)
                np.testing.assert_allclose(state['position_m'][:2], LAYOUT_XY_M[name], atol=.003)
                body_contacts = [
                    contact for contact in sim.data.contact
                    if {int(sim.model.geom_bodyid[contact.geom1]),
                        int(sim.model.geom_bodyid[contact.geom2])} == {body, sim.skill.table}
                ]
                self.assertTrue(body_contacts, f'{name} has no real table support')
            for left, right in combinations(bounds, 2):
                low, high = bounds[left]
                other_low, other_high = bounds[right]
                self.assertTrue(np.any(high[:2] <= other_low[:2])
                                or np.any(other_high[:2] <= low[:2]), (left, right))
        finally:
            sim.close()

    def test_object_states_follow_actual_poses_and_collision_clearance(self):
        sim = FiveObjectSimulation('spoon')
        try:
            # Move an unselected free object, then let real dynamics act on it.
            qpos = int(sim.model.joint('pen_free').qposadr[0])
            sim.data.qpos[qpos:qpos + 3] += [.02, -.01, .05]
            sim.data.qpos[qpos + 3:qpos + 7] = [np.cos(.2), 0., 0., np.sin(.2)]
            mujoco.mj_forward(sim.model, sim.data)
            raised_z = float(sim.data.body('pen_object').xpos[2])
            sim.step(.04)
            states = sim.object_states()
            self.assertEqual({state['target'] for state in states}, set(STUDY_OBJECTS))
            self.assertEqual(len(states), 5)
            table_top = collision_bounds(
                sim.model, sim.data, np.array([sim.model.geom('tabletop').id]))[1][2]
            for state in states:
                name = state['name']
                body = sim.data.body(name)
                np.testing.assert_array_equal(state['position_m'], body.xpos)
                np.testing.assert_array_equal(state['quaternion_wxyz'], body.xquat)
                geoms = np.flatnonzero(
                    (sim.model.geom_bodyid == sim.model.body(name).id)
                    & ((sim.model.geom_contype | sim.model.geom_conaffinity) != 0))
                bottom = collision_bounds(sim.model, sim.data, geoms)[0][2]
                self.assertAlmostEqual(state['table_clearance_m'], bottom - table_top)
            self.assertLess(sim.data.body('pen_object').xpos[2], raised_z)
            self.assertGreater(next(state for state in states if state['target'] == 'pen')
                               ['table_clearance_m'], .02)
            # Reset really restores all five, not just the currently selected spoon.
            sim.reset()
            np.testing.assert_allclose(sim.data.body('pen_object').xpos[:2],
                                       LAYOUT_XY_M['pen'], atol=.003)
            self.assertLess(abs(next(state for state in sim.object_states()
                                     if state['target'] == 'pen')['table_clearance_m']), .001)
        finally:
            sim.close()

    def test_bottle_is_lifted_and_retained_with_neighbors_on_table(self):
        from serve_grasp_study import StudyViewer

        viewer = StudyViewer()
        try:
            viewer.select('bottle')
            viewer.auto_advance = False
            while viewer.finished_at is None:
                self.assertLess(viewer.sim.data.time, 24.)
                viewer.advance()
            state = viewer.state()
            self.assertTrue(state['outcome'], state['failure_reason'])
            self.assertGreaterEqual(state['verified_hold_s'], 2.)
            self.assertGreaterEqual(state['lift_m'], .10)
            self.assertTrue(all(state['predicates'].values()))
            for item in state['objects']:
                if item['target'] == 'bottle':
                    self.assertGreater(item['table_clearance_m'], .10)
                else:
                    self.assertLess(abs(item['table_clearance_m']), .001)
            self.assertFalse(state['warnings'])
        finally:
            viewer.close()

    def test_unknown_target_is_rejected_before_creating_model(self):
        for target in ('unknown', '', None, [], 'Bowl'):
            with self.subTest(target=target):
                sim = object.__new__(FiveObjectSimulation)
                with self.assertRaises(ValueError):
                    sim.__init__(target)
                self.assertFalse(hasattr(sim, 'model'))
                self.assertFalse(hasattr(sim, 'data'))


if __name__ == '__main__':
    unittest.main()
