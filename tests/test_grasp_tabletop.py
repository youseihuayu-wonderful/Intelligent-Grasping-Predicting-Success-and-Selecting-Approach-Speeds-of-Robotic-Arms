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
        from grasp import GraspParameters

        sim = FiveObjectSimulation('bottle')
        try:
            sim.start_task('grasp', GraspParameters())
            held_s = 0.
            while held_s < 2.:
                self.assertLess(sim.data.time, 24.)
                sim.step(.02)
                self.assertNotEqual(sim.skill.status, 'failed', sim.skill.failure_reason)
                valid = (sim.skill.phase in ('hold', 'done')
                         and all(sim.skill.grasp_postconditions().values()))
                held_s = held_s + .02 if valid else 0.
            self.assertEqual(sim.skill.status, 'succeeded')
            for item in sim.object_states():
                if item['target'] == 'bottle':
                    self.assertGreater(item['table_clearance_m'], .10)
                else:
                    self.assertLess(abs(item['table_clearance_m']), .001)
            self.assertFalse(sim.warning_names())
        finally:
            sim.close()

    def test_open_cabinet_is_supported_inside_table_and_clear_of_objects(self):
        sim = FiveObjectSimulation('bowl')
        try:
            cabinet = sim.cabinet_state()
            low, high = np.array(cabinet['bounds_min_m']), np.array(cabinet['bounds_max_m'])
            table_low, table_high = collision_bounds(
                sim.model, sim.data, np.array([sim.model.geom('tabletop').id]))
            self.assertTrue(np.all(low[:2] >= table_low[:2]))
            self.assertTrue(np.all(high[:2] <= table_high[:2]))
            self.assertAlmostEqual(low[2], table_high[2])
            self.assertTrue(cabinet['fully_open'])
            for item in sim.object_states():
                body = sim.model.body(item['name']).id
                geoms = np.flatnonzero(
                    (sim.model.geom_bodyid == body)
                    & ((sim.model.geom_contype | sim.model.geom_conaffinity) != 0))
                object_low, object_high = collision_bounds(sim.model, sim.data, geoms)
                self.assertTrue(np.any(object_high[:2] <= low[:2])
                                or np.any(high[:2] <= object_low[:2]), item['target'])
        finally:
            sim.close()

    def test_cabinet_cavity_is_open_and_fits_each_object_individually(self):
        sim = FiveObjectSimulation('bowl')
        try:
            cabinet = sim.cabinet_state()
            low, high = np.array(cabinet['interior_min_m']), np.array(cabinet['interior_max_m'])
            center = (low + high) / 2
            hit = np.array([-1], dtype=np.int32)
            down = mujoco.mj_ray(sim.model, sim.data, center, np.array([0., 0., -1.]),
                                None, True, -1, hit)
            self.assertEqual(int(hit[0]), sim.model.geom('cabinet_drawer_floor').id)
            self.assertAlmostEqual(down, center[2] - low[2])
            up = mujoco.mj_ray(sim.model, sim.data, center, np.array([0., 0., 1.]),
                              None, True, -1, hit)
            self.assertEqual(up, -1.)
            for item in sim.object_states():
                body = sim.model.body(item['name']).id
                geoms = np.flatnonzero(
                    (sim.model.geom_bodyid == body)
                    & ((sim.model.geom_contype | sim.model.geom_conaffinity) != 0))
                object_low, object_high = collision_bounds(sim.model, sim.data, geoms)
                self.assertTrue(np.all(object_high - object_low + .003 < high - low),
                                item['target'])
        finally:
            sim.close()

    def test_drawer_moves_physically_and_has_a_reachable_open_approach(self):
        sim = FiveObjectSimulation('bowl')
        try:
            before = sim.cabinet_state()
            joint = sim.model.joint('cabinet_bottom_level')
            sim.data.qvel[int(joint.dofadr[0])] = .1
            sim.step(.05)
            after = sim.cabinet_state()
            self.assertGreater(after['drawer_position_m'], before['drawer_position_m'] + .0005)
            self.assertGreater(after['interior_min_m'][1], before['interior_min_m'][1] + .0005)
            sim.reset()
            restored = sim.cabinet_state()
            self.assertAlmostEqual(restored['drawer_position_m'], before['drawer_position_m'])
            point = (np.array(restored['interior_min_m']) + restored['interior_max_m']) / 2
            point[2] = restored['interior_max_m'][2] + .08
            sim.skill._inverse_kinematics(point)
            np.testing.assert_allclose(sim.skill.work.site_xpos[sim.skill.tcp], point, atol=.0003)
        finally:
            sim.close()

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
