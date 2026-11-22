"""Deterministic real-contact regressions for drawer-storage observations."""

import unittest

import mujoco
import numpy as np

from collisions import collision_bounds
from drawer_storage_checks import DrawerFacts
from grasp_tabletop import FiveObjectSimulation


class DrawerFactsTests(unittest.TestCase):
    def setUp(self):
        self.sim = FiveObjectSimulation('spoon')
        self.addCleanup(self.sim.close)
        self.model, self.data = self.sim.model, self.sim.data
        self.facts = DrawerFacts(self.sim)
        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()
        self.initial_ctrl = self.data.ctrl.copy()

    def restore(self):
        # Pose initialization belongs only to fixtures, never the observer.
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        self.data.ctrl[:] = self.initial_ctrl
        mujoco.mj_forward(self.model, self.data)

    def geoms(self, target):
        body = self.model.body(f'{target}_object').id
        return np.flatnonzero((self.model.geom_bodyid == body)
                              & ((self.model.geom_contype | self.model.geom_conaffinity) != 0))

    def translate(self, target, offset):
        joint = self.model.joint(f'{target}_free')
        qpos, dof = int(joint.qposadr[0]), int(joint.dofadr[0])
        self.data.qpos[qpos:qpos + 3] += offset
        self.data.qvel[dof:dof + 6] = 0.
        mujoco.mj_forward(self.model, self.data)

    def place_in_drawer(self, target, gap=.002):
        joint = self.model.joint(f'{target}_free')
        qpos = int(joint.qposadr[0])
        self.data.qpos[qpos + 3:qpos + 7] = [1., 0., 0., 0.]
        mujoco.mj_forward(self.model, self.data)
        low, high = collision_bounds(self.model, self.data, self.geoms(target))
        region = self.model.site('cabinet_bottom_region').id
        center = self.data.site_xpos[region]
        offset = center - (low + high) / 2
        offset[2] = center[2] - self.model.site_size[region, 2] + gap - low[2]
        self.translate(target, offset)

    def settle(self, seconds=1.5):
        # Bypass mission control: only real gravity, joints and contact dynamics.
        for _ in range(round(seconds / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def test_initial_table_objects_are_not_drawer_supported_or_stored(self):
        state = self.facts.state()
        self.assertEqual([item['target'] for item in state['objects']], list(DrawerFacts.TARGETS))
        for item in state['objects']:
            with self.subTest(target=item['target']):
                self.assertFalse(item['inside'])
                self.assertFalse(item['supported'])
                self.assertEqual(item['support_force_n'], 0.)
                self.assertTrue(item['detached'])
                self.assertFalse(item['stored'])
        self.assertFalse(state['all_stored'])
        self.assertTrue(state['hand_empty'])
        self.assertFalse(state['drawer_closed'])

    def test_each_primitive_object_physically_settles_on_drawer_floor(self):
        for target in DrawerFacts.TARGETS:
            with self.subTest(target=target):
                self.restore()
                self.place_in_drawer(target)
                self.settle()
                item = self.facts.measure(target)
                self.assertTrue(item['inside'], item)
                self.assertTrue(item['supported'], item)
                self.assertTrue(item['stable'], item)
                self.assertTrue(item['detached'], item)
                self.assertTrue(item['stored'], item)
                body = self.model.body(f'{target}_object').id
                np.testing.assert_array_equal(item['position_m'], self.data.xpos[body])
                np.testing.assert_array_equal(item['quaternion_wxyz'], self.data.xquat[body])
                low, high = collision_bounds(self.model, self.data, self.geoms(target))
                np.testing.assert_array_equal(item['bounds_min_m'], low)
                np.testing.assert_array_equal(item['bounds_max_m'], high)
                # The measured support is enough to carry this object's weight,
                # not a force-free geometric overlap or a contact margin.
                weight = self.model.body_mass[body] * abs(self.model.opt.gravity[2])
                self.assertGreater(item['support_force_n'], weight * .8)
                self.assertLess(item['support_force_n'], weight * 1.2)
                self.assertFalse(self.facts.state()['all_stored'])

    def test_containment_without_real_floor_force_is_not_storage(self):
        for target in DrawerFacts.TARGETS:
            with self.subTest(target=target):
                self.restore()
                self.place_in_drawer(target, gap=.012)
                item = self.facts.measure(target)
                self.assertTrue(item['inside'], item)
                self.assertTrue(item['stable'], item)
                self.assertTrue(item['detached'], item)
                self.assertFalse(item['supported'], item)
                self.assertEqual(item['support_force_n'], 0.)
                self.assertFalse(item['stored'])

    def test_above_or_partly_outside_collision_bounds_are_not_contained(self):
        region = self.model.site('cabinet_bottom_region').id
        for target in DrawerFacts.TARGETS:
            for axis in (0, 2):
                with self.subTest(target=target, axis=axis):
                    self.restore()
                    self.place_in_drawer(target, gap=.012)
                    low, high = collision_bounds(self.model, self.data, self.geoms(target))
                    limit = self.data.site_xpos[region, axis] + self.model.site_size[region, axis]
                    offset = np.zeros(3)
                    offset[axis] = limit + .006 - high[axis]
                    self.translate(target, offset)
                    item = self.facts.measure(target)
                    self.assertFalse(item['inside'], item)
                    self.assertFalse(item['stored'])

    def test_linear_and_angular_motion_prevent_stored_success(self):
        for target in DrawerFacts.TARGETS:
            with self.subTest(target=target):
                self.restore()
                self.place_in_drawer(target)
                self.settle()
                self.assertTrue(self.facts.measure(target)['stored'])
                dof = int(self.model.joint(f'{target}_free').dofadr[0])
                for component, speed in ((0, .05), (3, .5)):
                    with self.subTest(component=component):
                        self.data.qvel[dof:dof + 6] = 0.
                        self.data.qvel[dof + component] = speed
                        mujoco.mj_forward(self.model, self.data)
                        item = self.facts.measure(target)
                        self.assertTrue(item['inside'], item)
                        self.assertTrue(item['supported'], item)
                        self.assertTrue(item['detached'], item)
                        self.assertFalse(item['stable'], item)
                        self.assertFalse(item['stored'])

    def test_other_object_support_does_not_count_as_drawer_floor_support(self):
        self.place_in_drawer('glass')
        self.settle()
        self.assertTrue(self.facts.measure('glass')['stored'])
        self.place_in_drawer('pen', gap=.01)
        glass_low, glass_high = collision_bounds(self.model, self.data, self.geoms('glass'))
        pen_low, pen_high = collision_bounds(self.model, self.data, self.geoms('pen'))
        offset = (glass_low + glass_high - pen_low - pen_high) / 2
        offset[2] = glass_high[2] - .0002 - pen_low[2]
        self.translate('pen', offset)
        bodies = {self.model.body(f'{target}_object').id for target in ('glass', 'pen')}
        force = np.zeros(6)
        supporting_contact = 0.
        for index, contact in enumerate(self.data.contact):
            pair = {int(self.model.geom_bodyid[contact.geom1]),
                    int(self.model.geom_bodyid[contact.geom2])}
            if pair == bodies and contact.efc_address >= 0:
                mujoco.mj_contactForce(self.model, self.data, index, force)
                supporting_contact += max(0., float(force[0]))
        self.assertGreater(supporting_contact, .001)
        item = self.facts.measure('pen')
        self.assertTrue(item['inside'], item)
        self.assertFalse(item['supported'], item)
        self.assertEqual(item['support_force_n'], 0.)
        self.assertFalse(item['stored'])

    def test_force_bearing_finger_contact_marks_any_object_as_attached(self):
        finger = self.model.body('robot_left_finger').id
        pads = np.flatnonzero((self.model.geom_bodyid == finger)
                             & (self.model.geom_type == mujoco.mjtGeom.mjGEOM_BOX)
                             & ((self.model.geom_contype | self.model.geom_conaffinity) != 0))
        self.assertGreater(len(pads), 0)
        low, high = collision_bounds(self.model, self.data, self.geoms('pen'))
        # A fixture overlap produces a real solved contact; no fake force or
        # attachment is inserted, and no integration of this overlap is needed.
        self.translate('pen', self.data.geom_xpos[pads[0]] - (low + high) / 2)
        item = self.facts.measure('pen')
        self.assertFalse(item['detached'], item)
        self.assertFalse(item['stored'])
        self.assertFalse(self.facts.state()['hand_empty'])
        self.assertTrue(self.facts.measure('bottle')['detached'])
        self.restore()
        self.assertTrue(self.facts.state()['hand_empty'])

    def test_containment_tracks_live_drawer_motion_not_initial_coordinates(self):
        self.place_in_drawer('bowl', gap=.01)
        before = self.facts.measure('bowl')
        self.assertTrue(before['inside'])
        joint = self.model.joint('cabinet_bottom_level')
        self.data.qpos[int(joint.qposadr[0])] += .13
        mujoco.mj_forward(self.model, self.data)
        after = self.facts.measure('bowl')
        np.testing.assert_array_equal(before['position_m'], after['position_m'])
        self.assertFalse(after['inside'])
        self.translate('bowl', np.array([0., .13, 0.]))
        self.assertTrue(self.facts.measure('bowl')['inside'])

    def test_closed_state_requires_real_joint_position_and_low_velocity(self):
        joint = self.model.joint('cabinet_bottom_level')
        qpos, dof = int(joint.qposadr[0]), int(joint.dofadr[0])
        for position, velocity, expected in (
            (float(joint.range[0]), 0., False),
            (float(joint.range[1]), 0., True),
            (float(joint.range[1]) - .02, 0., False),
            (float(joint.range[1]), .03, False),
            (float(joint.range[1]), -.03, False),
            (float(joint.range[1]) - .001, .001, True),
        ):
            with self.subTest(position=position, velocity=velocity):
                self.data.qpos[qpos] = position
                self.data.qvel[dof] = velocity
                mujoco.mj_forward(self.model, self.data)
                self.assertEqual(self.facts.state()['drawer_closed'], expected)

    def test_observations_do_not_mutate_physics_or_return_live_pose_arrays(self):
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.empty(mujoco.mj_stateSize(self.model, spec))
        after = np.empty_like(before)
        mujoco.mj_getState(self.model, self.data, before, spec)
        item = self.facts.measure('spoon')
        self.facts.objects()
        self.facts.state()
        item['position_m'][0] += 100.
        item['bounds_min_m'][0] += 100.
        mujoco.mj_getState(self.model, self.data, after, spec)
        np.testing.assert_array_equal(after, before)
        self.assertLess(self.facts.measure('spoon')['position_m'][0], 2.)


if __name__ == '__main__':
    unittest.main()
