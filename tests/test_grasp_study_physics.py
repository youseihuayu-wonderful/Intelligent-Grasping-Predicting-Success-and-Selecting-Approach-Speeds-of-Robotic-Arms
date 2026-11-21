"""Real collision geometry and study-local support gates, not synthetic labels."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from collisions import CollisionTracker, collision_bounds
from grasp_study_objects import STUDY_OBJECTS
from grasp_study_physics import (
    BASE, PHYSICAL, ROOT, ExplorationSimulation, StudySceneCase,
    run_one,
)
from scenarios import SceneCase


class StudyGeometryTests(unittest.TestCase):
    def test_study_targets_do_not_expand_legacy_cases(self):
        for target in ('spoon', 'glass', 'pen'):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    SceneCase('invalid', target, 'nominal', 0, (0., 0.), 0.)
                with self.assertRaises(ValueError):
                    StudySceneCase('invalid', target, 'nominal', 0, (.09, 0.), 0.)

    def test_new_assets_are_finite_rigid_collision_bodies(self):
        dimensions = {'spoon': (.150, .036, .017547955871509924),
                      'glass': (.064, .064, .090), 'pen': (.140, .012, .012)}
        for target, expected in dimensions.items():
            with self.subTest(target=target):
                spec = mujoco.MjSpec.from_file(str(ROOT / STUDY_OBJECTS[target].asset_path))
                spec.worldbody.bodies[0].add_freejoint(name='free')
                model = spec.compile()
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                body = model.body('object').id
                geoms = np.flatnonzero(model.geom_bodyid == body)
                self.assertTrue(((model.geom_contype[geoms] | model.geom_conaffinity[geoms]) != 0).all())
                self.assertEqual(model.neq, 0)
                self.assertEqual(model.njnt, 1)
                self.assertGreater(model.body_mass[body], 0.)
                inertia = model.body_inertia[body]
                self.assertTrue(np.isfinite(inertia).all())
                self.assertTrue((inertia > 0.).all())
                self.assertLessEqual(float(max(inertia)), float(sum(inertia) - max(inertia)) + 1e-12)
                low, high = collision_bounds(model, data, geoms)
                np.testing.assert_allclose(high - low, expected, atol=1e-10)

    def test_glass_has_an_open_interior_and_solid_bottom_and_walls(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT / STUDY_OBJECTS['glass'].asset_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        hit = np.array([-1], dtype=np.int32)
        down = mujoco.mj_ray(model, data, np.array([0., 0., .08]),
                            np.array([0., 0., -1.]), None, True, -1, hit)
        side = mujoco.mj_ray(model, data, np.array([0., 0., .05]),
                            np.array([1., 0., 0.]), None, True, -1, hit)
        up = mujoco.mj_ray(model, data, np.array([0., 0., .05]),
                          np.array([0., 0., 1.]), None, True, -1, hit)
        self.assertAlmostEqual(down, .076)
        self.assertAlmostEqual(side, .029)
        self.assertEqual(up, -1.)

    def test_cylinder_bounds_use_round_support_not_box_corners(self):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><geom type="cylinder" size=".006 .058" '
            'euler="45 0 0"/></worldbody></mujoco>')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        low, high = collision_bounds(model, data, np.array([0]))
        extent = (.006 + .058) / np.sqrt(2.)
        np.testing.assert_allclose(high - low, [.012, 2 * extent, 2 * extent], atol=1e-12)

    def test_clearance_threshold_uses_each_actual_primitive_surface(self):
        for kind, size, height in (('box', '.02 .02 .006', .006),
                                   ('cylinder', '.006 .02', .02),
                                   ('capsule', '.006 .02', .026)):
            with self.subTest(kind=kind):
                model = mujoco.MjModel.from_xml_string(
                    '<mujoco><worldbody><body name="table">'
                    '<geom type="box" pos="0 0 -.01" size="1 1 .01"/>'
                    '</body><body name="target"><freejoint/>'
                    f'<geom type="{kind}" size="{size}"/>'
                    '</body></worldbody></mujoco>')
                data = mujoco.MjData(model)
                tracker = CollisionTracker(model, data, model.body('target').id,
                                           set(), model.body('table').id)
                for gap, clear in ((0., False), (.0005, False), (.0015, True)):
                    data.qpos[2] = height + gap
                    mujoco.mj_forward(model, data)
                    self.assertEqual(tracker._clear_of_table(), clear)




class StudyPhysicalIntegrationTests(unittest.TestCase):
    def test_scaled_new_objects_have_real_table_support_and_shared_gates(self):
        for target in ('spoon', 'glass', 'pen'):
            for scale in (.8, 1.2):
                with self.subTest(target=target, scale=scale):
                    mass = STUDY_OBJECTS[target].mass_levels_kg[0]
                    sim = ExplorationSimulation({**BASE, 'target': target, 'scale': scale,
                                                 'mass_kg': mass, 'sliding_friction': .4})
                    try:
                        self.assertAlmostEqual(sim.model.body_mass[sim.skill.target_body], mass)
                        sim.step(1.)
                        self.assertTrue(sim.skill._contacts()[1])
                        self.assertFalse(sim.collisions._clear_of_table())
                        with self.assertRaises(ValueError):
                            sim.start_task('store_bowl')
                        with self.assertRaises(ValueError):
                            sim.start_skill('place_and_release')
                        sim.skill.gripper_closed = True
                        with self.assertRaises(ValueError):
                            sim.start_skill('grasp_and_hold')
                    finally:
                        sim.close()

    def test_new_shapes_pass_unchanged_independent_two_second_hold(self):
        for target, mass in (('spoon', .025), ('glass', .080), ('pen', .010)):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                settings = {**BASE, 'target': target, 'mass_kg': mass}
                row = run_one((f'{target}.physical_regression', settings,
                               {'design': 'unit_test_real_physics_not_study_data'}), directory)
                self.assertEqual(row['trial_status'], 'completed')
                self.assertTrue(row['success'], row['failure_reason'])
                self.assertGreaterEqual(row['verified_hold_s'], 2.)
                self.assertTrue(all(row['postconditions'].values()))
                self.assertEqual(row['warnings'], [])
                self.assertEqual(row['actual_finger_target_sliding_frictions'],
                                 [settings['sliding_friction']])
                with gzip.open(directory / 'traces' / f'{target}.physical_regression.json.gz',
                               'rt', encoding='utf-8') as stream:
                    samples = json.load(stream)['samples']
                eligible = [sample for sample in samples if sample['verified_hold_s'] > 0.]
                self.assertTrue(all(all(sample['grasp_predicates'][name] for name in PHYSICAL)
                                    for sample in eligible))
                self.assertGreaterEqual(eligible[-1]['verified_hold_s'], 2.)
                early_terminal = [sample for sample in eligible
                                  if sample['status'] == 'succeeded' and sample['verified_hold_s'] < 2.]
                self.assertTrue(early_terminal, 'The observer must continue beyond controller completion')


if __name__ == '__main__':
    unittest.main()
