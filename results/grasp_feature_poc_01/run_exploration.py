"""Exploratory MuJoCo grasp-feature screening; not a trained probability model.

Run from the project root:
.venv/bin/python results/grasp_feature_poc_01/run_exploration.py --phase screening --workers 3
All negatives come from physical execution. Existing project source is unchanged.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from benchmark import initial_state_fingerprint
from collisions import CollisionTracker
from grasp import GraspParameters, GraspSkill
from run_grasp_learning import provenance
from scenarios import SceneCase, TARGETS
from simulation import ARM_HOME, Simulation

BASE = {
    'target': 'bottle', 'scale': 1.0, 'mass_kg': 0.06,
    'sliding_friction': 0.8, 'x_m': 0.50, 'y_m': -0.08, 'yaw_rad': 0.0,
    'bias_x_m': 0.0, 'bias_y_m': 0.0, 'bias_z_m': 0.0,
    'opening_m': 0.08, 'closing_force_n': 4.0,
    'close_trigger_m': 0.0, 'descent_speed_m_s': 0.10,
}
PHYSICAL = ('bilateral_grip', 'clear_of_table', 'target_lifted', 'relative_pose_stable')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


class ExplorationSkill(GraspSkill):
    def apply_control(self):
        super().apply_control()
        if self.gripper_closed:
            velocity = float(self.data.qvel[self.finger_dof].mean())
            force = -self.sim.settings['closing_force_n'] - 10.0 * velocity
            self.data.ctrl[self.sim.gripper_actuator] = np.clip(force, -100, 100)


class ExplorationSimulation(Simulation):
    """Same controller/robot, with variants applied BEFORE model compilation."""
    def __init__(self, settings):
        self.settings = settings
        target = settings['target']
        self.case = SceneCase('feature_probe', target, 'shared_workspace', 5646, (0., 0.), settings['yaw_rad'])
        original = TARGETS[target]
        bias = np.array([settings[f'bias_{axis}_m'] for axis in 'xyz'])
        self.target = replace(original, grasp_offset_m=tuple(
            np.array(original.grasp_offset_m) * settings['scale'] + bias))
        scene = mujoco.MjSpec.from_file(str(ROOT / 'scene.xml'))
        robot = mujoco.MjSpec.from_file(str(ROOT / 'vendor/panda/panda.xml'))
        for key in list(robot.keys):
            robot.delete(key)
        for light in list(robot.lights):
            robot.delete(light)
        robot.body('hand').add_site(name='grasp_center', pos=[0, 0, .1034],
                                   size=[.003, .003, .003], rgba=[0, 0, 0, 0])
        gripper = robot.actuator('actuator8')
        gripper.gainprm[:] = 0
        gripper.gainprm[0] = 1
        gripper.biasprm[:] = 0
        gripper.ctrlrange[:] = [-100, 100]
        scene.attach(robot, frame=scene.frame('robot_mount'), prefix='robot_')
        assets = ROOT / 'vendor/libero'
        objects = (
            ('cabinet', 'articulated_objects/white_cabinet.xml', False),
            ('bowl', 'stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml', True),
            ('bottle', 'turbosquid_objects/wine_bottle/wine_bottle.xml', True),
            ('rack', 'turbosquid_objects/wine_rack/wine_rack.xml', False),
        )
        for name, filename, movable in objects:
            part = mujoco.MjSpec.from_file(str(assets / filename))
            root = part.worldbody.bodies[0]
            root.name = 'root'
            part.body('object').pos[:] = 0
            if movable:
                root.add_freejoint(name='free')
            if name == target:
                scale = settings['scale']
                for geom in part.geoms:
                    geom.pos[:] *= scale
                    if geom.type != mujoco.mjtGeom.mjGEOM_MESH:
                        geom.size[:] *= scale
                    if geom.contype or geom.conaffinity:
                        geom.friction[:] = [settings['sliding_friction'], .005, .0001]
                for mesh in part.meshes:
                    mesh.scale[:] *= scale
                scene.frame(f'{name}_mount').pos[:] = [settings['x_m'], settings['y_m'], .77]
            elif name in ('bowl', 'bottle'):
                scene.frame(f'{name}_mount').pos[:] = [.80, -.32, .77]
            else:
                # Keep required workcell names for existing preconditions, but
                # remove storage furniture from the actual grasping workspace.
                scene.frame(f'{name}_mount').pos[:] = [1.8, 1.8 if name == 'rack' else -1.8, .72]
            scene.attach(part, frame=scene.frame(f'{name}_mount'), prefix=f'{name}_')
        self.model = scene.compile()
        body = self.model.body(f'{target}_object').id
        compiled_mass = float(self.model.body_mass[body])
        ratio = settings['mass_kg'] / compiled_mass
        self.model.body_mass[body] *= ratio
        self.model.body_inertia[body] *= ratio
        fingers = [self.model.body(f'robot_{side}_finger').id for side in ('left', 'right')]
        collidable = (self.model.geom_contype | self.model.geom_conaffinity) != 0
        finger_geoms = np.flatnonzero(np.isin(self.model.geom_bodyid, fingers) & collidable)
        # MuJoCo combines equal-priority friction using the maximum. Set BOTH
        # sides, including torsion/rolling, so low object friction takes effect.
        self.model.geom_friction[finger_geoms] = [settings['sliding_friction'], .005, .0001]
        self.data = mujoco.MjData(self.model)
        mujoco.mj_setConst(self.model, self.data)
        self.arm_qpos = np.array([self.model.joint(f'robot_joint{i}').qposadr[0] for i in range(1, 8)])
        self.arm_actuators = np.array([self.model.actuator(f'robot_actuator{i}').id for i in range(1, 8)])
        self.finger_qpos = np.array([self.model.joint(f'robot_finger_joint{i}').qposadr[0] for i in (1, 2)])
        self.gripper_actuator = self.model.actuator('robot_actuator8').id
        self.arm_home = ARM_HOME.copy()
        self.object_names = ('bowl_object', 'bottle_object', 'cabinet_cabinet_bottom')
        self.renderer = None
        self.camera = mujoco.MjvCamera()
        self.options = mujoco.MjvOption()
        self.options.geomgroup[0] = 0
        self.options.geomgroup[3] = 0
        self.skill = ExplorationSkill(self)
        self.collisions = CollisionTracker(self.model, self.data, self.skill.target_body,
                                          self.skill.fingers, self.skill.table)
        self.reset()
        self.variant_features = self.describe_variant(compiled_mass)

    def describe_variant(self, compiled_mass):
        geoms = self.collisions.target_geoms
        local_rotations = np.stack([self.data.geom_xmat[g].reshape(3, 3) for g in geoms])
        extents = np.einsum('nij,nj->ni', np.abs(local_rotations), self.model.geom_size[geoms])
        width = np.max(self.data.geom_xpos[geoms] + extents, axis=0) - np.min(self.data.geom_xpos[geoms] - extents, axis=0)
        return {
            'compiled_pre_mass_override_kg': compiled_mass,
            'actual_mass_kg': float(self.model.body_mass[self.skill.target_body]),
            'actual_principal_inertia_kg_m2': self.model.body_inertia[self.skill.target_body].tolist(),
            'collision_world_aabb_size_m': width.tolist(),
            'object_collision_sliding_friction': self.model.geom_friction[geoms, 0].tolist(),
            'finger_collision_sliding_friction': self.model.geom_friction[
                np.flatnonzero(np.isin(self.model.geom_bodyid, list(self.skill.fingers)) &
                               ((self.model.geom_contype | self.model.geom_conaffinity) != 0)), 0].tolist(),
            'grasp_offset_body_m': list(self.target.grasp_offset_m),
        }


def run_one(spec, directory):
    trial_id, settings, metadata = spec
    started = time.monotonic()
    sim = ExplorationSimulation(settings)
    samples = []
    dt = float(sim.model.opt.timestep)
    last_sample = -math.inf
    last_phase = None
    actual_contact_frictions = set()
    phase_order = []
    phase_first = {}
    last_active_phase = None
    hold_started = None
    hold_s = 0.0
    execution_start = None

    def sample(force=False):
        nonlocal last_sample, last_phase, last_active_phase
        phase = sim.skill.phase
        now = float(sim.data.time)
        if phase not in phase_order:
            phase_order.append(phase)
            phase_first[phase] = None if execution_start is None else now - execution_start
        if phase not in ('idle', 'done', 'failed'):
            last_active_phase = phase
        for contact in sim.data.contact:
            bodies = {int(sim.model.geom_bodyid[contact.geom1]), int(sim.model.geom_bodyid[contact.geom2])}
            if sim.skill.target_body in bodies and bodies.intersection(sim.skill.fingers):
                actual_contact_frictions.add(round(float(contact.friction[0]), 6))
        if force or phase != last_phase or now - last_sample >= .02 - 1e-12:
            samples.append({
                'time_s': now, 'phase': phase, 'status': sim.skill.status,
                'tcp_position_m': sim.data.site_xpos[sim.skill.tcp].tolist(),
                'target_position_m': sim.data.xpos[sim.skill.target_body].tolist(),
                'target_quaternion': sim.data.xquat[sim.skill.target_body].tolist(),
                'grasp_predicates': sim.skill.grasp_postconditions(),
                'verified_hold_s': hold_s,
            })
            last_sample, last_phase = now, phase

    try:
        for _ in range(round(1.0 / dt)):
            sim.step(dt)
            sample()
        initial_pose = sim.get_object_pose(f"{settings['target']}_object")
        initial_fp = initial_state_fingerprint(sim)
        initial_error = sim.skill.precondition_error('grasp_and_hold')
        params = GraspParameters(initial_opening_m=settings['opening_m'],
                                 descent_speed_m_s=settings['descent_speed_m_s'],
                                 close_trigger_height_m=settings['close_trigger_m'])
        if initial_error is None:
            execution_start = float(sim.data.time)
            sim.start_task('grasp', params)
            while True:
                now = float(sim.data.time)
                predicates = sim.skill.grasp_postconditions()
                eligible = sim.skill.status in ('running', 'succeeded') and sim.skill.phase in ('hold', 'done') and all(predicates[k] for k in PHYSICAL)
                if eligible:
                    if hold_started is None:
                        hold_started = now
                    hold_s = now - hold_started
                else:
                    hold_started, hold_s = None, 0.0
                complete = sim.skill.status == 'failed' or (sim.skill.status == 'succeeded' and hold_s >= 2.0)
                sample(force=complete)
                if complete:
                    break
                if now - execution_start > 22.1:
                    raise RuntimeError('Independent observer watchdog exceeded')
                sim.step(dt)
        sample(force=True)
        success = sim.skill.status == 'succeeded' and hold_s >= 2.0 if initial_error is None else None
        collisions = sim.collisions.state()
        if initial_error is not None:
            failure_kind = 'invalid_initial_state'
        elif success:
            failure_kind = None
        elif collisions['episodes']:
            failure_kind = 'unplanned_contact'
        elif last_active_phase in ('close', 'lift', 'hold'):
            failure_kind = 'grasp_contact_or_retention'
        else:
            failure_kind = 'motion_or_preparation'
        row = {
            'trial_id': trial_id, 'settings': settings, 'metadata': metadata,
            'variant_features': sim.variant_features,
            'trial_status': 'completed' if initial_error is None else 'invalid_initial_state',
            'success': success, 'failure_kind': failure_kind,
            'failure_reason': sim.skill.failure_reason if initial_error is None else initial_error,
            'last_active_phase': last_active_phase,
            'phase_order': phase_order, 'phase_first_sim_s': phase_first,
            'execution_sim_s': float(sim.data.time) - execution_start if execution_start is not None else None,
            'verified_hold_s': hold_s, 'initial_state_fingerprint': initial_fp,
            'initial_pose': initial_pose,
            'final_pose': sim.get_object_pose(f"{settings['target']}_object"),
            'postconditions': sim.skill.grasp_postconditions(),
            'actual_finger_target_sliding_frictions': sorted(actual_contact_frictions),
            'collisions': collisions, 'warnings': sim.warning_names(),
            'wall_s': time.monotonic() - started,
        }
        save(directory / 'trials' / f'{trial_id}.json', row)
        path = directory / 'traces' / f'{trial_id}.json.gz'
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'xt', encoding='utf-8') as stream:
            json.dump({'trial_id': trial_id, 'samples': samples}, stream, allow_nan=False)
        return row
    finally:
        sim.close()


def screening():
    choices = [('reference', {})]
    for feature, values in {
        'mass_kg': [.02, .18, .45], 'sliding_friction': [.05, .20, 1.2],
        'scale': [.70, 1.30], 'x_m': [.45, .55], 'y_m': [-.13, -.03],
        'yaw_rad': [-.20, .20],
        'bias_x_m': [-.012, .012], 'bias_y_m': [-.012, .012], 'bias_z_m': [-.015, .015],
        'opening_m': [.02, .04], 'closing_force_n': [2., 8.],
        'close_trigger_m': [.004, .02], 'descent_speed_m_s': [.04, .20],
    }.items():
        for index, value in enumerate(values):
            choices.append((f'{feature}_{index}', {feature: value}))
    return [(f'{target}.{name}', {**BASE, 'target': target, **updates},
             {'design': 'one_factor_at_a_time', 'factor': next(iter(updates), 'reference')})
            for target in ('bowl', 'bottle') for name, updates in choices]


def interactions():
    specs = []
    for target in ('bowl', 'bottle'):
        for mass in (.06, .18, .45):
            for friction in (.10, .40, .80):
                for force in (4., 8.):
                    name = f'{target}.mass_{mass:.2f}.mu_{friction:.2f}.force_{force:g}'
                    specs.append((name, {**BASE, 'target': target, 'mass_kg': mass,
                                        'sliding_friction': friction, 'closing_force_n': force},
                                  {'design': 'mass_friction_force_factorial'}))
    return specs


def jitter_trials():
    # A declared distribution over unobserved execution-point error, not repeated
    # identical deterministic resets. The same draws are paired across configs.
    specs = []
    for target in ('bowl', 'bottle'):
        for seed in range(10):
            rng = random.Random(7721 + seed)
            error = {f'bias_{axis}_m': rng.uniform(-.006, .006) for axis in 'xyz'}
            for name, updates in (
                ('light_standard', {'mass_kg': .06, 'closing_force_n': 4.}),
                ('heavy_standard', {'mass_kg': .45, 'closing_force_n': 4.}),
                ('heavy_stronger', {'mass_kg': .45, 'closing_force_n': 8.}),
            ):
                settings = {**BASE, 'target': target, **updates, **error}
                specs.append((f'{target}.{name}.jitter_{seed:02d}', settings, {
                    'design': 'paired_execution_error_jitter', 'condition': name,
                    'seed': 7721 + seed,
                    'error_distribution': 'Independent uniform [-6,+6] mm for body-frame XYZ grasp-point error; shared draws across configurations',
                    'error_observability': 'Latent nuisance for a possible future nominal-feature predictor; saved here for reproducibility, not observed by vision',
                }))
    return specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('smoke', 'screening', 'interactions', 'jitter'), required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent / args.phase
    directory.mkdir(exist_ok=False)
    specs = {'smoke': lambda: [(f'{t}.smoke', {**BASE, 'target': t}, {'design': 'smoke'}) for t in ('bowl', 'bottle')],
             'screening': screening, 'interactions': interactions, 'jitter': jitter_trials}[args.phase]()
    save(directory / 'protocol.json', {
        'phase': args.phase, 'trials': specs,
        'physical_setup': 'Existing Panda/table/assets; both targets use common world position; rack/cabinet moved outside workspace; non-target object relocated; sizes compiled before simulation',
        'fixed_contact_settings': 'Object and fingertip sliding friction equal specified mu; torsional 0.005 and rolling 0.0001. Table friction unchanged. Closing force is total tendon command, NOT measured per-finger normal force.',
        'mass_size_separation': 'Object mass held at specified absolute mass across scale changes; inertia preserves compiled uniform-density shape at that mass. Grasp offset scales with object geometry.',
        'label_rule': 'Actual bilateral grip, table clearance, lift >=0.10 m and relative-position stability, held continuously for two simulation seconds; invalid preconditions unlabelled; numerical errors abort',
        'statistical_scope': 'Exploratory screening, not independent evaluation or calibrated success probabilities. Two geometry families with rim/neck-specific grasp profiles, NOT isolated semantic-category effects.',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    save(directory / 'environment.json', provenance())
    started = time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_one, spec, directory): spec[0] for spec in specs}
        for future in as_completed(pending):
            row = future.result()
            rows.append(row)
            print(json.dumps({
                'trial': row['trial_id'], 'success': row['success'],
                'failure_kind': row['failure_kind'], 'failure': row['failure_reason'],
                'last_phase': row['last_active_phase'],
                'contact_mu': row['actual_finger_target_sliding_frictions'],
            }, ensure_ascii=False), flush=True)
    order = {spec[0]: i for i, spec in enumerate(specs)}
    rows.sort(key=lambda row: order[row['trial_id']])
    completed = [row for row in rows if row['trial_status'] == 'completed']
    summary = {
        'attempted': len(rows), 'labelled': len(completed),
        'successes': sum(row['success'] is True for row in completed),
        'failures': sum(row['success'] is False for row in completed),
        'invalid_initial_states': len(rows) - len(completed),
        'failure_kinds': dict(Counter(row['failure_kind'] for row in completed if not row['success'])),
        'failure_reasons': dict(Counter(row['failure_reason'] for row in completed if not row['success'])),
        'numerical_warning_trials': sum(bool(row['warnings']) for row in rows),
        'unexpected_contact_trials': sum(row['collisions']['episodes'] > 0 for row in completed),
        'wall_s': time.monotonic() - started,
    }
    save(directory / 'trials.json', {'rows': rows})
    save(directory / 'summary.json', summary)
    print('SUMMARY ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()