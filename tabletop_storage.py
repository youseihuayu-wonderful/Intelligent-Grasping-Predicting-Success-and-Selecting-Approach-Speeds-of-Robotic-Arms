"""Continuous five-object storage: no scene resets or object-pose commands."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from collisions import CollisionTracker, collision_bounds
from drawer_storage_checks import DrawerFacts
from grasp import GraspParameters, IKConvergenceError
from grasp_study_objects import STUDY_OBJECTS
from grasp_study_physics import BASE, PHYSICAL, ExplorationSkill
from grasp_tabletop import FiveObjectSimulation, MASSES_KG
from storage import BowlStorage


# Rear, low objects first. Coordinates are offsets from the live cavity centre.
PACKING_XY_M = {
    'spoon': (-.080, .033), 'pen': (.015, -.020), 'bowl': (-.100, -.035),
    'bottle': (.063, .010), 'glass': (.063, -.055),
}
ORDER = tuple(PACKING_XY_M)


class StorageServo(ExplorationSkill):
    """Existing grasp controller plus sequencer-owned, feedback-checked segments."""

    def __init__(self, sim):
        super().__init__(sim)
        self.managed = False
        self.moving = False
        self.arrived = False
        self.cartesian_knots = None

    def update(self):
        if not self.managed:
            super().update()
        elif self.status == 'running' and self.moving:
            self.arrived = self._motion_done(float(self.data.time) - self.phase_started_at)

    def plan_cartesian_line(self):
        start = self.data.site_xpos[self.tcp].copy()
        delta = self.target_position - start
        intervals = max(1, int(np.ceil(np.linalg.norm(delta) / .005)))
        knots = [self.data.qpos[self.sim.arm_qpos].copy()]
        try:
            for index in range(1, intervals + 1):
                knots.append(self._inverse_kinematics(start + delta * (index / intervals)))
        except IKConvergenceError as exc:
            self.fail(f'Cartesian handle path is unreachable: {exc}')
            return
        self.cartesian_knots = np.array(knots)
        for index in range(intervals):
            self.work.qpos[:] = self.data.qpos
            self.work.qpos[self.sim.arm_qpos] = (knots[index] + knots[index + 1]) * .5
            mujoco.mj_kinematics(self.model, self.work)
            expected = start + delta * ((index + .5) / intervals)
            if np.linalg.norm(self.work.site_xpos[self.tcp] - expected) > .001:
                self.fail('Cartesian handle path exceeds 1 mm planned interpolation error')
                return
        peak_joint_slope = float(np.max(np.abs(np.diff(self.cartesian_knots, axis=0)))) * intervals
        self.motion_duration = max(self.motion_duration, 1.875 * peak_joint_slope / .8)

    def _motion_done(self, elapsed):
        if self.cartesian_knots is None:
            return super()._motion_done(elapsed)
        u = min(1., elapsed / self.motion_duration)
        progress = u**3 * (10 - 15*u + 6*u*u)
        intervals = len(self.cartesian_knots) - 1
        coordinate = progress * intervals
        index = min(intervals - 1, int(coordinate))
        fraction = coordinate - index
        self.arm_target = (self.cartesian_knots[index] * (1 - fraction)
                           + self.cartesian_knots[index + 1] * fraction)
        arrived = np.linalg.norm(self.data.site_xpos[self.tcp] - self.target_position) < .004
        slow = np.max(np.abs(self.data.qvel[self.arm_dof])) < .08
        if elapsed > self.motion_duration + 2.:
            self.fail('Arm did not complete the Cartesian handle path in time')
            return False
        return elapsed >= self.motion_duration and arrived and slow

    def verify_terminal(self):
        # After intentional release, the session verifies storage, not grasp hold.
        if not self.managed:
            super().verify_terminal()


class StorageSession:
    REQUIRED_FINAL_HOLD_S = 3.0
    FINAL_SETTLE_TIMEOUT_S = 20.0

    def __init__(self):
        self.sim = FiveObjectSimulation(ORDER[0])
        self.facts = DrawerFacts(self.sim)
        self.targets = ORDER
        self.current_index = 0
        self.target = ORDER[0]
        self.status = 'running'
        self.phase = self.mode = 'settling'
        self.failure_reason = None
        self.completed_targets = []
        self.events = []
        self.contact_reports = []
        self.grasp_hold_s = self.verified_hold_s = 0.
        self.hold_started = None
        self.started_at = float(self.sim.data.time)
        self.events.append({'time_s': self.started_at, 'target': self.target, 'phase': 'settling'})
        self.initial_reset_count = self.sim.scene_reset_count
        self.released = False
        self.push_contact_peak_n = 0.
        self._lost_contact_s = 0.
        self._stable_s = 0.
        self._hand_relative = None
        self._phase_time = self.started_at
        m = self.sim.model
        self._handle = m.geom('cabinet_drawer_handle').id
        self._region = m.site('cabinet_bottom_region').id
        self._drawer = m.body('cabinet_cabinet_bottom').id
        self._floor = m.geom('cabinet_drawer_floor').id
        self._cabinet_bodies = {m.body(name).id for name in ('cabinet_base', 'cabinet_cabinet_bottom')}
        self._hand_bodies = {m.body(name).id for name in ('robot_hand', 'robot_left_finger', 'robot_right_finger')}
        self._robot_bodies = {i for i in range(m.nbody) if m.body(i).name.startswith('robot_')}
        self._force = np.zeros(6)

    def close(self):
        self.sim.close()

    def _enter(self, phase):
        self.phase = self.mode = phase
        self._phase_time = float(self.sim.data.time)
        self._stable_s = 0.
        self.events.append({'time_s': self._phase_time, 'target': self.target, 'phase': phase})

    def _fail(self, reason):
        if self.status == 'failed':
            return
        self.failure_reason = f'{self.target}/{self.phase}: {reason}'
        self.status = 'failed'
        self.sim.skill.fail(self.failure_reason)
        self._enter('failed')

    def _begin_item(self):
        if not self.facts.state()['hand_empty']:
            self._fail('Hand is not empty before the next object')
            return
        s = self.sim
        self.target = self.targets[self.current_index]
        s.target = STUDY_OBJECTS[self.target].grasp_target
        body = s.model.body(f'{self.target}_object').id
        s.settings = {**BASE, 'target': self.target, 'mass_kg': MASSES_KG[self.target],
                      'x_m': float(s.data.xpos[body, 0]), 'y_m': float(s.data.xpos[body, 1])}
        s.skill = StorageServo(s)
        s.skill.arm_target = s.data.qpos[s.arm_qpos].copy()
        s.collisions = CollisionTracker(s.model, s.data, s.skill.target_body, s.skill.fingers, s.skill.table)
        self.released = False
        self.grasp_hold_s = 0.
        self.hold_started = None
        reason = s.skill.precondition_error('grasp_and_hold')
        if reason:
            self._fail(reason)
            return
        s.start_task('grasp', GraspParameters(initial_opening_m=.08, descent_speed_m_s=.10,
                                             close_trigger_height_m=0.))
        if self.target in ('spoon', 'glass'):
            s.skill.orientation_target = np.array([0., 2**-.5, 2**-.5, 0.])
        self.mode = 'grasp'
        self.phase = s.skill.phase
        self.events.append({'time_s': float(s.data.time), 'target': self.target, 'phase': self.phase})

    def _move(self, phase, point, speed=.10):
        k = self.sim.skill
        k.managed = True
        k.status = 'running'
        k.arrived = False
        k.moving = True
        k.cartesian_knots = None
        self._enter(phase)
        k._move(phase, np.asarray(point, dtype=float))
        if k.status != 'failed':
            distance = float(np.linalg.norm(np.asarray(point) - self.sim.data.site_xpos[k.tcp]))
            k.motion_duration = max(k.motion_duration, 1.875 * distance / speed)
            if phase in ('push_drawer', 'push_withdraw'):
                k.plan_cartesian_line()

    def _hold(self, phase):
        k = self.sim.skill
        k.moving = False
        k.arm_target = self.sim.data.qpos[self.sim.arm_qpos].copy()
        k._enter(phase)
        self._enter(phase)

    def _release(self):
        self._hold('release')
        k = self.sim.skill
        k.gripper_closed = False
        self._release_opening_required = min(
            .075, float(self.sim.data.qpos[self.sim.finger_qpos].sum()) + .005)
        k.opening_target = .08
        self._release_support_seen = self.facts.measure(self.target)['supported']

    def _tcp(self):
        return self.sim.data.site_xpos[self.sim.skill.tcp].copy()

    def _slot(self):
        return self.sim.data.site_xpos[self._region, :2] + PACKING_XY_M[self.target]

    def _align(self, xy, bottom_z):
        item = self.facts.measure(self.target)
        low, high = np.array(item['bounds_min_m']), np.array(item['bounds_max_m'])
        point = self._tcp()
        point[:2] += np.asarray(xy) - (low[:2] + high[:2]) * .5
        point[2] += bottom_z - low[2]
        return point

    def planned_contact(self, a, b, phase):
        m = self.sim.model
        ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
        if self.released and self._floor in (a, b) and self.sim.skill.target_body in (ba, bb):
            return True
        if self.mode in ('lower', 'release', 'withdraw', 'verify_release'):
            if {ba, bb} == {self.sim.skill.target_body, self._drawer}:
                return True
        return (self.mode in ('push_drawer', 'push_withdraw') and self._handle in (a, b)
                and bool({ba, bb} & self._hand_bodies))

    def _check_robot_contacts(self):
        m, d = self.sim.model, self.sim.data
        for i, c in enumerate(d.contact):
            a, b = int(c.geom1), int(c.geom2)
            ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
            if not ({ba, bb} & self._robot_bodies and {ba, bb} & self._cabinet_bodies):
                continue
            mujoco.mj_contactForce(m, d, i, self._force)
            force = float(self._force[0])
            if self.planned_contact(a, b, self.phase):
                if self.mode == 'push_drawer':
                    self.push_contact_peak_n = max(self.push_contact_peak_n, force)
            elif force > 1.:
                self._fail(f'Unplanned robot/cabinet contact: {m.body(ba).name}/{m.body(bb).name}, {force:.2f} N')
                return

    def _start_transfer(self):
        s, k = self.sim, self.sim.skill
        k.managed = True
        k.status = 'running'
        k.storage = self
        self._hand_relative = s.data.site_xmat[k.tcp].reshape(3, 3).T @ (
            s.data.xpos[k.target_body] - self._tcp())
        self._lost_contact_s = 0.
        item = self.facts.measure(self.target)
        self._low_profile = item['bounds_max_m'][2] - item['bounds_min_m'][2] < .04
        roof = s.cabinet_state()['bounds_max_m'][2]
        point = self._tcp()
        point[2] += max(0., roof + .010 - item['bounds_min_m'][2])
        point[2] = max(point[2], 1.05)
        self._move('raise', point)

    def _finish_item(self):
        item = self.facts.measure(self.target)
        if not item['stored']:
            self._fail('Object no longer contained, floor-supported, stable and detached after withdrawal')
            return
        self.completed_targets.append(self.target)
        self.contact_reports.append({'target': self.target, 'contacts': self.sim.collisions.state()})
        self.current_index += 1
        if self.current_index < len(self.targets):
            self._begin_item()
        else:
            self._start_push()

    def _start_push(self):
        facts = self.facts.state()
        if not (facts['all_stored'] and facts['hand_empty']):
            self._fail('Cannot close drawer before all five objects are stored and the hand is empty')
            return
        self._start_home('prepush_home')

    def _start_home(self, phase):
        s, k = self.sim, self.sim.skill
        k.managed = True
        k.status = 'running'
        k.moving = False
        self._home_start = s.data.qpos[s.arm_qpos].copy()
        self._home_duration = max(
            1., 1.875 * float(np.max(np.abs(s.arm_home - self._home_start))) / .8)
        self._enter(phase)

    def _begin_push_from_home(self):
        k = self.sim.skill
        k.gripper_closed = False
        k.opening_target = .014
        k.orientation_target = np.array(BowlStorage.PUSH_ORIENTATION)
        lo, hi = collision_bounds(self.sim.model, self.sim.data, [self._handle])
        self._push_x = float((lo[0] + hi[0]) * .5)
        self._push_z = float((lo[2] + hi[2]) * .5)
        self._push_front_y = float(lo[1] - .080)
        joint = self.sim.model.joint('cabinet_bottom_level')
        travel = float(joint.range[1] - self.sim.data.qpos[int(joint.qposadr[0])])
        self._push_end_y = float(hi[1] + travel + .020)
        self._move('push_approach', [self._push_x, self._push_front_y, self._push_z])

    def _update(self, dt):
        s, k = self.sim, self.sim.skill
        now = float(s.data.time)
        if self.status == 'failed':
            return
        if self.status == 'succeeded':
            facts = self.facts.state()
            if not (facts['all_stored'] and facts['drawer_closed'] and facts['hand_empty']):
                self._fail('Terminal physical storage conditions were lost')
            return
        if now - self.started_at > 480.:
            self._fail('Continuous storage watchdog exceeded 480 simulated seconds')
            return
        if k.status == 'failed':
            self._fail(k.failure_reason)
            return
        if self.mode == 'settling':
            if now >= 1.:
                self._begin_item()
            return
        if self.mode == 'grasp':
            if self.phase != k.phase:
                self.phase = k.phase
                self.events.append({'time_s': now, 'target': self.target, 'phase': k.phase})
            ok = k.phase in ('hold', 'done') and all(k.grasp_postconditions()[p] for p in PHYSICAL)
            if ok:
                if self.hold_started is None:
                    self.hold_started = now
                self.grasp_hold_s = now - self.hold_started
            else:
                self.hold_started = None
                self.grasp_hold_s = 0.
            if k.status == 'succeeded' and self.grasp_hold_s >= 2.:
                self._start_transfer()
            return
        self._check_robot_contacts()
        if self.status == 'failed':
            return
        if self.mode in ('raise', 'rotate', 'transfer', 'entry', 'insert', 'lower'):
            bilateral, _ = k._contacts()
            on_floor = self.mode == 'lower' and self.facts.measure(self.target)['supported']
            relative = s.data.site_xmat[k.tcp].reshape(3, 3).T @ (s.data.xpos[k.target_body] - self._tcp())
            held = bilateral and np.linalg.norm(relative - self._hand_relative) < .015
            self._lost_contact_s = 0. if held or on_floor else self._lost_contact_s + dt
            if self._lost_contact_s > .15:
                self._fail('Lost bilateral grip or slipped while carrying')
                return
            if on_floor:
                self._release()
                return
        if self.mode == 'release':
            item = self.facts.measure(self.target)
            opened = float(s.data.qpos[s.finger_qpos].sum()) >= self._release_opening_required
            # Support is a transition event: opening can briefly unload a
            # contact. Strict support is checked again after hand withdrawal.
            self._release_support_seen |= item['supported']
            ready = item['inside'] and self._release_support_seen and opened
            self._stable_s = self._stable_s + dt if ready else 0.
            if self._stable_s >= .2:
                point = self._tcp()
                point[2] = 1.03
                self._move('withdraw', point)
            elif now - self._phase_time > 4.:
                self._fail('Gripper did not begin opening with the object contained and floor-supported')
            return
        if self.mode == 'verify_release':
            # An object may lean against an open finger. Verify actual
            # detachment after withdrawing, never infer it from opening.
            ready = self.facts.measure(self.target)['stored']
            self._stable_s = self._stable_s + dt if ready else 0.
            if self._stable_s >= .4:
                self.released = True
                self._move('clear', [.50, -.10, 1.05])
            elif now - self._phase_time > 3.:
                self._fail('Object did not settle, remain contained and detach after withdrawal')
            return
        if self.mode == 'verify':
            facts = self.facts.state()
            valid = facts['all_stored'] and facts['drawer_closed'] and facts['hand_empty']
            home = np.max(np.abs(s.data.qpos[s.arm_qpos] - s.arm_home)) < .03
            self.verified_hold_s = self.verified_hold_s + dt if valid and home else 0.
            if self.verified_hold_s >= self.REQUIRED_FINAL_HOLD_S:
                self.status = 'succeeded'
                self._hold('done')
                k.status = 'idle'
            elif now - self._phase_time > self.FINAL_SETTLE_TIMEOUT_S:
                self._fail(
                    f'Final physical quiet hold reached {self.verified_hold_s:.3f}s/'
                    f'{self.REQUIRED_FINAL_HOLD_S:.3f}s within the '
                    f'{self.FINAL_SETTLE_TIMEOUT_S:.1f}s settling deadline')
            return
        if self.mode in ('prepush_home', 'return_home'):
            elapsed = now - self._phase_time
            u = min(1., elapsed / self._home_duration)
            k.arm_target = self._home_start + u**3 * (10 - 15*u + 6*u*u) * (s.arm_home - self._home_start)
            if elapsed >= self._home_duration and np.max(np.abs(s.data.qpos[s.arm_qpos] - s.arm_home)) < .03:
                if self.mode == 'prepush_home':
                    self._begin_push_from_home()
                else:
                    self._hold('verify')
            elif elapsed > self._home_duration + 2.:
                self._fail('Robot did not return home')
            return
        if self.mode == 'push_drawer':
            cabinet = s.cabinet_state()
            closed = (cabinet['drawer_position_m'] >= -.001
                      and abs(cabinet['drawer_velocity_m_s']) < .005)
            self._stable_s = self._stable_s + dt if closed else 0.
            if self._stable_s >= .3 and self.push_contact_peak_n >= .1:
                self._move('push_withdraw', [self._push_x, self._push_front_y, self._push_z], speed=.04)
                return
        if not k.arrived:
            return
        slot = self._slot()
        floor_z = s.cabinet_state()['interior_min_m'][2]
        if self.mode == 'raise':
            angle = np.pi / 2
            k.orientation_target = np.array([0., np.cos(angle/2), np.sin(angle/2), 0.])
            self._move('rotate', self._tcp())
        elif self.mode == 'rotate':
            xy = slot.copy()
            if self.target in ('spoon', 'pen'):
                xy[1] = s.cabinet_state()['interior_min_m'][1] + .085
            elif self.target == 'bottle':
                xy = slot + [0., -.065]
            item = self.facts.measure(self.target)
            self._move('transfer', self._align(xy, item['bounds_min_m'][2]))
        elif self.mode == 'transfer':
            xy = (np.array(self.facts.measure(self.target)['bounds_min_m'][:2])
                  + self.facts.measure(self.target)['bounds_max_m'][:2]) * .5
            # Low objects enter below the housing roof before moving rearward.
            clearance = .060 if self._low_profile else .020
            self._move('entry', self._align(xy, floor_z + clearance), speed=.05)
        elif self.mode == 'entry':
            bottom = self.facts.measure(self.target)['bounds_min_m'][2]
            self._move('insert', self._align(slot, bottom), speed=.04)
        elif self.mode == 'insert':
            if self._low_profile:
                # Small rigid items settle through a real 6 cm gravity drop;
                # the forearm never has to penetrate the drawer walls.
                self._release()
            else:
                self._move('lower', self._align(slot, floor_z - .0005), speed=.025)
        elif self.mode == 'lower':
            self._fail('Reached floor approach without actual drawer support')
        elif self.mode == 'withdraw':
            self._hold('verify_release')
        elif self.mode == 'clear':
            self._finish_item()
        elif self.mode == 'push_approach':
            self._move('push_drawer', [self._push_x, self._push_end_y, self._push_z], speed=.025)
        elif self.mode == 'push_drawer':
            self._fail('Drawer not physically closed through verified handle contact')
        elif self.mode == 'push_withdraw':
            self._start_home('return_home')

    def advance(self, duration_s=.1):
        dt = float(self.sim.model.opt.timestep)
        steps = round(duration_s / dt)
        if steps < 1 or not np.isclose(steps * dt, duration_s):
            raise ValueError('Duration must be a positive multiple of the physics timestep')
        for _ in range(steps):
            self._update(dt)
            self.sim.step(dt)

    def state(self):
        facts = self.facts.state()
        return {
            **facts, 'scene_mode': 'continuous_drawer_storage',
            'target': self.target, 'targets': list(self.targets),
            'phase': 'hold' if self.mode == 'grasp' and self.phase == 'done' else self.phase,
            'controller_status': self.status,
            'current_index': self.current_index, 'completed_targets': list(self.completed_targets),
            'stored_count': sum(item['stored'] for item in facts['objects']),
            'cabinet': self.sim.cabinet_state(), 'time_s': float(self.sim.data.time),
            'failure_reason': self.failure_reason, 'finished': self.status != 'running',
            'outcome': None if self.status == 'running' else self.status == 'succeeded',
            'grasp_hold_s': self.grasp_hold_s, 'verified_hold_s': self.verified_hold_s,
            'final_settle_timeout_s': self.FINAL_SETTLE_TIMEOUT_S,
            'settings': self.sim.settings, 'push_contact_peak_n': self.push_contact_peak_n,
            'joint_positions_rad': self.sim.data.qpos[self.sim.arm_qpos].tolist(),
            'gripper_opening_m': float(self.sim.data.qpos[self.sim.finger_qpos].sum()),
            'lift_m': None if self.sim.skill.reference_z is None else float(
                self.sim.data.xpos[self.sim.skill.target_body, 2] - self.sim.skill.reference_z),
            'warnings': self.sim.warning_names(),
            'scene_resets_during_mission': self.sim.scene_reset_count - self.initial_reset_count,
        }


def _file_sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    source_paths = [root / name for name in (
        'tabletop_storage.py', 'drawer_storage_checks.py', 'grasp_tabletop.py',
        'grasp.py', 'simulation.py', 'grasp_study_physics.py', 'grasp_study_objects.py',
        'scene.xml', 'vendor/panda/panda.xml',
        'vendor/libero/turbosquid_objects/wine_rack/wine_rack.xml',
        'assets/grasp_study/tabletop_cabinet.xml')]
    source_paths.extend(root / spec.asset_path for spec in STUDY_OBJECTS.values())
    asset_roots = {root / 'vendor/panda', root / 'vendor/libero/turbosquid_objects/wine_rack'}
    asset_roots.update((root / spec.asset_path).parent for spec in STUDY_OBJECTS.values())
    source_paths.extend(path for directory in asset_roots for path in directory.rglob('*')
                        if path.is_file() and not path.name.startswith('.')
                        and '__pycache__' not in path.parts)
    source_paths = sorted(set(source_paths))
    hashes = {str(path.relative_to(root)): _file_sha256(path) for path in source_paths}
    session = StorageSession()
    model_path = args.output / 'model.mjb'
    mujoco.mj_saveModel(session.sim.model, str(model_path), None)
    model_hash = _file_sha256(model_path)
    times, positions, velocities, controls, predicates, support_forces = [], [], [], [], [], []
    predicate_names = ('inside', 'supported', 'detached', 'stable')
    try:
        last = None
        while session.status == 'running':
            session.advance(.1)
            times.append(float(session.sim.data.time))
            positions.append(session.sim.data.qpos.copy())
            velocities.append(session.sim.data.qvel.copy())
            controls.append(session.sim.data.ctrl.copy())
            objects = session.facts.objects()
            predicates.append([[item[name] for name in predicate_names] for item in objects])
            support_forces.append([item['support_force_n'] for item in objects])
            current = (session.target, session.phase)
            if current != last:
                print(f'{session.sim.data.time:.2f}s {session.target}: {session.phase}', flush=True)
                last = current
        result = {**session.state(), 'events': session.events,
                  'contact_reports': session.contact_reports,
                  'physics_timestep_s': float(session.sim.model.opt.timestep),
                  'experiment_id': 'experiment_02_five_object_storage',
                  'experiment_label': 'Experiment 02',
                  'recording_rate_hz': 10., 'recorded_frames': len(times),
                  'release_strategy': 'open_and_withdraw_then_verify_actual_detachment',
                  'compiled_model': {'path': 'model.mjb', 'sha256': model_hash},
                  'mujoco_version': mujoco.__version__, 'source_sha256': hashes}
        (args.output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
        np.savez_compressed(args.output / 'trajectory.npz', time_s=times, qpos=positions,
                            qvel=velocities, ctrl=controls,
                            object_order=np.array(DrawerFacts.TARGETS),
                            predicate_names=np.array(predicate_names),
                            predicates=np.array(predicates, dtype=bool),
                            support_force_n=support_forces)
        print(json.dumps(result, indent=2))
        if session.status != 'succeeded':
            raise SystemExit(1)
    finally:
        session.close()


if __name__ == '__main__':
    main()
