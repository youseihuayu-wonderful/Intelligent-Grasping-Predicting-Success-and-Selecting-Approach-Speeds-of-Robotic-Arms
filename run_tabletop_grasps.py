"""Observe independent five-object grasp-and-hold attempts in the CURRENT scene.

This is a task reproduction, not an exact replay of the earlier Experiment01 scene.
Each target gets a fresh scene; nothing resets or rewrites object poses mid-attempt.
"""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from grasp import GraspParameters
from grasp_study_objects import STUDY_OBJECTS
from grasp_study_physics import PHYSICAL
from grasp_tabletop import FiveObjectSimulation

REQUIRED_HOLD_S = 2.0
WATCHDOG_S = 24.0
NEIGHBOR_DRIFT_M = .015
NEIGHBOR_CLEARANCE_M = .001


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def run_target(target, output, show_viewer):
    sim = FiveObjectSimulation(target)
    try:
        initial = sim.object_states()
        initial_xy = {item['target']: np.array(item['position_m'][:2]) for item in initial}
        max_drift = 0.
        max_clearance = 0.
        start = float(sim.data.time)
        hold_started = None
        hold_s = 0.
        reason = sim.skill.precondition_error('grasp_and_hold')
        if reason is None:
            sim.start_task('grasp', GraspParameters())
        if show_viewer:
            import mujoco.viewer
            viewer_context = mujoco.viewer.launch_passive(sim.model, sim.data)
        else:
            viewer_context = nullcontext(None)
        with viewer_context as viewer, (output / f'{target}_trace.jsonl').open('x') as trace:
            dt = float(sim.model.opt.timestep)
            next_sample = start
            last_wall = time.monotonic()
            while True:
                now = float(sim.data.time)
                predicates = sim.skill.grasp_postconditions()
                eligible = (sim.skill.status in ('running', 'succeeded')
                            and sim.skill.phase in ('hold', 'done')
                            and all(predicates[name] for name in PHYSICAL))
                if eligible:
                    if hold_started is None:
                        hold_started = now
                    hold_s = now - hold_started
                else:
                    hold_started, hold_s = None, 0.
                complete = (reason is not None or sim.skill.status == 'failed'
                            or (sim.skill.status == 'succeeded' and hold_s >= REQUIRED_HOLD_S))
                if now - start >= WATCHDOG_S and not complete:
                    reason = 'Independent observer watchdog exceeded 24 simulation seconds'
                    complete = True
                if now >= next_sample - 1e-10 or complete:
                    objects = sim.object_states()
                    for item in objects:
                        if item['target'] != target:
                            drift = np.linalg.norm(np.array(item['position_m'][:2]) - initial_xy[item['target']])
                            max_drift = max(max_drift, float(drift))
                            max_clearance = max(max_clearance, abs(item['table_clearance_m']))
                    sample = {
                        'time_s': now, 'phase': sim.skill.phase, 'status': sim.skill.status,
                        'postconditions': predicates, 'verified_hold_s': hold_s,
                        'objects': objects,
                    }
                    trace.write(json.dumps(sample, allow_nan=False) + '\n')
                    next_sample = now + .02
                    if viewer is not None:
                        if not viewer.is_running() and not complete:
                            reason = 'Viewer closed before the attempt completed'
                            complete = True
                        else:
                            viewer.sync()
                            time.sleep(max(0., .02 - (time.monotonic() - last_wall)))
                            last_wall = time.monotonic()
                if complete:
                    break
                sim.step(dt)
        neighbors_ok = max_drift <= NEIGHBOR_DRIFT_M and max_clearance <= NEIGHBOR_CLEARANCE_M
        warnings = sim.warning_names()
        grasp_success = sim.skill.status == 'succeeded' and hold_s >= REQUIRED_HOLD_S
        result = {
            'target': target, 'success': bool(grasp_success and neighbors_ok and not warnings and reason is None),
            'grasp_success': bool(grasp_success), 'neighbors_undisturbed': bool(neighbors_ok),
            'controller_status': sim.skill.status,
            'failure_reason': reason or sim.skill.failure_reason,
            'verified_hold_s': hold_s, 'execution_sim_s': float(sim.data.time) - start,
            'postconditions': sim.skill.grasp_postconditions(),
            'initial_objects': initial, 'final_objects': sim.object_states(),
            'max_non_target_xy_drift_m': max_drift,
            'max_non_target_abs_table_clearance_m': max_clearance,
            'warnings': warnings, 'physics_timestep_s': float(sim.model.opt.timestep),
            'settings': sim.settings,
        }
        write_json(output / f'{target}.json', result)
        print(f'{target}: success={result["success"]}, hold={hold_s:.3f} simulation s', flush=True)
        return result
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New output directory; never overwrite evidence')
    parser.add_argument('--targets', nargs='+', choices=tuple(STUDY_OBJECTS), default=list(STUDY_OBJECTS))
    parser.add_argument('--viewer', action='store_true', help='Display the actual running physics in a passive MuJoCo viewer')
    args = parser.parse_args()
    if len(set(args.targets)) != len(args.targets):
        parser.error('--targets must not contain duplicates')
    args.output.mkdir(parents=True, exist_ok=False)
    results = [run_target(target, args.output, args.viewer) for target in args.targets]
    root = Path(__file__).resolve().parent
    names = ('run_tabletop_grasps.py', 'grasp_tabletop.py', 'grasp.py', 'simulation.py',
             'grasp_study_physics.py', 'grasp_study_objects.py', 'collisions.py',
             'assets/grasp_study/tabletop_cabinet.xml', 'scene.xml')
    hashes = {}
    for name in names:
        with (root / name).open('rb') as stream:
            hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = {
        'task': 'current_scene_independent_five_object_grasp_and_hold',
        'historical_exact_replay': False, 'mujoco_version': mujoco.__version__,
        'required_continuous_hold_s': REQUIRED_HOLD_S,
        'success_unit': 'One independent target grasp, continuous physical hold and undisturbed neighbors',
        'neighbor_checks_sample_period_s': .02,
        'neighbor_xy_drift_limit_m': NEIGHBOR_DRIFT_M,
        'neighbor_abs_table_clearance_limit_m': NEIGHBOR_CLEARANCE_M,
        'hold_predicates': list(PHYSICAL), 'hold_observer_period': 'Every physics timestep',
        'scene_policy': 'Fresh complete five-object scene before every target; no mid-attempt resets',
        'source_sha256': hashes, 'runs': results,
        'passed': all(result['success'] for result in results),
        'limits': ['Current Experiment02-era cabinet/layout, not the original Experiment01 exact source.',
                   'Nominal demonstrations, not a statistical success-rate estimate.',
                   'Not storage completion and not classifier labels or a new training batch.'],
    }
    write_json(args.output / 'verification.json', report)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
