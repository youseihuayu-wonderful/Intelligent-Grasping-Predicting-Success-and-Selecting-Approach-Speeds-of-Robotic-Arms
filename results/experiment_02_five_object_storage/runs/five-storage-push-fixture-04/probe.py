"""Run from repository root: isolated PUSH fixture, NOT a robot-storage trial."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))
import mujoco
import numpy as np
from collisions import CollisionTracker, collision_bounds
from tabletop_storage import StorageSession, StorageServo, PACKING_XY_M
from grasp_study_objects import STUDY_OBJECTS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
s = StorageSession()
try:
    m, d = s.sim.model, s.sim.data
    for name, offset in PACKING_XY_M.items():
        q = int(m.joint(f'{name}_free').qposadr[0])
        d.qpos[q+3:q+7] = [2**-.5, 0, 0, 2**-.5] if name == 'pen' else [1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        low, high = collision_bounds(m, d, s.sim._object_geoms[name])
        xy = d.site_xpos[s._region, :2] + offset
        d.qpos[q:q+2] += xy - (low[:2] + high[:2])*.5
        d.qpos[q+2] += .7425 - low[2]
    mujoco.mj_forward(m, d)
    s.sim.step(1.)
    s.target = 'glass'
    s.current_index = 5
    s.sim.target = STUDY_OBJECTS['glass'].grasp_target
    s.sim.skill = StorageServo(s.sim)
    s.sim.skill.arm_target = d.qpos[s.sim.arm_qpos].copy()
    s.sim.skill.storage = s
    s.sim.collisions = CollisionTracker(m, d, s.sim.skill.target_body, s.sim.skill.fingers, s.sim.skill.table)
    s.released = True
    s._start_push()
    times, qposes, velocities, controls = [], [], [], []
    last = None
    while s.status == 'running':
        s.advance(.1)
        times.append(float(d.time))
        qposes.append(d.qpos.copy())
        velocities.append(d.qvel.copy())
        controls.append(d.ctrl.copy())
        if s.phase != last:
            print(round(d.time, 2), s.phase, flush=True)
            last = s.phase
    result = {**s.state(), 'experiment_id': 'experiment_02_push_fixture',
              'task_scope': 'push_only_preloaded_fixture_NOT_a_grasp_and_storage_trial',
              'initialization': 'Objects initialized inside drawer, then settled by gravity before push',
              'events': s.events}
    (args.output/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(args.output/'trajectory.npz', time_s=times, qpos=qposes,
                        qvel=velocities, ctrl=controls)
    print(json.dumps(result, indent=2))
    if s.status != 'succeeded':
        raise SystemExit(1)
finally:
    s.close()
