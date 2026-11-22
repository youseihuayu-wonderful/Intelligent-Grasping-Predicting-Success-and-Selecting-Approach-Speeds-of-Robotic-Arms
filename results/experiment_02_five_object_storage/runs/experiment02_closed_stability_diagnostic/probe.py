"""Continue actual recorded closed state; NOT an end-to-end storage trial."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))
import mujoco
import numpy as np
from grasp_tabletop import FiveObjectSimulation
from drawer_storage_checks import DrawerFacts

source = Path('/tmp/five-storage-probe-12')
out = Path('/tmp/experiment02_closed_stability_diagnostic')
out.mkdir(exist_ok=False)
a = np.load(source/'trajectory.npz')
s = FiveObjectSimulation('spoon')
try:
    s.data.qpos[:] = a['qpos'][-1]
    s.data.qvel[:] = a['qvel'][-1]
    s.data.ctrl[:] = a['ctrl'][-1]
    s.data.time = float(a['time_s'][-1])
    mujoco.mj_forward(s.model, s.data)
    s.skill.arm_target = (s.data.ctrl[s.arm_actuators]
        - s.data.qfrc_bias[s.skill.arm_dof]/s.model.actuator_gainprm[s.arm_actuators, 0])
    s.skill.gripper_closed = False
    s.skill.opening_target = .014
    facts = DrawerFacts(s)
    start = float(s.data.time)
    streak = longest = 0.
    times, qposes, velocities, controls = [], [], [], []
    failed = {name: {p: 0 for p in ('inside','supported','detached','stable')}
              for name in DrawerFacts.TARGETS}
    steps = 0
    while s.data.time-start < 20. and streak < 3.:
        s.step(.002)
        state = facts.state()
        valid = state['all_stored'] and state['drawer_closed'] and state['hand_empty']
        streak = streak+.002 if valid else 0.
        longest = max(longest, streak)
        for item in state['objects']:
            for name in failed[item['target']]:
                failed[item['target']][name] += not item[name]
        steps += 1
        if steps % 50 == 0:
            times.append(float(s.data.time));qposes.append(s.data.qpos.copy())
            velocities.append(s.data.qvel.copy());controls.append(s.data.ctrl.copy())
    result = {'task_scope':'closed-state stability continuation from a recorded failed final state; no robot grasping performed',
              'source_recording':str(source),'initial_recorded_time_s':start,
              'continued_simulation_s':float(s.data.time)-start,
              'three_second_strict_hold_observed':streak>=3.,
              'longest_strict_hold_s':longest,'failed_physics_intervals':failed,
              'final_physical_facts':facts.state(),
              'solver':{'iterations':s.model.opt.iterations,'tolerance':s.model.opt.tolerance,
                        'solver_type':int(s.model.opt.solver),'timestep':s.model.opt.timestep}}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'trajectory.npz',time_s=times,qpos=qposes,qvel=velocities,ctrl=controls)
    print(json.dumps(result,indent=2))
finally:
    s.close()
