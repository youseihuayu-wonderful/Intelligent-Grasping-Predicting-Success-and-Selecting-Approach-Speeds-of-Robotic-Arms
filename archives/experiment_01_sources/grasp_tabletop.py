"""Independent five-object tabletop demonstration, not a frozen-study replay."""

from pathlib import Path

import mujoco
import numpy as np

from collisions import CollisionTracker, collision_bounds
from grasp_study_objects import STUDY_OBJECTS
from grasp_study_physics import BASE, ExplorationSkill, StudySceneCase
from simulation import ARM_HOME, Simulation


ROOT = Path(__file__).resolve().parent
LAYOUT_XY_M = {
    'bowl': (.48, -.27),
    'bottle': (.50, -.08),
    'spoon': (.40, .10),
    'glass': (.62, -.10),
    'pen': (.38, -.08),
}
MASSES_KG = {'bowl': .06, 'bottle': .18, 'spoon': .025, 'glass': .15, 'pen': .01}


class FiveObjectSimulation(Simulation):
    """All five free objects coexist; reset restores the entire trial layout."""

    def __init__(self, target: str):
        if not isinstance(target, str) or target not in STUDY_OBJECTS:
            raise ValueError(f'Unknown tabletop target: {target!r}')
        x, y = LAYOUT_XY_M[target]
        self.settings = {
            **BASE, 'target': target, 'mass_kg': MASSES_KG[target],
            'x_m': x, 'y_m': y,
        }
        self.case = StudySceneCase('five_object_tabletop', target, 'shared_workspace',
                                   5646, (0., 0.), 0.)
        self.target = STUDY_OBJECTS[target].grasp_target
        scene = mujoco.MjSpec.from_file(str(ROOT / 'scene.xml'))
        scene.option.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD
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

        assets = [(name, ROOT / item.asset_path, True)
                  for name, item in STUDY_OBJECTS.items()]
        assets.extend((
            ('cabinet', ROOT / 'vendor/libero/articulated_objects/white_cabinet.xml', False),
            ('rack', ROOT / 'vendor/libero/turbosquid_objects/wine_rack/wine_rack.xml', False),
        ))
        for name, filename, movable in assets:
            if name not in ('bowl', 'bottle', 'cabinet', 'rack'):
                scene.worldbody.add_frame(name=f'{name}_mount')
            frame = scene.frame(f'{name}_mount')
            part = mujoco.MjSpec.from_file(str(filename))
            root = part.worldbody.bodies[0]
            root.name = 'root'
            part.body('object').pos[:] = 0
            if movable:
                root.add_freejoint(name='free')
                frame.pos[:] = [*LAYOUT_XY_M[name], .77]
                for geom in part.geoms:
                    if geom.contype or geom.conaffinity:
                        geom.friction[:] = [.8, .005, .0001]
            else:
                # Legacy reset/storage observers require these names, not furniture
                # inside the tabletop workspace.
                frame.pos[:] = [1.8, 1.8 if name == 'rack' else -1.8, .72]
            scene.attach(part, frame=frame, prefix=f'{name}_')
        bowl_material = scene.material('bowl_akita_black_bowl')
        bowl_material.rgba[:] = [.10, .12, .15, 1.]
        bowl_material.reflectance = 0
        bowl_material.specular = .1

        self.model = scene.compile()
        collidable = (self.model.geom_contype | self.model.geom_conaffinity) != 0
        self._object_geoms = {}
        for name, mass in MASSES_KG.items():
            body = self.model.body(f'{name}_object').id
            ratio = mass / float(self.model.body_mass[body])
            self.model.body_mass[body] *= ratio
            self.model.body_inertia[body] *= ratio
            self._object_geoms[name] = np.flatnonzero(
                (self.model.geom_bodyid == body) & collidable)
        fingers = [self.model.body(f'robot_{side}_finger').id for side in ('left', 'right')]
        finger_geoms = np.flatnonzero(np.isin(self.model.geom_bodyid, fingers) & collidable)
        self.model.geom_friction[finger_geoms] = [.8, .005, .0001]
        self._tabletop_geoms = np.array([self.model.geom('tabletop').id])
        self.data = mujoco.MjData(self.model)
        mujoco.mj_setConst(self.model, self.data)
        self.arm_qpos = np.array([self.model.joint(f'robot_joint{i}').qposadr[0]
                                  for i in range(1, 8)])
        self.arm_actuators = np.array([self.model.actuator(f'robot_actuator{i}').id
                                       for i in range(1, 8)])
        self.finger_qpos = np.array([self.model.joint(f'robot_finger_joint{i}').qposadr[0]
                                     for i in (1, 2)])
        self.gripper_actuator = self.model.actuator('robot_actuator8').id
        self.arm_home = ARM_HOME.copy()
        self.object_names = tuple(f'{name}_object' for name in STUDY_OBJECTS) + (
            'cabinet_cabinet_bottom',)
        self.renderer = None
        self.camera = mujoco.MjvCamera()
        self.options = mujoco.MjvOption()
        self.options.geomgroup[0] = 0
        self.options.geomgroup[3] = 0
        self.skill = ExplorationSkill(self)
        self.collisions = CollisionTracker(self.model, self.data, self.skill.target_body,
                                          self.skill.fingers, self.skill.table)
        self.reset()

    def reset(self) -> None:
        super().reset()
        table_low, table_high = collision_bounds(self.model, self.data, self._tabletop_geoms)
        bounds = []
        for name, geoms in self._object_geoms.items():
            low, _ = collision_bounds(self.model, self.data, geoms)
            qpos = int(self.model.joint(f'{name}_free').qposadr[0])
            self.data.qpos[qpos + 2] += table_high[2] + .0005 - low[2]
        mujoco.mj_forward(self.model, self.data)
        for name, geoms in self._object_geoms.items():
            low, high = collision_bounds(self.model, self.data, geoms)
            if np.any(low[:2] < table_low[:2]) or np.any(high[:2] > table_high[:2]):
                raise ValueError(f'{name} collision bounds leave the tabletop')
            for other, other_low, other_high in bounds:
                if np.all(low[:2] < other_high[:2]) and np.all(other_low[:2] < high[:2]):
                    raise ValueError(f'{name} and {other} collision bounds overlap')
            bounds.append((name, low, high))
        # Only gravity/contact dynamics settle the bodies. No pose is frozen or
        # rewritten after this initialization, including during grasp execution.
        self.step(.6)
        self.skill.reset()
        self.collisions.reset()

    def object_states(self) -> list[dict]:
        table_top = collision_bounds(self.model, self.data, self._tabletop_geoms)[1][2]
        return [
            {'target': name, **self.get_object_pose(f'{name}_object'),
             'table_clearance_m': float(
                 collision_bounds(self.model, self.data, geoms)[0][2] - table_top)}
            for name, geoms in self._object_geoms.items()
        ]
