"""Read-only physical facts for the continuous five-object drawer mission."""

import mujoco
import numpy as np

from collisions import collision_bounds


class DrawerFacts:
    """Observe refreshed MuJoCo state, independently of controller bookkeeping.

    The caller refreshes dynamics after stepping. Observation never steps,
    forwards, changes poses, or creates constraints. Contact tolerance is 1 mm;
    stability means speeds below 1 cm/s and 0.1 rad/s. Direct floor support and
    force-bearing hand contact use a 1 µN numerical force threshold.
    """

    TARGETS = ('spoon', 'pen', 'bowl', 'glass', 'bottle')
    CONTACT_TOLERANCE_M = .001
    FORCE_EPSILON_N = 1e-6
    LINEAR_SPEED_LIMIT_M_S = .01
    ANGULAR_SPEED_LIMIT_RAD_S = .1
    CLOSED_POSITION_TOLERANCE_M = .003
    CLOSED_SPEED_LIMIT_M_S = .005

    def __init__(self, sim):
        self.model = sim.model
        self.data = sim.data
        model = self.model
        collidable = (model.geom_contype | model.geom_conaffinity) != 0
        self._bodies = tuple(model.body(f'{name}_object').id for name in self.TARGETS)
        self._geoms = tuple(np.flatnonzero((model.geom_bodyid == body) & collidable)
                            for body in self._bodies)
        supported_types = (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER,
                           mujoco.mjtGeom.mjGEOM_CAPSULE)
        for geoms in self._geoms:
            if not len(geoms) or not np.isin(model.geom_type[geoms], supported_types).all():
                raise ValueError('Drawer observations require primitive collision geometry')
        self._dofs = tuple(int(model.joint(f'{name}_free').dofadr[0])
                           for name in self.TARGETS)
        self._target_indices = {name: index for index, name in enumerate(self.TARGETS)}
        self._geom_objects = np.full(model.ngeom, -1, dtype=np.int32)
        for index, geoms in enumerate(self._geoms):
            self._geom_objects[geoms] = index
        hand_bodies = tuple(model.body(name).id for name in (
            'robot_left_finger', 'robot_right_finger', 'robot_hand'))
        self._hand_geoms = np.isin(model.geom_bodyid, hand_bodies) & collidable
        self._floor = model.geom('cabinet_drawer_floor').id
        self._region = model.site('cabinet_bottom_region').id
        joint = model.joint('cabinet_bottom_level')
        self._drawer_qpos = int(joint.qposadr[0])
        self._drawer_dof = int(joint.dofadr[0])
        self._drawer_closed_position = float(joint.range[1])
        self._support = np.zeros(len(self.TARGETS))
        self._attached = np.zeros(len(self.TARGETS), dtype=bool)
        self._force = np.zeros(6)
        self._force_projection = np.empty(3)
        self._local_center = np.empty(3)
        self._world_center = np.empty(3)
        self._half_bounds = np.empty(3)
        self._local_extent = np.empty(3)
        self._abs_rotation = np.empty((3, 3))

    def _contacts(self):
        self._support.fill(0.)
        self._attached.fill(False)
        floor_center = self.data.geom_xpos[self._floor]
        floor_up = self.data.geom_xmat[self._floor].reshape(3, 3)[:, 2]
        floor_top = self.model.geom_size[self._floor, 2]
        for index, contact in enumerate(self.data.contact):
            if contact.efc_address < 0:
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            object_a, object_b = self._geom_objects[a], self._geom_objects[b]
            if object_a < 0 and object_b < 0:
                continue
            on_floor = a == self._floor or b == self._floor
            on_hand = self._hand_geoms[a] or self._hand_geoms[b]
            if not (on_floor or on_hand):
                continue
            mujoco.mj_contactForce(self.model, self.data, index, self._force)
            if self._force[0] <= self.FORCE_EPSILON_N:
                continue
            if on_hand:
                if object_a >= 0:
                    self._attached[object_a] = True
                if object_b >= 0:
                    self._attached[object_b] = True
            if on_floor:
                # Side/underside contacts with the floor are not top support.
                height = sum((contact.pos[axis] - floor_center[axis]) * floor_up[axis]
                             for axis in range(3))
                if height < floor_top - self.CONTACT_TOLERANCE_M:
                    continue
                np.matmul(contact.frame.reshape(3, 3), floor_up,
                          out=self._force_projection)
                upward = float(np.dot(self._force[:3], self._force_projection))
                # mj_contactForce is the force exerted on geom2 by geom1.
                if b == self._floor:
                    upward = -upward
                object_index = object_b if a == self._floor else object_a
                self._support[object_index] += max(0., upward)

    def _measure(self, index):
        body = self._bodies[index]
        low, high = collision_bounds(self.model, self.data, self._geoms[index])
        rotation = self.data.site_xmat[self._region].reshape(3, 3)
        np.add(low, high, out=self._world_center)
        self._world_center *= .5
        self._world_center -= self.data.site_xpos[self._region]
        np.matmul(rotation.T, self._world_center, out=self._local_center)
        np.subtract(high, low, out=self._half_bounds)
        self._half_bounds *= .5
        np.abs(rotation.T, out=self._abs_rotation)
        np.matmul(self._abs_rotation, self._half_bounds, out=self._local_extent)
        # Test the entire world collision AABB against the live oriented region.
        # This remains conservative if the cabinet itself is rotated.
        inside = all(abs(self._local_center[axis]) + self._local_extent[axis]
                     <= self.model.site_size[self._region, axis] + self.CONTACT_TOLERANCE_M
                     for axis in range(3))
        dof = self._dofs[index]
        linear = self.data.qvel[dof:dof + 3]
        angular = self.data.qvel[dof + 3:dof + 6]
        linear_speed = float(np.linalg.norm(linear))
        angular_speed = float(np.linalg.norm(angular))
        stable = (linear_speed < self.LINEAR_SPEED_LIMIT_M_S
                  and angular_speed < self.ANGULAR_SPEED_LIMIT_RAD_S)
        supported = bool(self._support[index] > self.FORCE_EPSILON_N)
        detached = not bool(self._attached[index])
        return {
            'target': self.TARGETS[index],
            'name': f'{self.TARGETS[index]}_object',
            'position_m': self.data.xpos[body].tolist(),
            'quaternion_wxyz': self.data.xquat[body].tolist(),
            'bounds_min_m': low.tolist(),
            'bounds_max_m': high.tolist(),
            'inside': bool(inside),
            'supported': supported,
            'stable': stable,
            'detached': detached,
            'stored': bool(inside and supported and stable and detached),
            'support_force_n': float(self._support[index]),
            'linear_speed_m_s': linear_speed,
            'angular_speed_rad_s': angular_speed,
        }

    def measure(self, target):
        """Measure one canonical target using the current solved contacts."""
        index = self._target_indices[target]
        self._contacts()
        return self._measure(index)

    def objects(self):
        """Measure all five objects in mission order, scanning contacts once."""
        self._contacts()
        return [self._measure(index) for index in range(len(self.TARGETS))]

    def state(self):
        objects = self.objects()
        position = float(self.data.qpos[self._drawer_qpos])
        speed = float(self.data.qvel[self._drawer_dof])
        return {
            'objects': objects,
            'all_stored': all(item['stored'] for item in objects),
            'hand_empty': all(item['detached'] for item in objects),
            'drawer_closed': bool(
                abs(position - self._drawer_closed_position) <= self.CLOSED_POSITION_TOLERANCE_M
                and abs(speed) < self.CLOSED_SPEED_LIMIT_M_S),
        }
