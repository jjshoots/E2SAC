import math
import numpy as np

from pybullet_utils import bullet_client
from railway_drone.environment.utilities import *


class RailObject:
    def __init__(
        self,
        p: bullet_client.BulletClient,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        rail_mesh_ids: np.ndarray,
    ):

        self.p = p
        self.rail_mesh_ids = rail_mesh_ids
        self.tex_id = None
        self.exists = True

        rail = RailSingle(p, start_pos, start_orn, self.rail_mesh_ids, 0)

        self.Ids = np.array([rail.Id])
        self.head = rail.get_end(0)
        self.tail = rail.get_end(1)

    def handle_rail_bounds(self, drone_xy: np.ndarray, spawn_id: int = -1) -> int:
        """
        extends and deletes rails on the fly
            drone_xy: 2d array for drone x and y positions
            spawn_id:
                -2 for no spawn
                -1 for spawn if required
                +0 for spawn straight rail
                +1 for spawn left
                +2 for spawn right
        """
        dis2head = np.sum((self.head.base_pos[:2] - drone_xy) ** 2) ** 0.5
        dis2tail = np.sum((self.tail.base_pos[:2] - drone_xy) ** 2) ** 0.5

        # delete the head if it's too far and get the new one
        if dis2head > 20:
            deleted, self.head, still_exists = self.head.delete(0)
            if not still_exists:
                self.exists = False
                return -1
            self.Ids = [id for id in self.Ids if id not in deleted]

        # if the tail is too far away, just delete it
        if dis2tail > 100:
            deleted, self.tail, still_exists = self.tail.delete(0)
            if not still_exists:
                self.exists = False
                return -1
            self.Ids = [id for id in self.Ids if id not in deleted]

        # create new tail if it's too near if allowed
        if not spawn_id == -2:
            if dis2tail < 40 or spawn_id != -1:
                # if don't have spawn direction, just random
                if spawn_id == -1:
                    spawn_id = np.random.randint(0, 3)

                self.tail.add_child(self.rail_mesh_ids, spawn_id)
                self.tail = self.tail.get_end(1)
                self.Ids = np.append(self.Ids, self.tail.Id)
                if self.tex_id is not None:
                    self.p.changeVisualShape(
                        self.tail.Id, -1, textureUniqueId=self.tex_id
                    )

                return spawn_id

        return -1

    def change_rail_texture(self, tex_id):
        self.tex_id = tex_id
        for id in self.Ids:
            self.p.changeVisualShape(id, -1, textureUniqueId=self.tex_id)


class RailSingle:
    def __init__(
        self,
        p: bullet_client.BulletClient,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        rail_mesh_ids: np.ndarray,
        spawn_id: int,
        parent=None,
    ):

        self.p = p

        # values are obtained from the readme in the rails models folder
        ROT = 0.1 * math.pi
        c, s, = np.cos(
            -start_orn[-1]
        ), np.sin(-start_orn[-1])
        rot_mat = np.array([[c, -s], [s, c]]).T

        if spawn_id == 0:
            end_offset = np.matmul(rot_mat, np.array([0, 10.125]))
            end_offset = np.array([*end_offset, 0.0])

            self.start_pos = start_pos
            self.start_orn = start_orn
            self.end_pos = self.start_pos + 2 * end_offset
            self.end_orn = self.start_orn
            self.base_pos = self.start_pos + end_offset
            self.base_orn = self.start_orn
        elif spawn_id == 1:
            end_offset = np.matmul(rot_mat, np.array([-2.209, 20.186]))
            end_offset = np.array([*end_offset, 0.0])
            base_offset = np.matmul(rot_mat, np.array([0.0, 10.061]))
            base_offset = np.array([*base_offset, 0.0])

            self.start_pos = start_pos
            self.start_orn = start_orn
            self.end_pos = self.start_pos + end_offset
            self.end_orn = self.start_orn + np.array([0, 0, ROT])
            self.base_pos = self.start_pos + base_offset
            self.base_orn = self.start_orn
        elif spawn_id == 2:
            end_offset = np.matmul(rot_mat, np.array([2.209, 20.186]))
            end_offset = np.array([*end_offset, 0.0])
            base_offset = np.matmul(rot_mat, np.array([0.0, 10.061]))
            base_offset = np.array([*base_offset, 0.0])

            self.start_pos = start_pos
            self.start_orn = start_orn
            self.end_pos = self.start_pos + end_offset
            self.end_orn = self.start_orn + np.array([0, 0, -ROT])
            self.base_pos = self.start_pos + base_offset
            self.base_orn = self.start_orn
        else:
            print("IMPORTING UNKNOWN RAIL OBJECT")
            self.start_pos = start_pos
            self.start_orn = start_orn
            self.end_pos = start_pos
            self.end_orn = start_orn
            self.base_pos = start_pos
            self.base_orn = start_orn

        self.Id = loadOBJ(
            self.p,
            visualId=rail_mesh_ids[spawn_id],
            basePosition=self.base_pos,
            baseOrientation=self.p.getQuaternionFromEuler(self.base_orn),
        )

        # linked = [parent, child]
        self.linked = [None, None]
        self.linked[0] = parent

    def add_child(self, rail_mesh_ids, spawn_id):
        """adds a single child to the end of the rail"""
        self.linked[1] = RailSingle(
            self.p, self.end_pos, self.end_orn, rail_mesh_ids, spawn_id, self
        )

    def delete(self, dir: int):
        """deletes self and all connected links in dir, 0 for parent, 1 for child"""
        still_exists = True
        deleted = np.array([])
        node = self
        # traverse to the end of the chain
        while node.linked[dir] is not None:
            node = node.linked[dir]

        while True:
            # delete the node model
            self.p.removeBody(node.Id)
            deleted = np.append(deleted, node.Id)

            # move up the chain by a step and dereference
            if node.linked[1 - dir] is not None:
                node = node.linked[1 - dir]
                node.linked[dir] = None

                if node.Id == self.linked[1 - dir].Id:
                    break
            else:
                # this indicates that there is no more rails on this line
                still_exists = False
                break

        return deleted, node, still_exists

    def get_end(self, dir: int):
        """gets the end of the track, depending on dir, 0 for parent, 1 for child"""
        node = self
        while node.linked[dir] is not None:
            node = node.linked[dir]

        return node
