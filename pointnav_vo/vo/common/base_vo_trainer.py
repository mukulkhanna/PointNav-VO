import os
import sys

import numpy as np
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.utils.visualizations import maps

from pointnav_vo.config.vo_config.default import get_config as get_vo_config
from pointnav_vo.rl.common.base_trainer_with_vo import BaseRLTrainerWithVO
from pointnav_vo.utils.geometry_utils import compute_global_state


def get_polar_angle(rotation):
    """
    Source: https://github.com/facebookresearch/habitat-lab/blob/main/habitat/tasks/nav/nav.py#L883-L894
    """
    heading_vector = quaternion_rotate_vector(rotation.inverse(), np.array([0, 0, -1]))

    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    z_neg_z_flip = np.pi

    return np.array(phi) + z_neg_z_flip


class BaseVOTrainer(BaseRLTrainerWithVO):
    def __init__(self, cfg_file):
        vo_config = get_vo_config(cfg_file, None)
        super().__init__(vo_config)
        super()._setup_vo_model(vo_config)
        super()._set_up_vo_obs_transformer()

    def get_topdown_map_with_trajectory(self, sim, positions=None):
        topdown_map = maps.get_topdown_map_from_sim(sim.sim)  # fetch topdown map
        topdown_map = maps.colorize_topdown_map(topdown_map)
        if positions is not None:
            agent_map_coord = maps.to_grid(  # maps 3D pose onto topdown map
                positions[0][2], positions[0][0], topdown_map.shape[0:2], sim.sim
            )
            path_points = [
                maps.to_grid(p[2], p[0], topdown_map.shape[0:2], sim=sim.sim)
                for p in positions
            ]
            maps.draw_path(topdown_map, path_points, (0, 255, 0), 4)
        else:
            agent_map_coord = maps.to_grid(  # maps 3D pose onto topdown map
                sim.sim.get_agent_state().position[2],
                sim.sim.get_agent_state().position[0],
                topdown_map.shape[0:2],
                sim.sim,
            )
        agent_map_angle = get_polar_angle(sim.sim.get_agent_state().rotation)

        maps.draw_agent(  # draw agent sprite
            image=topdown_map,
            agent_center_coord=agent_map_coord,
            agent_rotation=agent_map_angle,
            agent_radius_px=min(topdown_map.shape[0:2]) // 32,
        )
        return topdown_map

    def draw_egomotion_trajectory(self, topdown_map, ego_map_coords):
        maps.draw_path(topdown_map, ego_map_coords, (0, 0, 255), 4)
        return topdown_map

    def get_global_state_and_map_coords(
        self, global_state, local_deltas, topdown_map_shape, sim
    ):
        global_state = compute_global_state(global_state, local_deltas)
        global_rot, global_pos = global_state
        map_coord = maps.to_grid(
            global_pos[2], global_pos[0], topdown_map_shape[0:2], sim.sim
        )
        return map_coord, global_state
