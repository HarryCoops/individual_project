# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import numpy as np
import torch
from scipy.spatial import distance
import random, math, gym
from sys import path
from collections import OrderedDict
from marl_scalability.baselines.common.image_state_preprocessor import ImageStatePreprocessor
from marl_scalability.baselines.common.yaml_loader import load_yaml

path.append("./marl_scalability")
from marl_scalability.utils.common import (
    rotate2d_vector,
)

seed = 0
random.seed(seed)


class ImageBaselineAdapter:
    def __init__(self, agent_name, agent_config_path=None):
        self.policy_params = load_yaml(
            (agent_config_path if agent_config_path is not None else
            f"marl_scalability/baselines/{agent_name}/{agent_name}/image_params.yaml")
        )
        self.num_image_channels = self.policy_params["n_in_channels"]
        self.image_height = self.policy_params["image_height"]
        self.image_width = self.policy_params["image_width"]
        self.image_dimensions = image_dimensions = (
            self.num_image_channels,
            self.image_height,
            self.image_width
        )
        self.state_preprocessor = ImageStatePreprocessor(
            self.image_dimensions
        )

        self.state_size = self.state_preprocessor.num_low_dim_states

    @property
    def observation_space(self):
        low_dim_states_shape = self.state_preprocessor.num_low_dim_states
        return gym.spaces.Dict(
            {
                "low_dim_states": gym.spaces.Box(
                    low=-1e10,
                    high=1e10,
                    shape=(low_dim_states_shape,),
                    dtype=torch.Tensor,
                ),
                "top_down_rgb": gym.spaces.Box(
                    low=-1e10,
                    high=1e10,
                    shape=self.image_dimensions,
                    dtype=torch.Tensor,
                ),
            }
        )

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=np.array([0.0, 0.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

    def observation_adapter(self, env_observation):
        return self.state_preprocessor(
            state=env_observation,
        )

    @staticmethod
    def reward_adapter(observation, reward):
        env_reward = reward
        ego_events = observation.events
        ego_observation = observation.ego_vehicle_state
        start = observation.ego_vehicle_state.mission.start

        linear_jerk = np.linalg.norm(ego_observation.linear_jerk)
        angular_jerk = np.linalg.norm(ego_observation.angular_jerk)

        # Distance to goal
        ego_2d_position = ego_observation.position[0:2]
        #goal_dist = distance.euclidean(ego_2d_position, goal.position)
        ego_position = ego_observation.position

        speed_fraction = max(0, ego_observation.speed / 14)
        ego_step_reward = 0.02 * min(speed_fraction, 1)
        ego_speed_reward = min(
            0, (14 - ego_observation.speed) * 0.01
        )  # m/s
        ego_collision = len(ego_events.collisions) > 0
        ego_collision_reward = -1.0 if ego_collision else 0.0
        ego_off_road_reward = -1.0 if ego_events.off_road else 0.0
        ego_off_route_reward = -1.0 if ego_events.off_route else 0.0
        ego_wrong_way = -0.02 if ego_events.wrong_way else 0.0
        ego_goal_reward = 0.0
        ego_time_out = 0.0
        #ego_dist_center_reward = -0.002 * min(1, abs(ego_dist_center))
        #ego_angle_error_reward = -0.005 * max(0, np.cos(angle_error))
        ego_reached_goal = 1.0 if ego_events.reached_goal else 0.0
        #ego_safety_reward = -0.02 if ego_num_violations > 0 else 0
        #social_safety_reward = -0.02 if social_num_violations > 0 else 0
        ego_lat_speed = 0.0  # -0.1 * abs(long_lat_speed[1])
        #ego_linear_jerk = -0.0001 * linear_jerk
        #ego_angular_jerk = -0.0001 * angular_jerk * math.cos(angle_error)
        env_reward /= 100
        # DG: Different speed reward
        #ego_speed_reward = -0.1 if speed_fraction >= 1 else 0.0
        #ego_speed_reward += -0.01 if speed_fraction < 0.01 else 0.0
        
        rewards = sum(
            [
                ego_goal_reward,
                ego_collision_reward,
                ego_off_road_reward,
                ego_off_route_reward,
                ego_wrong_way,
                ego_speed_reward,
                # ego_time_out,
               # ego_dist_center_reward,
                #ego_angle_error_reward,
                ego_reached_goal,
                ego_step_reward,
                env_reward,
                # ego_linear_jerk,
                # ego_angular_jerk,
                # ego_lat_speed,
                # ego_safety_reward,
                # social_safety_reward,
            ]
        )
        return rewards
