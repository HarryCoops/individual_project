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
import glob
import math
import os
from itertools import cycle
from sys import path

import numpy as np
import yaml, inspect
from scipy.spatial import distance

from smarts.core.scenario import Scenario
from smarts.env.hiway_env import HiWayEnv
from marl_scalability.baselines.adapter import BaselineAdapter
from marl_scalability.baselines.image_adapter import ImageBaselineAdapter
from marl_scalability.baselines.common.yaml_loader import load_yaml

path.append("./marl_scalability")
from marl_scalability.utils.common import ego_social_safety


class ScalabilityEnv(HiWayEnv):
    def __init__(
        self,
        agent_specs,
        scenarios,
        headless,
        timestep_sec,
        seed,
        eval_mode=False,
    ):
        self.timestep_sec = timestep_sec
        self.headless = headless
        self.marl_scalability_scores = BaselineAdapter.reward_adapter
        self.image_scores_adapter = ImageBaselineAdapter.reward_adapter
        
        super().__init__(
            scenarios=scenarios,
            agent_specs=agent_specs,
            headless=headless,
            timestep_sec=timestep_sec,
            seed=seed,
            visdom=False,
        )

    def generate_logs(self, observation, highwayenv_score):
        ego_state = observation.ego_vehicle_state
        start = observation.ego_vehicle_state.mission.start
        waypoints = getattr(observation, "waypoint_paths", None)
        if waypoints is not None:
            closest_wp = [min(wps, key=lambda wp: wp.dist_to(ego_state.position)) for wps in observation.waypoint_paths]
            closest_wp = min(closest_wp, key=lambda wp: wp.dist_to(ego_state.position))

            signed_dist_from_center = closest_wp.signed_lateral_error(ego_state.position)
            lane_width = closest_wp.lane_width * 0.5
            ego_dist_center = signed_dist_from_center / lane_width
            angle_error = closest_wp.relative_heading(ego_state.heading)
        

        linear_jerk = np.linalg.norm(ego_state.linear_jerk)
        angular_jerk = np.linalg.norm(ego_state.angular_jerk)

        ego_2d_position = ego_state.position[0:2]

        # This is kind of not efficient because the reward adapter is called again
        info = dict(
            position=ego_state.position,
            speed=ego_state.speed,
            steering=ego_state.steering,
            heading=ego_state.heading,
            dist_center=abs(ego_dist_center) if waypoints is not None else None,
            start=start,
            closest_wp=closest_wp if waypoints is not None else None,
            events=observation.events,
            #ego_num_violations=ego_num_violations,
            #social_num_violations=social_num_violations,
            linear_jerk=np.linalg.norm(ego_state.linear_jerk),
            angular_jerk=np.linalg.norm(ego_state.angular_jerk),
            env_score=self.marl_scalability_scores(observation, highwayenv_score) if waypoints is not None else self.image_scores_adapter(observation, highwayenv_score),
        )

        return info

    def step(self, agent_actions):
        agent_actions = {
            agent_id: self._agent_specs[agent_id].action_adapter(action)
            for agent_id, action in agent_actions.items()
        }

        observations, rewards, agent_dones, extras = self._smarts.step(agent_actions)

        infos = {
            agent_id: {"score": value, "env_obs": observations[agent_id]}
            for agent_id, value in extras["scores"].items()
        }

        for agent_id in observations:
            agent_spec = self._agent_specs[agent_id]
            observation = observations[agent_id]
            reward = rewards[agent_id]
            info = infos[agent_id]
            rewards[agent_id] = agent_spec.reward_adapter(observation, reward)
            observations[agent_id] = agent_spec.observation_adapter(observation)
            infos[agent_id] = agent_spec.info_adapter(observation, reward, info)
            infos[agent_id]["logs"] = self.generate_logs(observation, reward)

        for done in agent_dones.values():
            self._dones_registered += 1 if done else 0

        agent_dones["__all__"] = self._dones_registered == len(self._agent_specs)

        return observations, rewards, agent_dones, infos

    @property
    def info(self):
        return {
            "scenario_info": self.scenario_info,
            "timestep_sec": self.timestep_sec,
            "headless": self.headless,
        }
