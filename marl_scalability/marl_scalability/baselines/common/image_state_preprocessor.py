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
import collections.abc
import numpy as np
import torch

from marl_scalability.baselines.common.state_preprocessor import StatePreprocessor
from marl_scalability.utils.common import rotate2d_vector


class ImageStatePreprocessor(StatePreprocessor):
    """The State Preprocessor used by the baseline agents."""

    _NORMALIZATION_VALUES = {
        "speed": 30.0,
        "steering": 3.14,  # radians
        "heading": 3.14,  # radians
    }

    def __init__(
        self,
        image_dimensions,
    ):
        self.image_dimensions = image_dimensions
        self._state_description = self.get_state_description(
            image_dimensions
        )

    @staticmethod
    def get_state_description(
        image_dimensions
    ):
        return {
            "low_dim_states": {
                "speed": 1,
                "steering": 1,
                "heading": 1,
            },
            "top_down_rgb": image_dimensions
        }

    @property
    def num_low_dim_states(self):
        return sum(self._state_description["low_dim_states"].values())

    def _preprocess_state(
        self,
        state,
    ):
        low_dim_state = self._adapt_observation_for_baseline(state)
        # Normalize states and concatenate.
        normalized = [
            self._normalize(key, low_dim_state[key])
            for key in self._state_description["low_dim_states"]
        ]
        low_dim_states = [
            value
            if isinstance(value, collections.abc.Iterable)
            else np.asarray([value]).astype(np.float32)
            for value in normalized
        ]
        low_dim_states = torch.cat(
            [torch.from_numpy(e).float() for e in low_dim_states], dim=-1
        )

        rgb_data = state.top_down_rgb.data.astype(np.float32)
        if self.image_dimensions[0] == 1:
             img_data = np.dot(
                 rgb_data[...,:3], [0.299, 0.587, 0.114]
            )
             img_data = np.expand_dims(img_data, axis=0)
        else:
            H, W, C = rgb_data.shape
            img_data = rgb_data.reshape(C, H, W)
        # Get image state 
        out = {
            "low_dim_states": low_dim_states.numpy(),
            "top_down_rgb": img_data,
        }
        return out

    def _adapt_observation_for_baseline(self, state):
        # Get basic information about the ego vehicle.
        ego_position = self.extract_ego_position(state)
        ego_heading = self.extract_ego_heading(state)
        ego_speed = self.extract_ego_speed(state)
        ego_steering = self.extract_ego_steering(state)
        ego_start = self.extract_ego_start(state)
        ego_goal = self.extract_ego_goal(state)


        basic_state = dict(
            speed=ego_speed,
            steering=ego_steering,
            start=ego_start.position,
            heading=ego_heading,
            ego_position=ego_position,
            events=state.events,
        )
        return basic_state

    def _normalize(self, key, value):
        if key not in self._NORMALIZATION_VALUES:
            return value
        return value / self._NORMALIZATION_VALUES[key]
