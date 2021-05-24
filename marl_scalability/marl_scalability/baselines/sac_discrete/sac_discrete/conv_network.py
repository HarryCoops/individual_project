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
import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


# Modifications for discrete sac inspired by https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC_Discrete.py
class ImageSACNetwork(nn.Module):
    def __init__(
        self,
        n_in_channels,
        image_dim,
        action_size,
        discrete_action_choices,
        state_size,
        seed=None,
        hidden_units=64,
        initial_alpha=0.02,
        activation=nn.ReLU(),
    ):
        super(ImageSACNetwork, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.critic = DoubleCritic(
            n_in_channels=n_in_channels,
            image_dim=image_dim,
            action_size=action_size,
            state_size=state_size,
            seed=seed,
            hidden_units=hidden_units,
        )

        self.target = DoubleCritic(
            n_in_channels=n_in_channels,
            image_dim=image_dim,
            action_size=action_size,
            state_size=state_size,
            seed=seed,
            hidden_units=hidden_units,
        )

        self.actor = Actor(
            n_in_channels=n_in_channels,
            image_dim=image_dim,
            action_size=discrete_action_choices,
            state_size=state_size,
            seed=seed,
            hidden_units=hidden_units,
        )

        # self.init_last_layer(self.actor)
        self.alpha = nn.Parameter(torch.FloatTensor([0.0]))
        self.log_alpha = nn.Parameter(torch.FloatTensor([-3.0]))

    def init_last_layer(self, actor):
        """Initialize steering to zero and throttle to maximum"""
        pass

    def sample(self, state, training=False):
        return self.actor(state, training=training)


class DoubleCritic(nn.Module):
    """This class is a double critic that can produce q1 and q2 when is fed with a state
    It is used to form the double critic networks and the corresponding double targets.
    """

    def __init__(
        self,
        n_in_channels,
        image_dim,
        action_size,
        state_size,
        seed=None,
        hidden_units=64,
        activation=nn.ReLU,
    ):
        super(DoubleCritic, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)

        self.im_feature_1 = nn.Sequential(
            nn.Conv2d(n_in_channels, 16, 8, 4),
            activation(),
            nn.Conv2d(16, 32, 4, 4),
            activation(),
            nn.Conv2d(32, 64, 3, 2),
            activation(),
            nn.Conv2d(64, 64, 3, 2),
            activation(),
            Flatten(),
        )

        self.im_feature_2 = nn.Sequential(
            nn.Conv2d(n_in_channels, 16, 8, 4),
            activation(),
            nn.Conv2d(16, 32, 4, 4),
            activation(),
            nn.Conv2d(32, 64, 3, 2),
            activation(),
            nn.Conv2d(64, 64, 3, 2),
            activation(),
            Flatten(),
        )

        dummy = torch.zeros((1, n_in_channels, *image_dim))
        im_feature_size = self.im_feature_1(dummy).data.cpu().numpy().size

        self.q1 = nn.Sequential(
            nn.Linear(im_feature_size + state_size + action_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(im_feature_size + state_size + action_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, state, action, training=False):
        low_dim_state = state["low_dim_states"]
        top_down_rgb = state["top_down_rgb"]
        im_q1 = self.im_feature_1(top_down_rgb)
        im_q2 = self.im_feature_2(top_down_rgb)
        action_state_1 = torch.cat((im_q1, action.float(), low_dim_state), 1)
        action_state_2 = torch.cat((im_q2, action.float(), low_dim_state), 1)
        q1 = self.q1(action_state_1)
        q2 = self.q2(action_state_2)

        if training:
            return q1, q2, {}
        else:
            return q1, q2


class Actor(nn.Module):
    def __init__(
        self,
        n_in_channels,
        image_dim,
        state_size,
        action_size,
        hidden_units,
        seed=None,
        activation=nn.ReLU
    ):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)

        self.common = nn.Sequential(
            nn.Conv2d(n_in_channels, 32, 8, 4),
            activation(),
            nn.Conv2d(32, 64, 4, 4),
            activation(),
            nn.Conv2d(64, 64, 4, 4),
            activation(),
            Flatten()
        )

        dummy = torch.zeros((1, n_in_channels, *image_dim))
        im_feature_size = self.common(dummy).data.cpu().numpy().size

        self.action_probs = nn.Sequential(
            nn.Linear(im_feature_size + state_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, action_size),
            nn.Softmax(dim=1)
        )

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20
        self.EPSILON = 1e-6

    def forward(self, state, training=False):
        # get the low dimensional states from the obs dict
        low_dim_state = state["low_dim_states"]
        top_down_rgb = state["top_down_rgb"] / 255
        common_state = self.common(top_down_rgb)
        x = torch.cat([common_state, low_dim_state], dim=-1)
        action_probs = self.action_probs(x)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        max_prob_action = torch.argmax(action_probs)

        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)

        if training:
            return action, log_action_probabilities, {}
        else:
            return action, log_action_probabilities, max_prob_action
