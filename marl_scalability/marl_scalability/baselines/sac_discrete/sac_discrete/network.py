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

# Modifications for discrete sac inspired by https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC_Discrete.py
class SACNetwork(nn.Module):
    def __init__(
        self,
        action_size,
        state_size,
        seed=None,
        hidden_units=64,
        initial_alpha=0.02,
        social_feature_encoder_class=None,
        social_feature_encoder_params=None,
    ):
        super(SACNetwork, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)
        self.social_feature_encoder_class = social_feature_encoder_class
        self.social_feature_encoder_params = social_feature_encoder_params

        self.critic = DoubleCritic(
            action_size=action_size,
            state_size=state_size,
            seed=seed,
            hidden_units=hidden_units,
            social_feature_encoder=self.social_feature_encoder_class(
                **self.social_feature_encoder_params
            )
            if self.social_feature_encoder_class
            else None,
        )

        self.target = DoubleCritic(
            action_size=action_size,
            state_size=state_size,
            seed=seed,
            hidden_units=hidden_units,
            social_feature_encoder=self.social_feature_encoder_class(
                **self.social_feature_encoder_params
            )
            if self.social_feature_encoder_class
            else None,
        )

        self.actor = Actor(
            action_size=action_size,
            state_size=state_size,
            seed=seed,
            hidden_units=hidden_units,
            social_feature_encoder=self.social_feature_encoder_class(
                **self.social_feature_encoder_params
            )
            if self.social_feature_encoder_class
            else None,
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
        action_size,
        state_size,
        seed=None,
        hidden_units=64,
        social_feature_encoder=None,
    ):
        super(DoubleCritic, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)
        self.social_feature_encoder = social_feature_encoder
        if self.social_feature_encoder is not None:
            self.init_social_encoder_weight()

        self.q1 = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

    # Initialize weights of PointNet
    def init_social_encoder_weight(self):
        def init_weights(m):
            if hasattr(m, "weight"):
                torch.nn.init.normal_(m.weight, 0.0, 0.01)

        self.social_feature_encoder.apply(init_weights)

    def forward(self, state, action, training=False):

        low_dim_state = state["low_dim_states"]
        social_vehicles_state = state["social_vehicles"]
        aux_losses = {}
        social_feature = []
        if self.social_feature_encoder is not None:
            social_feature, social_encoder_aux_losses = self.social_feature_encoder(
                social_vehicles_state, training
            )
            aux_losses.update(social_encoder_aux_losses)
        else:
            social_feature = [e.reshape(1, -1) for e in social_vehicles_state]

        social_feature = torch.cat(social_feature, 0) if len(social_feature) > 0 else []
        # print(">>>>", social_feature.shape, low_dim_state.shape)
        state = (
            torch.cat([low_dim_state, social_feature], -1)
            if len(social_feature) > 0
            else low_dim_state
        )

        action_state = torch.cat((action, state), 1)
        q1 = self.q1(action_state)
        q2 = self.q1(action_state)

        if training:
            return q1, q2, aux_losses
        else:
            return q1, q2


class Actor(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        hidden_units,
        seed=None,
        social_feature_encoder=None,
    ):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)

        self.social_feature_encoder = social_feature_encoder

        self.common = nn.Sequential(
            nn.Linear(state_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
        )
        self.action_probs = nn.Sequential(
            nn.Linear(hidden_units, action_size),
            nn.Softmax(dim=1)
        )

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20
        self.EPSILON = 1e-6

    def forward(self, state, training=False):
        # get the low dimensional states from the obs dict
        low_dim_state = state["low_dim_states"]
        social_vehicles_state = state["social_vehicles"]
        aux_losses = {}
        social_feature = []
        if self.social_feature_encoder is not None:
            social_feature, social_encoder_aux_losses = self.social_feature_encoder(
                social_vehicles_state, training
            )
            aux_losses.update(social_encoder_aux_losses)
        else:
            social_feature = [e.reshape(1, -1) for e in social_vehicles_state]

        social_feature = torch.cat(social_feature, 0) if len(social_feature) > 0 else []
        # print(">>>>", social_feature.shape, low_dim_state.shape)
        state = (
            torch.cat([low_dim_state, social_feature], -1)
            if len(social_feature) > 0
            else low_dim_state
        )
        # print(state.shape)
        # feed the state into the networks to get action probabilities
        common_state = self.common(state)
        action_probs = self.action_probs(common_state)

        action_distribution = Categorical(action_probs)
        action = action_distribution.sample().cpu()
        max_prob_action = torch.argmax(action_probs)

        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)

        if training:
            return action, log_action_probabilities, aux_losses
        else:
            return action, log_action_probabilities, max_prob_action
