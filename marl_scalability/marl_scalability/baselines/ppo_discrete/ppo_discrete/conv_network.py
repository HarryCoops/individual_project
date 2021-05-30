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
from torch.distributions import Categorical


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class ActorNetwork(nn.Module):
    def __init__(
        self,
        n_in_channels, 
        image_dim,
        state_size, 
        action_size, 
        hidden_dim=128,
        activation = nn.ReLU
    ):
        super(ActorNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(n_in_channels, 32, 8, 4),
            activation(),
            nn.Conv2d(32, 64, 4, 4),
            activation(),
            nn.Conv2d(64, 64, 3, 2),
            activation(),
            nn.Conv2d(64, 64, 3, 2),
            activation(),
            Flatten(),
        )

        dummy = torch.zeros((1, n_in_channels, *image_dim))
        im_feature_size = self.model(dummy).data.cpu().numpy().size

        self.outs = nn.Sequential(
            nn.Linear(im_feature_size + state_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size),
            nn.Softmax(dim=1)
        )


    def forward(self, state, training=False):
        low_dim_state = state["low_dim_states"]
        image = state["top_down_rgb"] / 255
        im_feature = self.model(image)
        x = torch.cat([im_feature, low_dim_state], dim=-1)
        x = self.outs(x)
        return x, {}


class CriticNetwork(nn.Module):
    def __init__(
        self,
        n_in_channels, 
        image_dim,
        state_size, 
        hidden_dim=128,
        activation = nn.ReLU
    ):
        super(CriticNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(n_in_channels, 32, 8, 4),
            activation(),
            nn.Conv2d(32, 64, 4, 4),
            activation(),
            nn.Conv2d(64, 64, 3, 2),
            activation(),
            nn.Conv2d(64, 64, 3, 2),
            activation(),
            Flatten(),
        )

        dummy = torch.zeros((1, n_in_channels, *image_dim))
        im_feature_size = self.model(dummy).data.cpu().numpy().size

        self.out = nn.Sequential(
            nn.Linear(im_feature_size + state_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )


    def forward(self, state, training=False):
        low_dim_state = state["low_dim_states"]
        image = state["top_down_rgb"] / 255
        im_feature = self.model(image)
        x = torch.cat([im_feature, low_dim_state], dim=-1)
        x = self.out(x)
        return x, {}


class ImagePPONetwork(nn.Module):
    def __init__(
        self,
        n_in_channels,
        image_dim,
        action_size,
        state_size,
        seed=None,
        hidden_units=64,
        init_std=0.5,
        social_feature_encoder_class=None,
        social_feature_encoder_params=None,
    ):
        super(ImagePPONetwork, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)

        self.critic = CriticNetwork(
            n_in_channels,
            image_dim,
            state_size,
            hidden_units,
        )

        self.actor = ActorNetwork(
            n_in_channels,
            image_dim,
            state_size,
            action_size,
            hidden_units
        )

        # self.init_last_layer(self.actor)
        self.log_std = nn.Parameter(torch.log(init_std * torch.ones(1, action_size)))

    def forward(self, x, training=False):
        value, critic_aux_loss = self.critic(x, training=training)
        action_probs, actor_aux_loss = self.actor(x, training=training)
        dist = Categorical(action_probs)
        if training:
            aux_losses = {}
            for k, v in actor_aux_loss.items():
                aux_losses.update({"actor/{}".format(k): v})
            for k, v in critic_aux_loss.items():
                aux_losses.update({"critic/{}".format(k): v})
            return (dist, value), aux_losses
        else:
            return dist, value

    def init_last_layer(self, actor):
        """Initialize steering to zero and throttle to maximum"""
        # nn.init.constant_(actor.model[-2].weight.data[0], 1.0)
        # nn.init.constant_(actor.model[-2].weight.data[1], 0.0)
        nn.init.constant_(actor.model[-2].bias.data[0], 2.0)
