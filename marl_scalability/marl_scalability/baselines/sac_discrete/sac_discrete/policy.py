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
# some parts of this implementation is inspired by https://github.com/openai/spinningup
import torch
import numpy as np
import torch.nn as nn
from sys import path
from marl_scalability.baselines.sac_discrete.sac_discrete.network import SACNetwork
from marl_scalability.baselines.sac_discrete.sac_discrete.conv_network import ImageSACNetwork
import torch.nn.functional as F
import pathlib, os, yaml, copy
from marl_scalability.utils.common import compute_sum_aux_losses, to_3d_action, to_2d_action
from smarts.core.agent import Agent
from marl_scalability.baselines.common.replay_buffer import ReplayBuffer
from marl_scalability.baselines.common.image_replay_buffer import ImageReplayBuffer
from marl_scalability.baselines.common.social_vehicle_config import get_social_vehicle_configs
from marl_scalability.baselines.common.yaml_loader import load_yaml
from marl_scalability.baselines.common.baseline_state_preprocessor import BaselineStatePreprocessor
from marl_scalability.baselines.common.image_state_preprocessor import ImageStatePreprocessor

torch.autograd.set_detect_anomaly(True)


class DiscreteSACPolicy(Agent):

    def __init__(
        self,
        policy_params=None,
        checkpoint_dir=None,
        marb=None,
        agent_id="",
        compression="",
    ):
        self.agent_id = agent_id
        self.marb = marb
        self.lane_actions = ["keep_lane", "slow_down", "change_lane_left", "change_lane_right"]
        self.policy_params = policy_params
        self.agent_type = policy_params["agent_type"]
        self.gamma = float(policy_params["gamma"])
        self.critic_lr = float(policy_params["critic_lr"])
        self.actor_lr = float(policy_params["actor_lr"])
        self.critic_update_rate = int(policy_params["critic_update_rate"])
        self.policy_update_rate = int(policy_params["policy_update_rate"])
        self.warmup = int(policy_params["warmup"])
        self.seed = int(policy_params["seed"])
        self.batch_size = int(policy_params["batch_size"])
        self.hidden_units = int(policy_params["hidden_units"])
        self.tau = float(policy_params["tau"])
        self.initial_alpha = float(policy_params["initial_alpha"])
        self.logging_freq = int(policy_params["logging_freq"])
        self.action_size = int(policy_params["action_size"])
        self.discrete_action_choices = int(policy_params["discrete_action_choices"])
        self.prev_action = np.zeros(self.action_size)

        if self.agent_type == "social":
            # state preprocessing
            self.social_policy_hidden_units = int(
                policy_params["social_vehicles"].get("social_policy_hidden_units", 0)
            )
            self.social_capacity = int(
                policy_params["social_vehicles"].get("social_capacity", 0)
            )
            self.observation_num_lookahead = int(
                policy_params.get("observation_num_lookahead", 0)
            )
            self.social_policy_init_std = int(
                policy_params["social_vehicles"].get("social_policy_init_std", 0)
            )
            self.num_social_features = int(
                policy_params["social_vehicles"].get("num_social_features", 0)
            )
            self.social_vehicle_config = get_social_vehicle_configs(
                **policy_params["social_vehicles"]
            )
            self.prev_action_size = 1

            self.social_vehicle_encoder = self.social_vehicle_config["encoder"]
            self.state_description = BaselineStatePreprocessor.get_state_description(
                policy_params["social_vehicles"],
                policy_params["observation_num_lookahead"],
                self.action_size,
            )
            self.social_feature_encoder_class = self.social_vehicle_encoder[
                "social_feature_encoder_class"
            ]
            self.social_feature_encoder_params = self.social_vehicle_encoder[
                "social_feature_encoder_params"
            ]
        else:
            self.n_in_channels = int(policy_params["n_in_channels"])
            self.image_height = int(policy_params["image_height"])
            self.image_width = int(policy_params["image_width"])
            self.state_description = ImageStatePreprocessor.get_state_description(
                (self.image_height, self.image_width),
            )

        # others
        self.checkpoint_dir = checkpoint_dir
        self.device_name = "cuda:2" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_name)
        self.save_codes = (
            policy_params["save_codes"] if "save_codes" in policy_params else None
        )
        if self.agent_type == "image":
            self.memory = ImageReplayBuffer(
                buffer_size=int(policy_params["replay_buffer"]["buffer_size"]),
                batch_size=int(policy_params["replay_buffer"]["batch_size"]),
                device_name=self.device_name,
                compression=compression,
                dimensions=(self.n_in_channels, self.image_height, self.image_width)
            )
        elif self.marb is None:
            self.memory = ReplayBuffer(
                buffer_size=int(policy_params["replay_buffer"]["buffer_size"]),
                batch_size=int(policy_params["replay_buffer"]["batch_size"]),
                device_name=self.device_name,
            )
        else:
            self.marb.add_agent(self.agent_id)

        self.current_iteration = 0
        self.steps = 0
        self.init_networks()
        if checkpoint_dir:
            self.load(checkpoint_dir)

    @property
    def state_size(self):
        # Adjusting state_size based on number of features (ego+social)
        size = sum(self.state_description["low_dim_states"].values())
        if self.agent_type == "image":
            return size + self.action_size
        if self.social_feature_encoder_class:
            size += self.social_feature_encoder_class(
                **self.social_feature_encoder_params
            ).output_dim
        else:
            size += self.social_capacity * self.num_social_features
        # adding the previous action
        size += self.prev_action_size
        return size

    def init_networks(self):
        if self.agent_type == "social":
            self.sac_net = SACNetwork(
                action_size=self.action_size,
                discrete_action_choices=self.discrete_action_choices,
                state_size=self.state_size,
                hidden_units=self.hidden_units,
                seed=self.seed,
                initial_alpha=self.initial_alpha,
                social_feature_encoder_class=self.social_feature_encoder_class,
                social_feature_encoder_params=self.social_feature_encoder_params,
            ).to(self.device_name)
        elif self.agent_type == "image":
            self.sac_net = ImageSACNetwork(
                n_in_channels=self.n_in_channels,
                image_dim=(self.image_width, self.image_height),
                action_size=self.action_size,
                discrete_action_choices=self.discrete_action_choices,
                state_size=self.state_size,
                hidden_units=self.hidden_units,
                seed=self.seed,
                initial_alpha=self.initial_alpha,
            ).to(self.device_name)
        
        self.actor_optimizer = torch.optim.Adam(
            self.sac_net.actor.parameters(), lr=self.actor_lr
        )

        self.critic_optimizer = torch.optim.Adam(
            self.sac_net.critic.parameters(), lr=self.critic_lr
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.sac_net.log_alpha], lr=self.critic_lr
        )

    def act(self, state, explore=True):
        state = copy.deepcopy(state)
        state["low_dim_states"] = np.float32(
            np.append(state["low_dim_states"], self.prev_action)
        )
        if self.agent_type == "social":
            state["social_vehicles"] = (
                torch.from_numpy(state["social_vehicles"]).unsqueeze(0).to(self.device)
            )
        elif self.agent_type == "image":
            state["top_down_rgb"] = (
                    torch.Tensor(state["top_down_rgb"]).unsqueeze(0).to(self.device)
                )

        state["low_dim_states"] = (
            torch.from_numpy(state["low_dim_states"]).unsqueeze(0).to(self.device)
        )

        action, _, mean = self.sac_net.sample(state)
        for name, param in self.sac_net.actor.named_parameters():
            if hasattr(param, "data"):
                pass
                #print(name, param.data.max())

        if explore:  # training mode
            action = torch.squeeze(action, 0)
            action = action.detach().cpu().numpy()
        else:  # testing mode
            mean = torch.squeeze(mean, 0)
            action = mean.detach().cpu().numpy()
        return self.lane_actions[action]

    def step(self, state, action, reward, next_state, done, info):
        # dont treat timeout as done equal to True
        action = self.lane_actions.index(action)
        max_steps_reached = info["logs"]["events"].reached_max_episode_steps
        if max_steps_reached:
            done = False
        if self.marb is None:
            self.memory.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=float(done),
                others=None,
                prev_action=self.prev_action,
            )
        else:
            self.marb.add(
                agent_id=self.agent_id,
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=float(done),
                others=None,    
                prev_action=self.prev_action
            )
        self.steps += 1
        output = {}
        if self.steps > max(self.warmup, self.batch_size):
            if self.marb is None:
                states, actions, rewards, next_states, dones, others = self.memory.sample(
                    device=self.device_name
                )
            else:
                states, actions, rewards, next_states, dones, others = self.marb.collect_sample(
                    self.agent_id
                )
                self.marb.request_sample(self.agent_id)
            if self.steps % self.critic_update_rate == 0:
                critic_loss = self.update_critic(
                    states, actions, rewards, next_states, dones
                )
                output["loss/critic_loss"] = {
                    "type": "scalar",
                    "data": critic_loss.item(),
                    "freq": 2,
                }

            if self.steps % self.policy_update_rate == 0:
                actor_loss, temp_loss = self.update_actor_temp(
                    states, actions, rewards, next_states, dones
                )
                output["loss/actor_loss"] = {
                    "type": "scalar",
                    "data": actor_loss.item(),
                    "freq": self.logging_freq,
                }
                output["loss/temp_loss"] = {
                    "type": "scalar",
                    "data": temp_loss.item(),
                    "freq": self.logging_freq,
                }
                output["others/alpha"] = {
                    "type": "scalar",
                    "data": self.sac_net.alpha.item(),
                    "freq": self.logging_freq,
                }
                self.current_iteration += 1
            self.target_soft_update(self.sac_net.critic, self.sac_net.target, self.tau)
        self.prev_action = action if not done else np.zeros(self.action_size)
        return output

    def update_critic(self, states, actions, rewards, next_states, dones):
        q1_current, q2_current, aux_losses = self.sac_net.critic(
            states, actions, training=True
        )
        with torch.no_grad():
            next_actions, log_probs, _ = self.sac_net.sample(next_states)
            q1_next, q2_next = self.sac_net.target(next_states, next_actions.unsqueeze(1))
            v_next = (
                torch.min(q1_next, q2_next) - self.sac_net.alpha.detach() * log_probs
            )
            v_next = v_next.sum(dim=1).unsqueeze(-1)
            q_target = (rewards + ((1 - dones) * self.gamma * v_next)).detach()

        critic_loss = F.mse_loss(q1_current, q_target) + F.mse_loss(
            q2_current, q_target
        )
        aux_losses = compute_sum_aux_losses(aux_losses)
        overall_loss = critic_loss + aux_losses
        self.critic_optimizer.zero_grad()
        overall_loss.backward()
        self.critic_optimizer.step()

        return critic_loss

    def update_actor_temp(self, states, actions, rewards, next_states, dones):

        for p in self.sac_net.target.parameters():
            p.requires_grad = False
        for p in self.sac_net.critic.parameters():
            p.requires_grad = False

        # update actor:
        actions, log_probs, aux_losses = self.sac_net.sample(states, training=True)
        if len(actions.shape) == 1:
            q1, q2 = self.sac_net.critic(states, actions.unsqueeze(1))
        else:
            q1, q2 = self.sac_net.critic(states, actions)
        
        q_old = torch.min(q1, q2)
        actor_loss = (self.sac_net.alpha.detach() * log_probs - q_old).mean()
        aux_losses = compute_sum_aux_losses(aux_losses)
        overall_loss = actor_loss + aux_losses
        
        nn.utils.clip_grad_norm_(self.sac_net.actor.parameters(), 0.5)
        self.actor_optimizer.zero_grad()
        overall_loss.backward()
        self.actor_optimizer.step()
        # update temp:
        temp_loss = (
            self.sac_net.log_alpha.exp()
            * (-log_probs.detach().mean() + self.action_size).detach()
        )
        self.log_alpha_optimizer.zero_grad()
        temp_loss.backward()
        nn.utils.clip_grad_norm_(self.sac_net.actor.parameters(), 0.5)
        self.log_alpha_optimizer.step()
        self.sac_net.alpha.data = self.sac_net.log_alpha.exp().detach()

        for p in self.sac_net.target.parameters():
            p.requires_grad = True
        for p in self.sac_net.critic.parameters():
            p.requires_grad = True

        return actor_loss, temp_loss

    def target_soft_update(self, critic, target_critic, tau):
        with torch.no_grad():
            for critic_param, target_critic_param in zip(
                critic.parameters(), target_critic.parameters()
            ):
                target_critic_param.data = (
                    tau * critic_param.data + (1 - tau) * target_critic_param.data
                )

    def load(self, model_dir):
        model_dir = pathlib.Path(model_dir)
        map_location = None
        if self.device and self.device.type == "cpu":
            map_location = "cpu"
        self.sac_net.actor.load_state_dict(
            torch.load(model_dir / "actor.pth", map_location=map_location)
        )
        self.sac_net.target.load_state_dict(
            torch.load(model_dir / "target.pth", map_location=map_location)
        )
        self.sac_net.critic.load_state_dict(
            torch.load(model_dir / "critic.pth", map_location=map_location)
        )
        print("<<<<<<< MODEL LOADED >>>>>>>>>", model_dir)

    def save(self, model_dir):
        model_dir = pathlib.Path(model_dir)
        # with open(model_dir / "params.yaml", "w") as file:
        #     yaml.dump(policy_params, file)

        torch.save(self.sac_net.actor.state_dict(), model_dir / "actor.pth")
        torch.save(self.sac_net.target.state_dict(), model_dir / "target.pth")
        torch.save(self.sac_net.critic.state_dict(), model_dir / "critic.pth")
        print("<<<<<<< MODEL SAVED >>>>>>>>>", model_dir)

    def reset(self):
        pass
