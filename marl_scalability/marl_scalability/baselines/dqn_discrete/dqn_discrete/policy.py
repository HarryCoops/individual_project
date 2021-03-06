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
import numpy as np
from marl_scalability.baselines.dqn.dqn.network import *
from smarts.core.agent import Agent
from marl_scalability.utils.common import merge_discrete_action_spaces, to_3d_action, to_2d_action
import pathlib, os, copy
from marl_scalability.baselines.dqn_discrete.dqn_discrete.network import DQNCNN, DQNWithSocialEncoder
from marl_scalability.baselines.dqn.dqn.explore import EpsilonExplore
from marl_scalability.baselines.common.image_replay_buffer import ImageReplayBuffer
from marl_scalability.baselines.common.replay_buffer import ReplayBuffer
from marl_scalability.baselines.common.social_vehicle_config import get_social_vehicle_configs
from marl_scalability.baselines.common.yaml_loader import load_yaml
from marl_scalability.baselines.common.image_state_preprocessor import ImageStatePreprocessor
from marl_scalability.baselines.common.baseline_state_preprocessor import BaselineStatePreprocessor


class DiscreteDQNPolicy(Agent):
    lane_actions = ["keep_lane", "slow_down", "change_lane_left", "change_lane_right"]

    def __init__(
        self,
        policy_params=None,
        checkpoint_dir=None,
        marb=None,
        agent_id="",
        compression=None,
    ):
        self.agent_id = agent_id
        self.policy_params = policy_params
        self.agent_type = policy_params["agent_type"]
        if self.agent_type == "image":
            network_class = DQNCNN
        elif self.agent_type == "social":
            network_class = DQNWithSocialEncoder
        self.epsilon_obj = EpsilonExplore(1.0, 0.05, 10000)
        action_space_type = policy_params["action_space_type"]
        discrete_action_spaces = [[0],[1],[2],[3]]
        action_size = discrete_action_spaces
        self.merge_action_spaces = -1
        self.marb = marb
        self.step_count = 0
        self.update_count = 0
        self.num_updates = 0
        self.current_sticky = 0
        self.current_iteration = 0

        lr = float(policy_params["lr"])
        seed = int(policy_params["seed"])
        self.train_step = int(policy_params["train_step"])
        self.target_update = float(policy_params["target_update"])
        self.device_name = "cuda:2" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_name)
        self.warmup = int(policy_params["warmup"])
        self.gamma = float(policy_params["gamma"])
        self.batch_size = int(policy_params["batch_size"])
        self.use_ddqn = policy_params["use_ddqn"]
        self.sticky_actions = int(policy_params["sticky_actions"])
        prev_action_size = int(policy_params["prev_action_size"])
        self.prev_action = np.zeros(prev_action_size)
        self.action_size = prev_action_size
        

        index_to_actions = [
            e.tolist() if not isinstance(e, list) else e for e in action_size
        ]
        action_to_indexs = {
            str(k): v
            for k, v in zip(
                index_to_actions, np.arange(len(index_to_actions)).astype(np.int)
            )
        }
        self.index2actions, self.action2indexs = (
            [index_to_actions],
            [action_to_indexs],
        )
        self.num_actions = [len(index_to_actions)]
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
            self.state_description = BaselineStatePreprocessor.get_state_description(
                policy_params["social_vehicles"],
                policy_params["observation_num_lookahead"],
                prev_action_size,
            )
            self.social_vehicle_encoder = self.social_vehicle_config["encoder"]
            
            self.social_feature_encoder_class = self.social_vehicle_encoder[
                "social_feature_encoder_class"
            ]
            self.social_feature_encoder_params = self.social_vehicle_encoder[
                "social_feature_encoder_params"
            ]
            network_params = {
                "state_size": self.state_size,
                "social_feature_encoder_class": self.social_feature_encoder_class,
                "social_feature_encoder_params": self.social_feature_encoder_params,
            }
        elif self.agent_type == "image":
            self.n_in_channels = int(policy_params["n_in_channels"])
            self.image_height = int(policy_params["image_height"])
            self.image_width = int(policy_params["image_width"])
            self.state_description = ImageStatePreprocessor.get_state_description(
                (self.image_height, self.image_width),
            )
            network_params = {
                "state_size": self.state_size,
                "n_in_channels": self.n_in_channels,
                "image_dim": (self.image_width, self.image_height)
            }

        self.checkpoint_dir = checkpoint_dir
        self.reset()

        torch.manual_seed(seed)
        self.online_q_network = network_class(
            num_actions=self.num_actions,
            **(network_params if network_params else {}),
        ).to(self.device)
        self.target_q_network = network_class(
            num_actions=self.num_actions,
            **(network_params if network_params else {}),
        ).to(self.device)
        self.update_target_network()

        self.optimizers = torch.optim.Adam(
            params=self.online_q_network.parameters(), lr=lr
        )
        self.loss_func = nn.MSELoss(reduction="none")

        if self.checkpoint_dir:
            self.load(self.checkpoint_dir)

        self.action_space_type = action_space_type
        self.to_real_action = to_3d_action
        if self.agent_type == "social":
            self.replay = ReplayBuffer(
                buffer_size=int(policy_params["replay_buffer"]["buffer_size"]),
                batch_size=int(policy_params["replay_buffer"]["batch_size"]),
                device_name=self.device_name,
            )
        elif self.marb is None:
            self.replay = ImageReplayBuffer(
                buffer_size=int(policy_params["replay_buffer"]["buffer_size"]),
                batch_size=int(policy_params["replay_buffer"]["batch_size"]),
                device_name=self.device_name,
                compression=compression,
                dimensions=(self.n_in_channels, self.image_height, self.image_width)
            )
        else:
            self.marb.add_agent(self.agent_id)

    def lane_action_to_index(self, state):
        state = state.copy()
        if (
            len(state["action"]) == 3
            and (state["action"] == np.asarray([0, 0, 0])).all()
        ):  # initial action
            state["action"] = np.asarray([0])
        else:
            state["action"] = self.lane_actions.index(state["action"])
        return state

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
        size += self.action_size
        return size

    def reset(self):
        self.eps_throttles = []
        self.eps_steers = []
        self.eps_step = 0
        self.current_sticky = 0

    def soft_update(self, target, src, tau):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - tau) + param * tau)

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.online_q_network.state_dict().copy())

    def act(self, *args, **kwargs):
        if self.current_sticky == 0:
            self.action = self._act(*args, **kwargs)
        self.current_sticky = (self.current_sticky + 1) % self.sticky_actions
        self.current_iteration += 1
        if self.action_space_type == "continuous":
            return self.to_real_action(self.action)
        else: 
            return self.lane_actions[self.action[0]]

    def _act(self, state, explore=True):
        epsilon = self.epsilon_obj.get_epsilon()
        if not explore or np.random.rand() > epsilon:
            state = copy.deepcopy(state)
            state["low_dim_states"] = np.float32(
                np.append(state["low_dim_states"], self.prev_action)
            )
            if self.agent_type == "image":

                state["top_down_rgb"] = (
                    torch.Tensor(state["top_down_rgb"]).unsqueeze(0).to(self.device)
                )
                # Normalise to 0..1 expected by network
                state["top_down_rgb"].div_(255)
            elif self.agent_type == "social":
                state["social_vehicles"] = (
                    torch.from_numpy(state["social_vehicles"]).unsqueeze(0).to(self.device)
                )
            
            state["low_dim_states"] = (
                torch.from_numpy(state["low_dim_states"]).unsqueeze(0).to(self.device)
            )
            
            self.online_q_network.eval()
            with torch.no_grad():
                qs = self.online_q_network(state)
            qs = [q.data.cpu().numpy().flatten() for q in qs]
            # out_str = " || ".join(
            #     [
            #         " ".join(
            #             [
            #                 "{}: {:.4f}".format(index2action[j], q[j])
            #                 for j in range(num_action)
            #             ]
            #         )
            #         for index2action, q, num_action in zip(
            #             self.index2actions, qs, self.num_actions
            #         )
            #     ]
            # )
            # print(out_str)
            inds = [np.argmax(q) for q in qs]
        else:
            inds = [np.random.randint(num_action) for num_action in self.num_actions]
        action = []
        for j, ind in enumerate(inds):
            action.extend(self.index2actions[j][ind])
        self.epsilon_obj.step()
        self.eps_step += 1
        action = np.asarray(action)
        return action

    def save(self, model_dir):
        model_dir = pathlib.Path(model_dir)
        torch.save(self.online_q_network.state_dict(), model_dir / "online.pth")
        torch.save(self.target_q_network.state_dict(), model_dir / "target.pth")

    def load(self, model_dir, cpu=False):
        model_dir = pathlib.Path(model_dir)
        print("loading from :", model_dir)

        map_location = None
        if cpu:
            map_location = torch.device("cpu")
        self.online_q_network.load_state_dict(
            torch.load(model_dir / "online.pth", map_location=map_location)
        )
        self.target_q_network.load_state_dict(
            torch.load(model_dir / "target.pth", map_location=map_location)
        )
        print("Model loaded")

    def step(self, state, action, reward, next_state, done, info, others=None):
        # dont treat timeout as done equal to True
        max_steps_reached = info["logs"]["events"].reached_max_episode_steps
        if max_steps_reached:
            done = False
        if self.action_space_type == "continuous":
            action = to_2d_action(action)
            _action = (
                [[e] for e in action]
                if not self.merge_action_spaces
                else [action.tolist()]
            )
            action_index = np.asarray(
                [
                    action2index[str(e)]
                    for action2index, e in zip(self.action2indexs, _action)
                ]
            )
        else:
            action_index = self.lane_actions.index(action)
            action = action_index
        if self.marb is not None:
            self.marb.add(
                agent_id=self.agent_id,
                state=state,
                action=action_index,
                reward=reward,
                next_state=next_state,
                done=done,
                others=others,
                prev_action=self.prev_action
            )
        self.step_count += 1
        out = {}
        if self.step_count > max(self.batch_size, self.warmup):
            out = self.learn()
            self.update_count += 1
            if (self.step_count % self.train_step == self.train_step - 1 
                and self.marb is not None):
                self.marb.request_sample(self.agent_id)
            if self.step_count % self.train_step == 0:
                out = self.learn()
                self.update_count += 1

        if self.target_update > 1 and self.step_count % self.target_update == 0:
            self.update_target_network()
        elif self.target_update < 1.0:
            self.soft_update(
                self.target_q_network, self.online_q_network, self.target_update
            )
        if self.marb is None:
            self.replay.add(
                state=state,
                action=action_index,
                reward=reward,
                next_state=next_state,
                done=done,
                others=others,
                prev_action=self,prev_action
            )
        self.prev_action = action
        return out

    def learn(self):
        if self.marb is None:
            states, actions, rewards, next_states, dones, others = self.replay.sample(
                device=self.device
            )
        else:
            states, actions, rewards, next_states, dones, others = self.marb.collect_sample(
                self.agent_id
            )
        if not self.merge_action_spaces:
            actions = torch.chunk(actions, len(self.num_actions), -1)
        else:
            actions = [actions]

        self.target_q_network.eval()
        with torch.no_grad():
            qs_next_target = self.target_q_network(next_states)

        if self.use_ddqn:
            self.online_q_network.eval()
            with torch.no_grad():
                qs_next_online = self.online_q_network(next_states)
            next_actions = [
                torch.argmax(q_next_online, dim=1, keepdim=True)
                for q_next_online in qs_next_online
            ]
        else:
            next_actions = [
                torch.argmax(q_next_target, dim=1, keepdim=True)
                for q_next_target in qs_next_target
            ]

        qs_next_target = [
            torch.gather(q_next_target, 1, next_action)
            for q_next_target, next_action in zip(qs_next_target, next_actions)
        ]

        self.online_q_network.train()
        aux_losses = {}
        if self.agent_type == "social":
            qs, aux_losses = self.online_q_network(states, training=True)
        else:
            qs = self.online_q_network(states)
        qs = [torch.gather(q, 1, action.long()) for q, action in zip(qs, actions)]
        qs_target_value = [
            rewards + self.gamma * (1 - dones) * q_next_target
            for q_next_target in qs_next_target
        ]
        td_loss = [
            self.loss_func(q, q_target_value).mean()
            for q, q_target_value in zip(qs, qs_target_value)
        ]
        mean_td_loss = sum(td_loss) / len(td_loss)

        loss = mean_td_loss + sum(
            [e["value"] * e["weight"] for e in aux_losses.values()]
        )

        self.optimizers.zero_grad()
        loss.backward()
        self.optimizers.step()

        out = {}
        out.update(
            {
                "loss/td{}".format(j): {
                    "type": "scalar",
                    "data": td_loss[j].data.cpu().numpy(),
                    "freq": 10,
                }
                for j in range(len(td_loss))
            }
        )
        out.update(
            {
                "loss/{}".format(k): {
                    "type": "scalar",
                    "data": v["value"],  # .detach().cpu().numpy(),
                    "freq": 10,
                }
                for k, v in aux_losses.items()
            }
        )
        out.update({"loss/all": {"type": "scalar", "data": loss, "freq": 10}})

        self.num_updates += 1
        return out
