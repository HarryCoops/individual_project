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
# Do not make any change to this file when merging. Just use my version.
from collections import deque, namedtuple
import numpy as np
import random, copy
from numpy.core.fromnumeric import compress
import torch
from torch.utils import data
from marl_scalability.utils.common import normalize_im
from collections.abc import Iterable

from torch.utils.data import Dataset, Sampler, DataLoader
import zlib
from pympler import asizeof
import lz4.frame

Transition = namedtuple(
    "Transition",
    field_names=["state", "action", "reward", "next_state", "done", "others"],
    # others may contain importance sampling ratio, GVF rewards,... etc
    defaults=(None,) * 6,
)


class RandomRLSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.datasource = data_source
        self.batch_size = batch_size

    def __len__(self):
        return len(self.datasource)

    def __iter__(self):
        n = len(self.datasource)
        sample = torch.randperm(n).tolist()[0 : self.batch_size]
        return iter(sample)


class ReplayBufferDataset(Dataset):
    cpu = torch.device("cpu")

    def __init__(self, buffer_size, device, dimensions, compression=None):
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=self.buffer_size)
        self.device = device
        self.compression = compression
        self.dimensions = dimensions

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        prev_action,
        others=None,
    ):
        if others is None:
            others = {}
        # dereference the states
        state = copy.deepcopy(state)
        next_state = copy.deepcopy(state)
        state["low_dim_states"] = np.float32(
            np.append(state["low_dim_states"], prev_action)
        )
        next_state["low_dim_states"] = np.float32(
            np.append(next_state["low_dim_states"], action)
        )
        if self.compression == "zlib":
            state["top_down_rgb"] = zlib.compress(state["top_down_rgb"], 1)
            next_state["top_down_rgb"] = zlib.compress(next_state["top_down_rgb"], 1)
        elif self.compression == "lz4":
            state["top_down_rgb"] = lz4.frame.compress(state["top_down_rgb"])
            next_state["top_down_rgb"] = lz4.frame.compress(next_state["top_down_rgb"])
        action = np.asarray([action]) if not isinstance(action, Iterable) else action
        reward = np.asarray([reward])
        done = np.asarray([done])
        new_experience = Transition(state, action, reward, next_state, done, others)
        self.memory.append(new_experience)

    def __len__(self):
        return len(self.memory)

    def _get_raw(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        state, action, reward, next_state, done, others = tuple(self.memory[idx])
        return state, action, reward, next_state, done, others

    def __sizeof__(self): 
        return sum(asizeof.asizeof(transition) for transition in self.memory)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        transition = tuple(self.memory[idx])
        #transition = copy.deepcopy(transition)
        state, action, reward, next_state, done, others = transition
        _state = {}
        _next_state = {}
        if self.compression == "zlib":
            _state["top_down_rgb"] = np.frombuffer(zlib.decompress(state["top_down_rgb"]), np.uint8)
            _state["top_down_rgb"] = _state["top_down_rgb"].reshape(self.dimensions)
            _next_state["top_down_rgb"] = np.frombuffer(zlib.decompress(next_state["top_down_rgb"]), np.uint8)
            _next_state["top_down_rgb"] = _next_state["top_down_rgb"].reshape(self.dimensions)
        elif self.compression == "lz4":
            _state["top_down_rgb"] = np.frombuffer(lz4.frame.decompress(state["top_down_rgb"]), np.uint8)
            _state["top_down_rgb"] = _state["top_down_rgb"].reshape(self.dimensions)
            _next_state["top_down_rgb"] = np.frombuffer(lz4.frame.decompress(next_state["top_down_rgb"]), np.uint8)
            _next_state["top_down_rgb"] = _next_state["top_down_rgb"].reshape(self.dimensions)
    
        _state["low_dim_states"] = torch.from_numpy(state["low_dim_states"])
        _state["top_down_rgb"] = torch.from_numpy(_state["top_down_rgb"])
        _next_state["top_down_rgb"] = torch.from_numpy(_next_state["top_down_rgb"])
        _next_state["low_dim_states"] = torch.from_numpy(next_state["low_dim_states"])
        action = torch.from_numpy(action)
        done = torch.from_numpy(done)
        reward = torch.from_numpy(reward)
        return _state["low_dim_states"], _state["top_down_rgb"], action, reward, _next_state["low_dim_states"], _next_state["top_down_rgb"], done, others


class MARLImageReplayBuffer:
    def __init__(
        self,
        buffer_size,
        batch_size,
        device_name,
        pin_memory=False,
        num_workers=0,
        compression=None,
        dimensions=[]
    ):
        self.device = torch.device(device_name)
        self.agent_dataloaders = {}
        self.agent_datasets = {}
        self.needs_sample = []
        self.requested_sample = []
        self.compression = compression
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        self.dimensions = dimensions

    def add(self, agent_id, *args, **kwargs):
        self.agent_datasets[agent_id].add(*args, **kwargs)

    def __len__(self):
        return len(self.replay_buffer_dataset)

    def __getitem__(self, idx):
        return self.replay_buffer_dataset[idx]

    def __sizeof__(self):
        return (self.replay_buffer_dataset.__sizeof__() +
        self.sampler.__sizeof__() +
        self.data_loader.__sizeof__()
        )
    def _get_raw(self, idx):
        return self.replay_buffer_dataset._get_raw(idx)

    def make_state_from_dict(self, states, device):
        low_dim_states = (
            torch.cat(
                [e["low_dim_states"] for e in states], 
                dim=0).float().to(device)
        )
        images = (
            torch.cat(
                [e["top_down_rgb"] for e in states],
                dim=0).float().to(device)
        )
        out = {
            "top_down_rgb": images,
            "low_dim_states": low_dim_states,
        }
        return out
    
    def add_agent(self, agent_id):
        replay_buffer_dataset = ReplayBufferDataset(
            self.buffer_size, 
            device=None,
            compression=self.compression,
            dimensions=self.dimensions,
        )
        sampler = RandomRLSampler(replay_buffer_dataset, self.batch_size)
        data_loader = DataLoader(
            replay_buffer_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
        self.agent_dataloaders[agent_id] = data_loader
        self.agent_datasets[agent_id] = replay_buffer_dataset
    
    def len(self, agent_id):
        return len(self.agent_datasets[agent_id])

    def generate_samples(self):
        n_agents = len(self.requested_sample)
        if n_agents == 0:
            return 
        image_data = torch.empty(
            (n_agents, self.batch_size, *self.dimensions), 
            pin_memory=True,
            dtype=torch.uint8,
        )
        low_dim_data = torch.empty(
            (n_agents, self.batch_size, 4), pin_memory=True
        )
        all_actions = torch.empty(
            (n_agents, self.batch_size, 1), pin_memory=True
        )
        all_rewards = torch.empty(
            (n_agents, self.batch_size, 1), pin_memory=True
        )
        next_image_data = torch.empty(
            (n_agents, self.batch_size, *self.dimensions), 
            pin_memory=True,
            dtype=torch.uint8
        )
        next_low_dim_data = torch.empty(
            (n_agents, self.batch_size, 4), pin_memory=True
        )
        all_dones = torch.empty(
            (n_agents, self.batch_size, 1), pin_memory=True
        )
        for i, agent_id in enumerate(self.requested_sample):
            agent_batch = next(iter(self.agent_dataloaders[agent_id]))
            state_lds, state_tdrgb, actions, rewards, next_lds, next_tdrgb, dones, others = agent_batch

            image_data[i] = state_tdrgb
            low_dim_data[i] = state_lds
            next_image_data[i] = next_tdrgb
            next_low_dim_data[i] = next_lds
            all_rewards[i] = rewards
            all_dones[i] = dones
            all_actions[i] = actions
        self.image_data = image_data.to(
                self.device, non_blocking=True).float()
        self.next_image_data = next_image_data.to(
                self.device, non_blocking=True).float()
        self.actions = all_actions.to(
                self.device, non_blocking=True).float()
        self.low_dim_data = low_dim_data.to(
                self.device, non_blocking=True).float()
        self.next_low_dim_data = next_low_dim_data.to(
                self.device, non_blocking=True).float()
        self.rewards = all_rewards.to(
                self.device, non_blocking=True).float()
        self.dones = all_dones.to(
                self.device, non_blocking=True).float()
        self.needs_sample = self.requested_sample
        self.requested_sample = []


    def request_sample(self, agent_id):
        if agent_id not in self.requested_sample:
            self.requested_sample.append(agent_id)
    
    def reset(self):
        self.image_data = None
        self.next_image_data = None
        self.actions = None
        self.low_dim_data = None
        self.next_low_dim_data = None
        self.rewards = None
        self.dones = None
        self.requested_sample += [
            i for i in self.needs_sample if i is not None
        ]

    def collect_sample(self, agent_id, device=None):
        idx = self.needs_sample.index(agent_id)
        self.needs_sample[idx] = None
        state = {
            "top_down_rgb": self.image_data[idx],
            "low_dim_states": self.low_dim_data[idx]
        }
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        next_states = {
            "top_down_rgb": self.next_image_data[idx],
            "low_dim_states": self.next_low_dim_data[idx]
        }
        dones = self.dones[idx]
        return state, actions, rewards, next_states, dones, None

if __name__ == "__main__":
    marl_buffer = MARLImageReplayBuffer(
        buffer_size = 500,
        batch_size = 32,
        device_name="cpu",
        dimensions=(3, 256, 256)
    )
    agents = [f"a_{i}" for i in range(5)]
    for agent in agents:
        marl_buffer.add_agent(agent)
    for _ in range(500):
        for agent in agents:
            marl_buffer.add(
                agent, 
                { 
                    "top_down_rgb": np.random.randint(0, 255, (3, 256, 256)),
                    "low_dim_states": np.random.rand(10)
                },
                np.random.rand(1), 
                np.random.rand(1),
                { 
                    "top_down_rgb": np.random.randint(0, 255, (3, 256, 256)),
                    "low_dim_states": np.random.rand(10)
                },
                False,
                np.random.rand(2)
            )
    for agent in agents:
        marl_buffer.request_sample(agent)
    marl_buffer.generate_samples()
    state, action, reward, next_state, done, other = marl_buffer.collect_sample(agents[0])
    print("state", state["top_down_rgb"].shape, state["low_dim_states"].shape)
    print("action", action.shape)
    print("reward", reward.shape)
    print("done", done.shape)
