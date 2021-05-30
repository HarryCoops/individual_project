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
        return iter(torch.randperm(n).tolist()[0 : self.batch_size])


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
        transition = copy.deepcopy(transition)
        state, action, reward, next_state, done, others = transition
        if self.compression == "zlib":
            state["top_down_rgb"] = np.frombuffer(zlib.decompress(state["top_down_rgb"]), np.uint8)
            state["top_down_rgb"] = state["top_down_rgb"].reshape(self.dimensions)
            next_state["top_down_rgb"] = np.frombuffer(zlib.decompress(next_state["top_down_rgb"]), np.uint8)
            next_state["top_down_rgb"] = next_state["top_down_rgb"].reshape(self.dimensions)
        elif self.compression == "lz4":
            state["top_down_rgb"] = np.frombuffer(lz4.frame.decompress(state["top_down_rgb"]), np.uint8)
            state["top_down_rgb"] = state["top_down_rgb"].reshape(self.dimensions)
            next_state["top_down_rgb"] = np.frombuffer(lz4.frame.decompress(next_state["top_down_rgb"]), np.uint8)
            next_state["top_down_rgb"] = next_state["top_down_rgb"].reshape(self.dimensions)
    
        state["low_dim_states"] = torch.from_numpy(state["low_dim_states"]).to(self.device)
        state["top_down_rgb"] = torch.from_numpy(state["top_down_rgb"]).to(self.device)
        next_state["top_down_rgb"] = torch.from_numpy(next_state["top_down_rgb"]).to(self.device)
        next_state["low_dimstates"] = torch.from_numpy(next_state["low_dim_states"]).to(self.device)
        
        action = torch.from_numpy(action).to(self.device)
        done = torch.from_numpy(done).to(self.device)
        reward = torch.from_numpy(reward).float().to(self.device)
        return state, action, reward, next_state, done, others


class ImageReplayBuffer:
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
        self.replay_buffer_dataset = ReplayBufferDataset(
            buffer_size, 
            device=None,
            compression=compression,
            dimensions=dimensions,
        )
        self.sampler = RandomRLSampler(self.replay_buffer_dataset, batch_size)
        self.data_loader = DataLoader(
            self.replay_buffer_dataset,
            sampler=self.sampler,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        self.storage_device = torch.device(device_name)

    def add(self, *args, **kwargs):
        self.replay_buffer_dataset.add(*args, **kwargs)

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

    def sample(self, device=None):
        device = device if device else self.storage_device
        batch = list(iter(self.data_loader))
        states, actions, rewards, next_states, dones, others = zip(*batch)
        states = self.make_state_from_dict(states, device)
        next_states = self.make_state_from_dict(next_states, device)
        actions = torch.cat(actions, dim=0).float().to(device)
        rewards = torch.cat(rewards, dim=0).float().to(device)
        dones = torch.cat(dones, dim=0).float().to(device)
        return states, actions, rewards, next_states, dones, others
