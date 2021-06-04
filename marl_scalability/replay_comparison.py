import torch
import timeit
import numpy as np
from pyinstrument import Profiler
from marl_scalability.baselines.common.multi_agent_image_replay_buffer import MARLImageReplayBuffer 
from marl_scalability.baselines.common.image_replay_buffer import ImageReplayBuffer 

torch.cuda.init()
torch.rand(1, device="cuda:2")

def mutli_agent_sampling_test(n_agents):
    agents = [f"a_{i}" for i in range(n_agents)]
    replay_buffer = MARLImageReplayBuffer(
        buffer_size = 500,
        batch_size = 32,
        device_name="cuda:2",
        dimensions=(3, 256, 256)
    )
    for _ in range(500):
        for agent in agents:
            replay_buffer.add(
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
        replay_buffer.request_sample(agent)
    replay_buffer.generate_samples()
    for agent in agents:
        transition = replay_buffer.collect_sample(agent)
    return transition

def single_agent_sampling_test(n_agents):
    agents = {
        f"a_{i}": ImageReplayBuffer(
            buffer_size = 500,
            batch_size = 32,
            device_name="cuda:2",
            dimensions=(3, 256, 256)
        )  
        for i in range(n_agents)
    }
    for _ in range(500):
        for agent in agents:
            agents[agent].add(
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
        transition = agents[agent].sample()
    return transition

    

times = {}
_times = []
for f in ("multi" ,"single"):
    pr = Profiler()
    pr.start()
    for n in range(5, 75, 10):
        mutli_agent_sampling_test(n)
        t = timeit.timeit(
                f"{f}_agent_sampling_test({n})",
                setup=f"from __main__ import {f}_agent_sampling_test",
                number=5
            )
        _times.append(t)
    times[f] = _times
    pr.stop()
    print(pr.output_text(unicode=True, color=True))

print(times)