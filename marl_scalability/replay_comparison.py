import sys
import time
import os
import gc
import torch
import timeit
import psutil
import tracemalloc
import numpy as np
import pandas as pd
from pathlib import Path
from pyinstrument import Profiler
import torch.autograd.profiler as profiler
from marl_scalability.baselines.common.multi_agent_image_replay_buffer import MARLImageReplayBuffer 
from marl_scalability.baselines.common.image_replay_buffer import ImageReplayBuffer 

torch.cuda.init()
torch.rand(1, device="cuda:2")

N_SAMPLES = 20

def multi_agent_sampling_test(n_agents, tensors):
    agents = [f"a_{i}" for i in range(n_agents)]
    replay_buffer = MARLImageReplayBuffer(
        buffer_size = 500,
        batch_size = 32,
        device_name="cuda:2",
        dimensions=(3, 256, 256)
    )
    for agent in agents:
        replay_buffer.add_agent(agent)
    for i in range(500):
        for j, agent in enumerate(agents):
            replay_buffer.add(
                agent, 
                { 
                    "top_down_rgb": tensors[1000*j + i],
                    "low_dim_states": np.random.rand(10)
                },
                np.random.rand(1), 
                np.random.rand(1),
                { 
                    "top_down_rgb": tensors[1000*j + i + 500],
                    "low_dim_states": np.random.rand(10)
                },
                False,
                np.random.rand(2)
            )
    for _ in range(N_SAMPLES):
        for agent in agents:
            replay_buffer.request_sample(agent)
        replay_buffer.generate_samples()
        for agent in agents:
            replay_buffer.collect_sample(agent)
        replay_buffer.reset()
    del replay_buffer
    gc.collect()
    return

def single_agent_sampling_test(n_agents, tensors):
    agents = {
        f"a_{i}": ImageReplayBuffer(
            buffer_size = 500,
            batch_size = 32,
            device_name="cuda:2",
            dimensions=(3, 256, 256)
        )  
        for i in range(n_agents)
    }
    for i in range(500):
        for j, agent in enumerate(agents):
            agents[agent].add(
                { 
                    "top_down_rgb": tensors[1000*j + i],
                    "low_dim_states": np.random.rand(10)
                },
                np.random.rand(1), 
                np.random.rand(1),
                { 
                    "top_down_rgb": tensors[1000*j + i + 500],
                    "low_dim_states": np.random.rand(10)
                },
                False,
                np.random.rand(2)
            )
    for _ in range(N_SAMPLES):
        for agent in agents:
            agents[agent].sample()
    del agents 
    gc.collect()
    return 

def main(
    repeats=1, 
    trace_mallocs=False,
    max_agents=65,
    min_agents=5,
    increment=5,
    buffer_type="multi",
    output_file="results.csv"
):
    if trace_mallocs:
        tracemalloc.start(5)
    test_dir = Path("results", "MARB_testing")
    test_dir.mkdir(exist_ok=True, parents=True)
    torch.cuda.empty_cache()
    times = {}
    n_agents = list(range(min_agents, max_agents+1, increment))
    for f in (buffer_type,):
        pr = Profiler(interval=0.1)
        pr.start()
        prev_snapshot = None
        for _ in range(repeats):
            _times = []
            for n in n_agents:
                pr.stop()
                tensors = [
                        np.random.randint(
                            0, 255, (3, 256, 256), dtype=np.uint8
                        )
                        for _ in range(1000 * n)
                ]
                pr.start()
                t = time.time()
                if f == "multi":
                    multi_agent_sampling_test(n, tensors)
                else:
                    single_agent_sampling_test(n, tensors)
                elapsed = time.time() - t
                _times.append(elapsed)
                if trace_mallocs:
                    snapshot = tracemalloc.take_snapshot()
                    if prev_snapshot is not None:
                        top_stats = snapshot.compare_to(prev_snapshot, "lineno")
                        for stat in top_stats[:5]:
                            print(stat)
                    prev_snapshot = snapshot 
            print(_times)
            if f in times:
                times[f] += np.array(_times)
            else:
                times[f] = np.array(_times)
        pr.stop()
        print(pr.output_text(unicode=True, color=True))
        with open(test_dir / f"{output_file}.html", "w") as _f:
            _f.write(pr.output_html())
        times[f] /= repeats
    df = pd.DataFrame(times, index=n_agents)
    df.to_csv(test_dir / (output_file + ".csv"), index_label="n_agents")
    print(df)

if __name__ == "__main__":
    np.random.seed(1)
    torch.manual_seed(1)
    main(
        trace_mallocs=False,
        repeats=5, 
        min_agents=5,
        max_agents=85,
        increment=5,
        buffer_type="single",
        output_file="results_single_longer_5"
    )
