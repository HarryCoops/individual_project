# Environment
import gym
import highway_env

# Agent
from rl_agents.trainer.evaluation import Evaluation
from rl_agents.agents.common.factory import agent_factory
from rl_agents.agents.common.factory import load_agent, load_environment

# Visualisation
from tqdm import trange

# Profiling 
import io
import os
import json
import cProfile
import pstats

from pathlib import PurePath

# Torch setup
import torch
torch.cuda.set_device(2)
"""
Notes:
- What does "expanding terminal states" mean?
- What does the DeterministicPlannnerAgent really do?
- Informally, which configuration options affect the runtime the most?
- How can we then formally measure this?
- How do we train a model?

Profiling Tools
- pycallgraph
- cProfile
- line_profiler
- pympler
"""

def generate_cprofile(agent_config, env_config, num_episodes):
    # Set up agent and env
    env = load_environment(env_config)
    agent = load_agent(agent_config)
    evaluation = Evaluation(
            env,
            agent,
            num_episodes=num_episodes,
            display_env=False,
            display_agent=False,
            display_rewards=False
    )
        
    # Set up profiling
    pr = cProfile.Profile()
    pr.enable()

    # Train
    evaluation.train()

    # Save profiling result
    result = io.StringIO()
    ps = pstats.Stats(pr,stream=result)
    ps.sort_stats("cumulative")
    ps.print_stats()
    return result.getvalue(), agent.config, env.config


env_config = "configs/HighwayEnv/env.json"
agent_configs = [(PurePath(p), d, f) for p, d, f in os.walk("configs/HighwayEnv/agents")]
base_dir = PurePath("results", "highwayenv_0")
base_dir.mkdir(parents=True)

for path, dirs, files in agent_configs:
    save_dir = base_dir / path.parts[-1]
    for f_p in files:
        agent_config = path / f_p
        results, agent_config_dict, env_config_dict = generate_cprofile(agent_config, env_config, 2)

        # chop the string into a csv-like buffer
        result='ncalls'+result.split('ncalls')[-1]
        result='\n'.join([','.join(line.rstrip().split(None,5)) for line in result.split('\n')])
        # save it to disk
        
        profile_path = save_dir / "cProfile.csv"
        agent_config_path = save_dir / "agent_config.json"
        env_config_path = save_dir / "env_config.json"
        with profile_path.open("w") as f:
            f.write(result)
        with agent_config_path.open("w") as f:
            f.write(json.dumps(agent_config_dict))
        with env_config_path.open("w") as f:
            f.write(json.dumps(env_config_dict))