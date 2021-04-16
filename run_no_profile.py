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
from pyinstrument import Profiler

from pathlib import Path
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

def run_no_profile(agent_config, env_config, num_episodes, directory=None):
    # Set up agent and env
    env = load_environment(env_config)
    agent = load_agent(agent_config, env)    
    
    evaluation = Evaluation(
        env,
        agent,
        num_episodes=num_episodes,
        display_env=False,
        display_agent=False,
        display_rewards=False,
        directory=directory
    )
    # Train 
    evaluation.train()


def run_rl_agents_no_profiling(config, base_dir):
    agent_configs = config["agent_configs"]
    env_configs = config["env_configs"]
    for agent_config, env_config in zip(agent_configs, env_configs):
        agent_path = Path(agent_config)
        env_path = Path(env_config)
        save_dir = base_dir / env_path.parts[-2] / env_path.stem / agent_path.parts[-2] / agent_path.stem
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Training for agent {agent_config}")

        run_no_profile(
            agent_config, env_config, config["num_episodes"], directory=save_dir
        )
        
    info_path = base_dir / "info.txt"
    with info_path.open("w") as f:
        f.write(str(os.uname()))
    configs_path = base_dir / "configs.json"
    with configs_path.open("w") as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    from rl_agents.trainer import logger
    logger.configure("configs/verbose.json")
    agent_configs = [ 
        "configs/HighwayEnv/agents/DQNAgent/dueling_ddqn.json",
    ]
    env_configs = [
        "configs/HighwayEnv/env.json",
    ]
    base_dir = Path("results", "highway_line_profiling")
    base_dir.mkdir(parents=True, exist_ok=True)
    num_episodes = 20
    config = {
        "agent_configs": agent_configs,
        "env_configs": env_configs,
        "num_episodes": num_episodes
    }
    run_rl_agents_no_profiling(config, base_dir)
