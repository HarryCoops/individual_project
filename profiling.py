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

def generate_pyins_profile(agent_config, env_config, num_episodes, directory=None):
    # Set up agent and env
    env = load_environment(env_config)
    agent = load_agent(agent_config, env)    
    
    # Profiling 
    pr = Profiler()
    pr.start()

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

    pr.stop()
    # Return profiling result
    return pr.output_html()


def generate_cprofile(agent_config, env_config, num_episodes, directory=None):
    # Set up agent and env
    env = load_environment(env_config)
    agent = load_agent(agent_config, env)    
    
    # Profiling 
    pr = cProfile.Profile()
    pr.enable()
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

    pr.disable()
    # Save profiling result
    result = io.StringIO()
    ps = pstats.Stats(pr,stream=result)
    ps.sort_stats("cumulative")
    ps.print_stats()
    return result.getvalue(), agent.config, env.config

def run_rl_agents_profiling(config, base_dir):
    agent_configs = config["agent_configs"]
    env_configs = config["env_configs"]
    for agent_config, env_config in zip(agent_configs, env_configs):
        agent_path = Path(agent_config)
        env_path = Path(env_config)
        save_dir = base_dir / env_path.parts[-2] / env_path.stem / agent_path.parts[-2] / agent_path.stem
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Training for agent {agent_config}")
        result, agent_config_dict, env_config_dict = generate_cprofile(
            agent_config, env_config, config["num_episodes"], directory=save_dir
        )

        # Chop the string into a csv-like buffer
        result = "ncalls" + result.split("ncalls")[-1]
        result = "\n".join([",".join(line.rstrip().split(None,5)) for line in result.split('\n')])

        pyins_result_html = generate_pyins_profile(
            agent_config, env_config, config["num_episodes"], directory=save_dir
        )
        
        # Save profile and config info to disk
        pyins_profile_path = save_dir / "pyins_profile.html"
        cprofile_path = save_dir / "cProfile.csv"
        agent_config_path = save_dir / "agent_config.json"
        env_config_path = save_dir / "env_config.json"
        with pyins_profile_path.open("w") as f:
            f.write(pyins_result_html)
        with cprofile_path.open("w") as f:
            f.write(result)
        with agent_config_path.open("w") as f:
            json.dump(agent_config_dict, f)
        with env_config_path.open("w") as f:
            json.dump(env_config_dict, f)

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
        "configs/HighwayEnv/controlled_vehicles_experiment/env_125_agent.json",
    ]
    base_dir = Path("results", "vary_controlled_vehicles")
    base_dir.mkdir(parents=True, exist_ok=True)
    num_episodes = 25
    config = {
        "agent_configs": agent_configs,
        "env_configs": env_configs,
        "num_episodes": num_episodes
    }
    run_rl_agents_profiling(config, base_dir)
