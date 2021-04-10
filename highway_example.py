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
import cProfile
import pstats


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

# Make agent and env
env_config = "configs/HighwayEnv/env.json"
agent_config = "configs/HighwayEnv/agents/DQNAgent/ddqn.json"
env = load_environment(env_config)
agent = load_agent(agent_config, env)
evaluation = Evaluation(env, agent, num_episodes=10, display_env=False, display_agent=False, display_rewards=False)

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
result=result.getvalue()
# chop the string into a csv-like buffer
result='ncalls'+result.split('ncalls')[-1]
result='\n'.join([','.join(line.rstrip().split(None,5)) for line in result.split('\n')])
# save it to disk
         
with open('test.csv', 'w+') as f:
    f.write(result)
"""
with open("profilingStatsAsText.txt", "w") as f:
    ps = pstats.Stats(pr, stream=f)
    ps.sort_stats('cumulative')
    ps.print_stats()
"""


# Test
env = load_environment(env_config)
env.configure({"offscreen_rendering": True})
agent = load_agent(agent_config, env)
evaluation = Evaluation(env, agent, num_episodes=3, recover=True)
evaluation.test()
env.close()
