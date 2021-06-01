#!/usr/bin/env python

import gym
import gym.wrappers

env_names = [
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot"
]

for name in env_names:
    env = gym.make(f"procgen:procgen-{name}-v0", render_mode="rgb_array")
    env.metadata["render.modes"] = ["human", "rgb_array"]
    env = gym.wrappers.Monitor(env=env, directory=f"./videos/{name}", force=True)

    episodes = 1
    _ = env.reset()

    done = False
    while episodes > 0:
        _, _, done, _ = env.step(env.action_space.sample())
        if done:
            _ = env.reset()
            episodes -= 1