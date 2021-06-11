import cProfile 
import json
import os
import csv
import psutil
from pathlib import Path
import argparse
import pickle
import pandas as pd 
import dill
import gym
import psutil
import torch
from smarts.zoo.registry import make
from marl_scalability.utils.episode import episodes
from datetime import datetime
from memory_profiler import profile 
from pyinstrument import Profiler
import pstats
import io
from marl_scalability.baselines.common.multi_agent_image_replay_buffer import MARLImageReplayBuffer

def no_op_profile(stream):
    def _no_op_profile(func):
        return func
    return _no_op_profile

def outer_train(f, *args, **kwargs):
    @profile(stream=f)
    def train(
        scenario,
        n_agents,
        num_episodes,
        max_episode_steps,
        headless,
        policy_class,
        seed,
        log_dir,
        experiment_name,
        record_vehicle_lifespan,
        record_mem_usage,
        max_steps,
        maintain_agent_numbers,
        use_marb
    ):
        torch.set_num_threads(1)
        total_step = 0
        finished = False
        # Make agent_ids in the form of 000, 001, ..., 010, 011, ..., 999, 1000, ...;
        agent_ids = ["0" * max(0, 3 - len(str(i))) + str(i) for i in range(n_agents)]

        # Assign the agent classes
        agent_classes = {
            agent_id: policy_class
            for agent_id in agent_ids
        }
        marb = MARLImageReplayBuffer(
            buffer_size=int(1e6), 
            batch_size=32,
            device_name="cuda:2",
            num_workers=0,
            compression="lz4",
            dimensions=(3, 256, 256)
        ) if use_marb else None

        # Create the agent specifications matched with their associated ID.
        agent_specs = {
            agent_id: make(
                locator=policy_class, 
                max_episode_steps=max_episode_steps,
                replay_buffer=marb,
                agent_id=agent_id
            )
            for agent_id, policy_class in agent_classes.items()
        }
        print("Building agents...")
        # Create the agents matched with their associated ID.
        agents = {
            agent_id: agent_spec.build_agent()
            for agent_id, agent_spec in agent_specs.items()
        }
        print("Making env...")
        # Create the environment.
        env = gym.make(
            "marl_scalability.env:scalability-v0",
            agent_specs=agent_specs,
            scenarios=[scenario,],
            headless=headless,
            timestep_sec=0.1,
            seed=seed,
        )
        # Define an 'etag' for this experiment's data directory based off policy_classes.
        # E.g. From a ["marl_scalability.baselines.dqn:dqn-v0", "marl_scalability.baselines.ppo:ppo-v0"]
        # policy_classes list, transform it to an etag of "dqn-v0:ppo-v0".
        #etag = ":".join([policy_class.split(":")[-1] for policy_class in policy_classes])
        surviving_vehicles_total = []
        mem_usage = []
        mem_usage_interval = 100
        print("Starting training...")
        for episode in episodes(num_episodes, experiment_name=experiment_name, log_dir=log_dir, write_table=True, max_steps=max_steps):
            # Reset the environment and retrieve the initial observations.
            surviving_vehicles = []
            observations = env.reset()
            dones = {"__all__": False}
            infos = None
            episode.reset()
            experiment_dir = episode.experiment_dir
            # Save relevant agent metadata.
            if not os.path.exists(f"{experiment_dir}/agent_metadata.pkl"):
                if not os.path.exists(experiment_dir):
                    os.makedirs(experiment_dir)
                with open(f"{experiment_dir}/agent_metadata.pkl", "wb") as metadata_file:
                    dill.dump(
                        {
                            "agent_ids": agent_ids,
                            "agent_classes": agent_classes,
                            "agent_specs": agent_specs,
                        },
                        metadata_file,
                        pickle.HIGHEST_PROTOCOL,
                    )

            while not dones["__all__"]:
                # Request and perform actions on each agent that received an observation.
                actions = {
                    agent_id: agents[agent_id].act(observation, explore=True)
                    for agent_id, observation in observations.items()
                }
                if use_marb:
                    marb.generate_samples()
                next_observations, rewards, dones, infos = env.step(actions)
                # Active agents are those that receive observations in this step and the next
                # step. Step each active agent (obtaining their network loss if applicable).
                active_agent_ids = observations.keys() & next_observations.keys()
                surviving_vehicles.append(len(active_agent_ids))
                loss_outputs = {
                    agent_id: agents[agent_id].step(
                        state=observations[agent_id],
                        action=actions[agent_id],
                        reward=rewards[agent_id],
                        next_state=next_observations[agent_id],
                        done=dones[agent_id],
                        info=infos[agent_id],
                    )
                    for agent_id in active_agent_ids
                }
                if use_marb:
                    marb.reset()
                # Record the data from this episode.
                episode.record_step(
                    agent_ids_to_record=active_agent_ids,
                    infos=infos,
                    rewards=rewards,
                    total_step=total_step,
                    loss_outputs=loss_outputs,
                )
                # Update variables for the next step.
                total_step += 1
                observations = next_observations
                if maintain_agent_numbers and len(active_agent_ids) / n_agents < 0.6:
                    dones["__all__"] = True
                if record_mem_usage and total_step % mem_usage_interval == 0:
                    process = psutil.Process(os.getpid())
                    mem_usage.append(
                        (total_step, process.memory_info().rss)
                    )

            # Normalize the data and record this episode on tensorboard.
            episode.record_episode()
            episode.record_tensorboard()
            surviving_vehicles += [0,] * (max_episode_steps - len(surviving_vehicles))
            surviving_vehicles_total.append(surviving_vehicles)
            if finished:
                break
        if record_vehicle_lifespan:
            with open(Path(log_dir) / experiment_name / "surviving_vehicle_data.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerows(surviving_vehicles_total)
        if record_mem_usage:
            steps, mem_usage = zip(*mem_usage)
            mem_usage = pd.DataFrame(
                {
                    "mem_usage": pd.Series(mem_usage), 
                    "step": pd.Series(steps),
                }
            )
            mem_usage.to_csv(Path(log_dir) / experiment_name / "mem_usage.csv")
        env.close()
    train(*args, **kwargs)
    if f is not None:
        f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("intersection-training")
    parser.add_argument(
        "--scenario", help="Scenario to run", type=str, default="scenarios/big_circle"
    )
    parser.add_argument(
        "--n-agents",
        help="Number of ego agents to train",
        type=int,
        default=10
    )
    parser.add_argument(
        "--policy",
        help="Policies available : [ppo, sac, td3, dqn, bdqn]",
        type=str,
        default="sac",
    )
    parser.add_argument(
        "--episodes", help="Number of training episodes", type=int, default=1000000
    )
    parser.add_argument(
        "--max-episode-steps",
        help="Maximum number of steps per episode",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--headless", help="Run without envision", action="store_true", default=False
    )
    parser.add_argument(
        "--memprof", help="Run experiment with a memory profiler", 
        action="store_true", 
        default=False
    )
    parser.add_argument(
        "--profiler", 
        help="Run experiment with a specified exeuction profiler",
        type=str, 
        default=""
    )
    parser.add_argument(
        "--record-vehicle-lifespan", 
        help="Record the number of vehicles surviving at each timestep",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--seed",
        help="Environment seed",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--log-dir",
        help="Log directory location",
        default="logs",
        type=str,
    )
    parser.add_argument(
        "--max-steps",
        help="Maximum number of steps to run",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--full-memory-profile",
        help="Run a full memory profile line by line",
        default=False,
        action="store_true"
    )
    
    parser.add_argument(
        "--maintain-agent-numbers",
        help="Stop episode when less than 60% of agents survive",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--use-marb",
        help="Use Multi Agent Replay Buffer",
        default=False,
        action="store_true"
    )

    base_dir = os.path.dirname(__file__)
    pool_path = os.path.join(base_dir, "agent_pool.json")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)

    string_date = datetime.now().strftime("%Y:%m:%d:%H:%M:%S")
    experiment_name = f"exp_{args.policy}_{args.n_agents}_{args.episodes}_{string_date}"
    experiment_dir = log_dir / experiment_name
    experiment_dir.mkdir(exist_ok=True)
    # Obtain the policy class strings for each specified policy.
    policy_class = "sac"
    with open(pool_path, "r") as f:
        data = json.load(f)
        if args.policy in data["agents"].keys():
            policy_class = (data["agents"][args.policy]["path"] +
             ":" + data["agents"][args.policy]["locator"])
        else:
            raise ImportError("Invalid policy name. Please try again")

    
    train_args = {
        "scenario": args.scenario,
        "n_agents": args.n_agents,
        "num_episodes": int(args.episodes),
        "max_episode_steps": int(args.max_episode_steps),
        "headless": args.headless,
        "policy_class": policy_class,
        "seed": args.seed,
        "log_dir": args.log_dir,
        "experiment_name": experiment_name,
        "record_vehicle_lifespan": args.record_vehicle_lifespan,
        "record mem_usage": args.memprof,
        "max_steps": args.max_steps,
        "maintain_agent_numbers": args.maintain_agent_numbers,
        "use_marb": args.use_marb,
    }

    if args.profiler == "pyinstrument":
        pr = Profiler(interval=0.1)
        pr.start()
    elif args.profiler == "cProfile":
        pr = cProfile.Profile()
        pr.enable()
    if args.full_memory_profile:
        f = open(log_dir / experiment_name / "mem_profile.txt", "w")
    else:
        f = None 
        profile = no_op_profile
    outer_train(f, *train_args.values())
    
    if args.profiler == "pyinstrument":
        pr.stop()
        with open(log_dir / experiment_name / "profile.html", "w") as f:
            f.write(pr.output_html())
    elif args.profiler == "cProfile":
        result = io.StringIO()
        ps = pstats.Stats(pr, stream=result)
        ps.sort_stats("cumulative")
        ps.print_stats()
        result = result.getvalue()
        result = "ncalls" + result.split("ncalls")[-1]
        result = "\n".join([",".join(line.rstrip().split(None, 5)) for line in result.split("\n")])
        with open(log_dir / experiment_name / "cprofile.csv", "w") as f:
            f.write(result)

    with open(log_dir / experiment_name / "train_args.json", "w") as f:
        f.write(json.dumps(train_args))
