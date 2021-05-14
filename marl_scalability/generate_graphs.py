import csv
import os
import pandas as pd
import numpy as np
import argparse 
from pathlib import Path
from pprint import pprint

import matplotlib as mpl
import matplotlib.pyplot as plt

def extract_experiment_info(runs):
	info = {}
	for run in runs:
		run_info = {}
		run_info["path"] = run
		run_name = run.split("/")[1]
		run_name_parts = run_name.split("_")
		if len(run_name_parts) == 5:
			run_info["policy"] = run_name_parts[1]
			run_info["n_agents"] = int(run_name_parts[2])
			run_info["episodes"] = int(run_name_parts[3])
		else:
			run_info["policy"] = f"{run_name_parts[1]}_{run_name_parts[2]}"
			run_info["n_agents"] = int(run_name_parts[3])
			run_info["episodes"] = int(run_name_parts[4])
		if run_info["n_agents"] == 150:
			continue
		if run_info["policy"] in info:
			info[run_info["policy"]].append(run_info)
		else:
			info[run_info["policy"]] = [run_info,]
	for policy in info:
		info[policy].sort(key=lambda run_info: run_info["n_agents"])
	return info 

def extract_mem_usage(info):
	for policy in info:
		for run in info[policy]:
			mem_usage_csv_path = Path(run["path"]) / "mem_usage.csv"
			if mem_usage_csv_path.exists():
				mem_usage = pd.read_csv(mem_usage_csv_path)
				run["mem_usage"] = {}
				maximum_mem_usage = mem_usage["mem_usage"].max()
				run["mem_usage"]["max"] = maximum_mem_usage
				mean_mem_usage = mem_usage["mem_usage"].mean()
				run["mem_usage"]["mean"] = mean_mem_usage
				min_mem_usage = mem_usage["mem_usage"].min()
				run["mem_usage"]["min"] = min_mem_usage
				std_mem_usage = mem_usage["mem_usage"].std()
				run["mem_usage"]["std"] = std_mem_usage

def extract_stats_csv_info(info):
	for policy in info:
		for run in info[policy]:
			stats_csv_path = Path(run["path"]) / "stats.csv"
			if stats_csv_path.exists():
				stats = pd.read_csv(stats_csv_path, skipfooter=1)
				run["run_episode_stats"] = {}
				run["run_episode_stats"]["sim/wall"] = {}
				run["run_episode_stats"]["sim/wall"]["mean"] = stats["sim/wall"].mean()
				run["run_episode_stats"]["sim/wall"]["max"] = stats["sim/wall"].max()
				run["run_episode_stats"]["sim/wall"]["min"] = stats["sim/wall"].min()
				run["run_episode_stats"]["sim/wall"]["std"] = stats["sim/wall"].std()
				run["run_episode_stats"]["steps/sec"] = {}
				run["run_episode_stats"]["steps/sec"]["mean"] = stats["steps/sec"].mean()
				run["run_episode_stats"]["steps/sec"]["max"] = stats["steps/sec"].max()
				run["run_episode_stats"]["steps/sec"]["min"] = stats["steps/sec"].min()
				run["run_episode_stats"]["steps/sec"]["std"] = stats["steps/sec"].std()
				run["run_episode_stats"]["total_steps"] = {}
				run["run_episode_stats"]["total_steps"]["mean"] = stats["total steps"].mean()
				run["run_episode_stats"]["total_steps"]["max"] = stats["total steps"].max()
				run["run_episode_stats"]["total_steps"]["min"] = stats["total steps"].min()
				run["run_episode_stats"]["total_steps"]["std"] = stats["total steps"].std()


def extract_profile_info(info, add_norm=True):
	for policy in info:
		for run in info[policy]:
			profile_csv_path = Path(run["path"]) / "cprofile.csv"
			if profile_csv_path.exists():
				profile = pd.read_csv(profile_csv_path)
				total_time = profile.iloc[0]["cumtime"]
				func_col = "filename:lineno(function)"
				run["cprofile_stats"] = {}
				core_step_time = profile.loc[
					profile[func_col] == "/src/smarts/core/smarts.py:167(step)"
				]["cumtime"].iloc[0]
				run["cprofile_stats"]["core_step_time"] = core_step_time
				gen_logs_time = profile.loc[
					profile[func_col] == "/marl/marl_scalability/env/scalability_env.py:64(generate_logs)"
				]["cumtime"].iloc[0]
				run["cprofile_stats"]["gen_logs_time"] = gen_logs_time
				reward_time = profile.loc[
					profile[func_col] == "/marl/marl_scalability/baselines/adapter.py:153(reward_adapter)"
				]["cumtime"].iloc[0]
				run["cprofile_stats"]["reward_time"] = reward_time / 2
				obs_time = profile.loc[
					profile[func_col] == "/marl/marl_scalability/baselines/adapter.py:134(observation_adapter)"
				]["cumtime"].iloc[0]
				run["cprofile_stats"]["obs_time"] = obs_time
				policy_step_time = profile.loc[
					profile[func_col].str.match(r"/marl/marl_scalability/baselines.*policy\.py:.*(step)")
				]["cumtime"].iloc[0]
				run["cprofile_stats"]["policy_step_time"] = policy_step_time
				other_time = total_time - sum(run["cprofile_stats"].values())
				run["cprofile_stats"]["total_time"] = total_time
				run["cprofile_stats"]["other"] = other_time
				if add_norm:
					run["cprofile_stats_norm"] = {}
					run["cprofile_stats_norm"]["core_step_time"] = core_step_time / total_time
					run["cprofile_stats_norm"]["gen_logs_time"] = gen_logs_time / total_time
					run["cprofile_stats_norm"]["reward_time"] = (reward_time / 2) / total_time
					run["cprofile_stats_norm"]["obs_time"] = obs_time / total_time
					run["cprofile_stats_norm"]["policy_step_time"] = policy_step_time / total_time
					run["cprofile_stats_norm"]["other"] = other_time / total_time

def extract_vehicle_retention_info(info):
	for policy in info:
		for run in info[policy]:
			vehicle_retention_path = Path(run["path"]) / "surviving_vehicle_data.csv"
			if vehicle_retention_path.exists():
				with open(vehicle_retention_path) as f:
					reader = list(csv.reader(f, delimiter=","))
					if not reader:
						continue
					data = np.array(reader).astype(int)
					run["vehicle_retention"] = list(np.mean(data, axis=0))


def plot_mem_usage_graph(info, log_dir, errorbars=True):
	graphs_dir = log_dir / "graphs"
	graphs_dir.mkdir(exist_ok=True)
	fig, axs = plt.subplots(3)
	#ax.set_xlabel("Number of Ego Agents")
	#ax.set_ylabel("Memory Usage")
	colours = ["b", "g", "orange"]
	for i, policy in enumerate(info):
		ys = [run["mem_usage"]["mean"] for run in info[policy]]
		xs = [run["n_agents"] for run in info[policy]]
		if errorbars:
			yerr = [run["mem_usage"]["std"] for run in info[policy]]
			axs[i].errorbar(xs, ys, yerr, color=colours[i], ls="dotted", label=policy)
		else:
			axs[i].plot(xs, ys, color=colours[i], ls="dotted", label=policy)
		axs[i].set_title(policy)
	fig.legend()
	
	plt.savefig(graphs_dir / "memory_usage_graph.png")

def plot_max_mem_usage_graph(info, log_dir):
	graphs_dir = log_dir / "graphs"
	graphs_dir.mkdir(exist_ok=True)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlabel("Number of Ego Agents")
	ax.set_ylabel("Memory Usage")
	for policy in info:
		ys = [run["mem_usage"]["max"] for run in info[policy]]
		xs = [run["n_agents"] for run in info[policy]]
		ax.plot(xs, ys, ls="dotted", label=policy)
	ax.legend()
	ax.set_title("Maximum memory usage")
	plt.savefig(graphs_dir / "max_memory_usage_graph.png")

def plot_episode_steps_sec(info, log_dir, errorbars=True):
	graphs_dir = log_dir / "graphs"
	graphs_dir.mkdir(exist_ok=True)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlabel("Number of Ego Agents")
	ax.set_ylabel("Steps/Sec")
	for policy in info:
		ys = [run["run_episode_stats"]["steps/sec"]["mean"] for run in info[policy]]
		xs = [run["n_agents"] for run in info[policy]]
		if errorbars:
			yerr = [run["run_episode_stats"]["steps/sec"]["std"] for run in info[policy]]
			ax.errorbar(xs, ys, yerr, ls="dotted", label=policy)
		else:
			ax.plot(xs, ys, ls="dotted", label=policy)
	ax.legend()
	ax.set_title("Mean Steps/Sec")
	plt.savefig(graphs_dir / "steps_sec_graph.png")

def plot_episode_sim_wall(info, log_dir, errorbars=True):
	graphs_dir = log_dir / "graphs"
	graphs_dir.mkdir(exist_ok=True)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlabel("Number of Ego Agents")
	ax.set_ylabel("Sim/Wall")
	for policy in info:
		ys = [run["run_episode_stats"]["sim/wall"]["mean"] for run in info[policy]]
		xs = [run["n_agents"] for run in info[policy]]
		if errorbars:
			yerr = [run["run_episode_stats"]["sim/wall"]["std"] for run in info[policy]]
			ax.errorbar(xs, ys, yerr, ls="dotted", label=policy)
		else:
			ax.plot(xs, ys, ls="dotted", label=policy)
	ax.legend()
	ax.set_title("Mean Sim/Wall")
	plt.savefig(graphs_dir / "sim_wall_graph.png")


def plot_episode_sim_wall_min(info, log_dir):
	graphs_dir = log_dir / "graphs"
	graphs_dir.mkdir(exist_ok=True)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlabel("Number of Ego Agents")
	ax.set_ylabel("Minimum Sim/Wall")
	for policy in info:
		ys = [run["run_episode_stats"]["sim/wall"]["min"] for run in info[policy]]
		xs = [run["n_agents"] for run in info[policy]]	
		ax.plot(xs, ys, ls="dotted", label=policy)
	ax.legend()
	ax.set_title("Minimum Sim/Wall")
	plt.savefig(graphs_dir / "sim_wall_min_graph.png")


def plot_exeuction_time(info, log_dir):
	graphs_dir = log_dir / "graphs"
	graphs_dir.mkdir(exist_ok=True)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlabel("Number of Ego Agents")
	ax.set_ylabel("Total Execution time")
	for policy in info:
		ys = [run["cprofile_stats"]["total_time"] for run in info[policy]]
		xs = [run["n_agents"] for run in info[policy]]	
		ax.plot(xs, ys, label=policy)
	ax.legend()
	ax.set_title("Total Execution time")
	plt.savefig(graphs_dir / "total_execution_time.png")


def plot_profile_chart(policy_exp, name, log_dir, width=3):
	graphs_dir = log_dir / "graphs"
	graphs_dir.mkdir(exist_ok=True)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlabel("Number of Ego Agents")
	ax.set_ylabel("Proportion of execution time")
	profile_dicts = [run["cprofile_stats_norm"] for run in policy_exp]
	from operator import methodcaller
	dict_items = map(methodcaller('items'), profile_dicts)
	from collections import defaultdict
	dd = defaultdict(list)
	from itertools import chain
	for k, v in chain.from_iterable(dict_items):
		dd[k].append(v)
	xs = [run["n_agents"] for run in policy_exp]	
	bottom = [0 for _ in range(len(xs))]
	for func_name in dd:
		ax.bar(xs, dd[func_name], width, label=func_name, bottom=bottom)
		bottom = [bottom[i] + x for i, x in enumerate(dd[func_name])]
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	ax.set_title(f"Breakdown of exeuction time for {name} policy")
	plt.savefig(graphs_dir / f"{name}_profile_chart.png")	

def plot_vehicle_retention(policy_exp, name, log_dir, max_steps=None):
	graphs_dir = log_dir / "graphs"
	graphs_dir.mkdir(exist_ok=True)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlabel("Episode step")
	ax.set_ylabel("Number of active agents")

	hsv = plt.get_cmap('hsv')
	colors = hsv(np.linspace(0, 1.0, len(policy_exp)))
	for color, run in zip(colors, policy_exp):
		if "vehicle_retention" not in run:
			continue
		if max_steps is None:
			xs = range(0, len(run["vehicle_retention"]))
		else:
			xs = range(0, max_steps)
		ax.plot(
			xs, 
			run["vehicle_retention"][:max_steps], 
			color=color, 
			label=f"n={run['n_agents']}"
		)
	ax.legend()
	ax.set_ylim(bottom=0)
	ax.set_title(f"Vehicle retention over time for {name}")
	if max_steps is not None:
		plt.savefig(graphs_dir / f"vehicle_retention_{name}_{max_steps}.png")
	else:
		plt.savefig(graphs_dir / f"vehicle_retention_{name}.png")

if __name__ == "__main__":
	parser = argparse.ArgumentParser("generate-graphs")
	parser.add_argument(
		"--log-dir", help="Directory where experiment logs are located",
	)
	args = parser.parse_args()
	log_dir = Path(args.log_dir)
	runs = [f.path for f in os.scandir(log_dir) if f.is_dir() and f.name != "graphs"]
	experiment_info = extract_experiment_info(runs)
	#extract_mem_usage(experiment_info)
	#extract_stats_csv_info(experiment_info)
	#extract_profile_info(experiment_info)
	#plot_mem_usage_graph(experiment_info, log_dir)
	#plot_max_mem_usage_graph(experiment_info, log_dir)
	#plot_episode_steps_sec(experiment_info, log_dir, errorbars=False)
	#plot_episode_sim_wall(experiment_info, log_dir, errorbars=False)
	#plot_episode_sim_wall_min(experiment_info, log_dir)
	#plot_profile_chart(experiment_info["sac"], "sac", log_dir)
	#plot_profile_chart(experiment_info["ppo"], "ppo", log_dir)
	#plot_profile_chart(experiment_info["dqn"], "dqn", log_dir)
	#plot_profile_chart(experiment_info["dqn-discrete"], "dqn-discrete", log_dir)
	#plot_exeuction_time(experiment_info, log_dir)

	extract_vehicle_retention_info(experiment_info)
	#plot_vehicle_retention(experiment_info["dqn"], "dqn", log_dir, max_steps=50)
	#plot_vehicle_retention(experiment_info["ppo"], "ppo", log_dir, max_steps=50)
	#plot_vehicle_retention(experiment_info["sac"], "sac", log_dir, max_steps=50)
	plot_vehicle_retention(experiment_info["dqn_discrete"], "dqn discrete", log_dir, max_steps=100)
	plot_vehicle_retention(experiment_info["dqn_discrete"], "dqn discrete", log_dir, max_steps=50)
	plot_vehicle_retention(experiment_info["sac_discrete"], "sac discrete", log_dir, max_steps=100)
	plot_vehicle_retention(experiment_info["sac_discrete"], "sac discrete", log_dir, max_steps=50)
