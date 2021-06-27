import csv
import os
import json
import dill
import pandas as pd
import numpy as np
import argparse 
from pathlib import Path
from pprint import pprint

import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

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
		
		if run_info["policy"] in info:
			info[run_info["policy"]].append(run_info)
		else:
			info[run_info["policy"]] = [run_info,]
	for policy in info:
		info[policy].sort(key=lambda run_info: run_info["n_agents"])
	return info

def extract_agent_metadata(info):
	for policy in info:
		for run in info[policy]:
			agent_metadata_path = Path(run["path"]) / "agent_metadata.pkl"
			with open(agent_metadata_path, "rb") as f:
				run["agent_metadata"] = dill.loads(f.read())
			print(run["agent_metadata"])

def extract_args_info(info):
	for policy in info:
		for run in info[policy]:
			train_args_path = Path(run["path"]) / "train_args.json"
			with open(train_args_path) as f:
				run["train_args"] = json.loads(f.read())


def extract_mem_usage(info, store_df=False):
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
				if store_df:
					run["mem_usage"]["df"] = mem_usage

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
				run["run_episode_stats"]["overall_total_steps"] = stats["total steps"].sum()


def extract_profile_info(info, add_norm=True):
	for policy in info:
		for run in info[policy]:
			profile_csv_path = Path(run["path"]) / "cprofile.csv"
			if profile_csv_path.exists():
				profile = pd.read_csv(profile_csv_path)
				total_time = profile.iloc[0]["cumtime"]
				func_col = "filename:lineno(function)"
				run["cprofile_stats"] = {}
				sample_time = profile.loc[
					profile[func_col] == "/marl/marl_scalability/baselines/common/image_replay_buffer.py:200(sample)"
				]["cumtime"].iloc[0]
				run["cprofile_stats"]["sample_time"] = sample_time
				core_step_time = profile.loc[
					profile[func_col] == "/src/smarts/core/smarts.py:167(step)"
				]["cumtime"].iloc[0]
				run["cprofile_stats"]["core_step_time"] = core_step_time
				gen_logs_time = profile.loc[
					profile[func_col] == "/marl/marl_scalability/env/scalability_env.py:66(generate_logs)"
				]["cumtime"].iloc[0]
				run["cprofile_stats"]["gen_logs_time"] = gen_logs_time
				reward_time = profile.loc[
					profile[func_col] == "/marl/marl_scalability/baselines/image_adapter.py:92(reward_adapter)"
				]["cumtime"].iloc[0]
				run["cprofile_stats"]["reward_time"] = reward_time / 2
				obs_time = profile.loc[
						profile[func_col] == "/marl/marl_scalability/baselines/image_adapter.py:87(observation_adapter)"
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
					run["cprofile_stats_norm"]["sample_time"] = sample_time / total_time

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


def extract_execution_time_from_pyinstrument_info(info):
	for policy in info:
		for run in info[policy]:
			pyinstrument_path = Path(run["path"]) / "profile.html"
			if pyinstrument_path.exists():
				run["pyinstrument_info"] = {}
				with open(pyinstrument_path) as f:
					print(run)
					html = f.read()
					duration = float(html.split("\"duration\": ")[1][:4])
					run["pyinstrument_info"]["duration"] = duration
					env_step_time = float(
						html.split(
							"\"/marl/marl_scalability/env/scalability_env.py\",\"line_no\": 62,\"time\": "
						)[1][:4]
					)
					run["pyinstrument_info"]["env_step_time"] = env_step_time
					env_reset = html.split(
						"\"/src/smarts/env/hiway_env.py\",\"line_no\": 186,\"time\":" 
					)
					env_reset_time = float(
						env_reset[1][:4] if len(env_reset) > 1 else 0
					)
					run["pyinstrument_info"]["env_reset_time"] = env_reset_time
					sample_time = html.split(
						"multi_agent_image_replay_buffer.py\",\"line_no\": 221,\"time\": "
					)
			
					sample_time = float(sample_time[1][:4]) if len(sample_time) > 1 else None
					sample_time = float(
						html.split(
							"image_replay_buffer.py\",\"line_no\": 202,\"time\": " 
						)[1][:4]
					) if sample_time is None else sample_time
					run["pyinstrument_info"]["sample_time"] = sample_time
			
					agent_training_time = duration - (env_step_time + env_reset_time)
					run["pyinstrument_info"]["agent_training_time"] = agent_training_time


def plot_pyinst_execution_time(info, log_dir):
	graphs_dir = log_dir / "graphs"
	graphs_dir.mkdir(exist_ok=True)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlabel("Number of Ego Agents")
	ax.set_ylabel("Total execution time (s)")
	for policy in info:
		ys = [run["pyinstrument_info"]["duration"] for run in info[policy]]
		xs = [run["n_agents"] for run in info[policy]]
		ax.plot(xs, ys, label=policy)
		print(xs, ys)
	ax.legend()
	plt.savefig(graphs_dir / "execution_time_graph.png")

def plot_pyinst_agent_time(info, log_dir):
	graphs_dir = log_dir / "graphs"
	graphs_dir.mkdir(exist_ok=True)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlabel("Number of Ego Agents")
	ax.set_ylabel("Agent training time (s)")
	for policy in info:
		ys = [run["pyinstrument_info"]["agent_training_time"] for run in info[policy]]
		xs = [run["n_agents"] for run in info[policy]]
		ax.plot(xs, ys, label=policy)
	ax.legend()
	plt.savefig(graphs_dir / "training_time_graph.png")


def plot_pyinst_sample_time(info, log_dir):
	graphs_dir = log_dir / "graphs"
	graphs_dir.mkdir(exist_ok=True)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlabel("Number of Ego Agents")
	ax.set_ylabel("Buffer sample time (s)")
	for policy in info:
		ys = [run["pyinstrument_info"]["sample_time"] for run in info[policy]]
		xs = [run["n_agents"] for run in info[policy]]
		ax.plot(xs, ys, label=policy)
	ax.legend()
	plt.savefig(graphs_dir / "sample_time_graph.png")


def plot_pyinst_sample_time_propotion(info, log_dir):
	graphs_dir = log_dir / "graphs"
	graphs_dir.mkdir(exist_ok=True)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlabel("Number of Ego Agents")
	ax.set_ylabel("Proportion of training time spent sampling the replay buffer")
	for policy in info:
		ys = [
			run["pyinstrument_info"]["sample_time"] / run["pyinstrument_info"]["duration"]
			for run in info[policy]
		]
		xs = [run["n_agents"] for run in info[policy]]
		ax.plot(xs, ys, label=policy)
	ax.legend()
	plt.savefig(graphs_dir / "sample_time_proportion_graph.png")


def plot_mem_usage_graph(info, log_dir, errorbars=True):
	graphs_dir = log_dir / "graphs"
	graphs_dir.mkdir(exist_ok=True)
	fig, axs = plt.subplots(len(info))
	#ax.set_xlabel("Number of Ego Agents")
	#ax.set_ylabel("Memory Usage")
	#colours = ["b", "g", "orange"]
	# color=colours[i],
	# color=colours[i], 
	for i, policy in enumerate(info):
		ys = [run["mem_usage"]["mean"] for run in info[policy]]
		xs = [run["n_agents"] for run in info[policy]]
		if errorbars:
			yerr = [run["mem_usage"]["std"] for run in info[policy]]
			axs[i].errorbar(xs, ys, yerr, ls="dotted", label=policy)
		else:
			axs[i].plot(xs, ys, ls="dotted", label=policy)
		axs[i].set_title(policy)
	fig.legend()
	
	plt.savefig(graphs_dir / "memory_usage_graph.png")

def plot_max_mem_usage_graph(info, log_dir):
	graphs_dir = log_dir / "graphs"
	graphs_dir.mkdir(exist_ok=True)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlabel("Number of Ego Agents")
	ax.set_ylabel("Peak Memory Usage (GB)")
	for policy in info:
		ys = [run["mem_usage"]["max"] for run in info[policy]]
		xs = [run["n_agents"] for run in info[policy]]
		ax.plot(xs, ys, label=policy.split("_")[0].upper())
	ax.legend()
	gig_formatter = FuncFormatter(gigabyte)
	ax.yaxis.set_major_formatter(gig_formatter)
	#ax.set_yscale("log")
	#ax.set_title("Maximum memory usage")
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
			ax.errorbar(xs, ys, yerr, ls="dotted", label=policy.split("_")[0].upper())
		else:
			ax.plot(xs, ys, ls="solid", label=policy.split("_")[0].upper())
	ax.legend()
	ax.set_xlim(left=30)
	ax.set_ylim(top=2, bottom=0)
	#ax.set_title("Mean Steps/Sec")
	plt.savefig(graphs_dir / "steps_sec_graph.png")

def plot_episode_sim_wall(info, log_dir, errorbars=True, start=None):
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
			ax.plot(xs, ys, ls="solid", label=policy.split("_")[0].upper())
	ax.legend()
	#ax.set_title("Mean Sim/Wall")
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
		ax.plot(xs, ys, ls="solid", label=policy.split("_")[0].upper())
	ax.legend()
	#ax.set_title("Minimum Sim/Wall")
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
	#ax.set_title(f"Agent survival over time for {name.upper()}")
	if max_steps is not None:
		plt.savefig(graphs_dir / f"agent_survival_{name}_{max_steps}.png")
	else:
		plt.savefig(graphs_dir / f"agent_survival_{name}.png")

def gigabyte(y, pos):
	return "%1.1fGB" % (y*1e-9)

def gigabyte_from_kb(y, pos):
	return "%1.1fGB" % (y*1e-6)

def gigabyte_from_mb(y, pos):
	return "%1.1fGB" % (y*1e-3)

def hours(x, pos):
	seconds = int(x)
	hours, seconds = divmod(seconds, 3600)
	minutes, seconds = divmod(seconds, 60)
	return f"{hours}:{minutes}:{seconds}"

def plot_mem_usage_over_time_(policy_exp, name, log_dir):
	graphs_dir = log_dir / "graphs"
	graphs_dir.mkdir(exist_ok=True)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlabel("Step")
	ax.set_ylabel("Memory usage")
	for run in policy_exp:
		if "mem_usage" not in run:
			continue
		df = run["mem_usage"]["df"]
		ax.plot(
			df.index, 
			df["mem_usage"], 
			label=name
		)
	ax.legend()
	gig_formatter = FuncFormatter(gigabyte_from_mb)
	ax.yaxis.set_major_formatter(gig_formatter)
	plt.savefig(graphs_dir / f"mem_usage_over_time_{name}.png")

def plot_mem_usage_over_time(policy_exp, name, log_dir):
	graphs_dir = log_dir / "graphs"
	graphs_dir.mkdir(exist_ok=True)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlabel("Step")
	ax.set_ylabel("Memory usage")
	labels = ["uncompressed", "compressed", "256_greyscale", "256", "84_greyscale"]
	cmap = cm.get_cmap("hsv")
	colors = cmap(np.linspace(0, .9, len(policy_exp)))
	for color, run, label in zip(colors, policy_exp, labels):
		if "mem_usage" not in run:
			continue
		df = run["mem_usage"]["df"]
		col_list = list(df)
		if len(col_list) == 3:
			ind, mem_usage, step = col_list
			col_list = [ind, step, mem_usage]
		elif len(col_list) == 4:
			ind, mem_usage, step, replay_usage = col_list
			col_list = [ind, step, mem_usage, replay_usage]
		df.columns = col_list
		ax.plot(
			df["step"], 
			df["mem_usage"], 
			label=run["train_args"]["desc"] if "train_args" in run else label
		)
	#ax.legend()
	#ax.set_title(f"Memory usage over time for {name} agent with top-down RGB input")
	gig_formatter = FuncFormatter(gigabyte)
	ax.yaxis.set_major_formatter(gig_formatter)
	#ax.set_ylim(bottom=0)
	plt.savefig(graphs_dir / f"mem_usage_over_time_{name}.png")


def plot_replay_usage_over_time(policy_exp, name, log_dir):
	graphs_dir = log_dir / "graphs"
	graphs_dir.mkdir(exist_ok=True)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlabel("Step")
	ax.set_ylabel("Memory usage")
	labels = ["compressed", "uncompressed", "256_greyscale", "256", "84_greyscale"]
	cmap = cm.get_cmap("hsv")
	colors = cmap(np.linspace(0, .9, len(policy_exp)))
	for run, label in zip(policy_exp, labels):
		if "mem_usage" not in run:
			continue
		df = run["mem_usage"]["df"]
		col_list = list(df)
		if len(col_list) == 4:
			ind, mem_usage, step, replay_usage = col_list
			col_list = [ind, step, mem_usage, replay_usage]
		df.columns = col_list
		ax.plot(
			df["step"], 
			df["replay_usage"], 
			label=run["train_args"]["desc"] if "train_args" in run else label
		)
	ax.legend()
	#ax.set_title(f"Replay buffer memory usage over time for {name} agent\n with top-down RGB input")
	gig_formatter = FuncFormatter(gigabyte)
	ax.yaxis.set_major_formatter(gig_formatter)
	ax.set_ylim(bottom=0)
	plt.savefig(graphs_dir / f"replay_usage_over_time_{name}.png", bbox_inches="tight")

def plot_batch_size_graph(info, log_dir):
	graphs_dir = log_dir / "graphs"
	graphs_dir.mkdir(exist_ok=True)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlabel("Batch Size")
	ax.set_ylabel("Total Sample Time (s)")
	for policy in info:
		ys = sorted([run["cprofile_stats"]["sample_time"] for run in info[policy]])
		xs = sorted([int(run["train_args"]["desc"].split("_")[1]) for run in info[policy]])
		ax.plot(xs, ys, label=policy, linestyle="--", marker="o")
	ax.legend()
	ax.set_xticks([x for x in range(0, 257, 32)])
	plt.savefig(graphs_dir / f"batch_size_vs_execution time.png")



if __name__ == "__main__":
	parser = argparse.ArgumentParser("generate-graphs")
	parser.add_argument(
		"--log-dir", help="Directory where experiment logs are located",
	)
	args = parser.parse_args()
	log_dir = Path(args.log_dir)
	runs = [f.path for f in os.scandir(log_dir) if f.is_dir() and f.name != "graphs"]
	experiment_info = extract_experiment_info(runs)
	extract_execution_time_from_pyinstrument_info(experiment_info)
	plot_pyinst_agent_time(experiment_info, log_dir)
	plot_pyinst_execution_time(experiment_info, log_dir)
	plot_pyinst_sample_time(experiment_info, log_dir)
	plot_pyinst_sample_time_propotion(experiment_info, log_dir)
	#extract_mem_usage(experiment_info, store_df=True)
	#extract_stats_csv_info(experiment_info)
	#extract_profile_info(experiment_info)
	extract_args_info(experiment_info)
	#plot_mem_usage_graph(experiment_info, log_dir)
	#plot_max_mem_usage_graph(experiment_info, log_dir)
	#plot_batch_size_graph(experiment_info, log_dir)
	#plot_episode_steps_sec(experiment_info, log_dir, errorbars=False)
	#plot_episode_sim_wall(experiment_info, log_dir, errorbars=False)
	#plot_episode_sim_wall_min(experiment_info, log_dir)
	#plot_profile_chart(experiment_info["sac"], "sac", log_dir)
	#plot_profile_chart(experiment_info["ppo"], "ppo", log_dir)
	#plot_profile_chart(experiment_info["dqn"], "dqn", log_dir)
	#plot_profile_chart(experiment_info["dqn-discrete"], "dqn-discrete", log_dir)
	#plot_exeuction_time(experiment_info, log_dir)
	#plot_mem_usage_over_time(experiment_info["ppo_discreteRGB"], "ppo_discreteRGB", log_dir)
	#plot_mem_usage_over_time(experiment_info["dqn_discreteRGB"], "dqn_discreteRGB", log_dir)
	#plot_mem_usage_over_time(experiment_info["dqn_discreteRGB"], "dqn_discreteRGB", log_dir)
	#extract_vehicle_retention_info(experiment_info)
	#plot_vehicle_retention(experiment_info["ppo"], "ppo", log_dir, max_steps=50)
	#plot_vehicle_retention(experiment_info["sac"], "sac", log_dir, max_steps=50)
	#plot_vehicle_retention(experiment_info["dqn_discreteRGB"], "dqn discrete", log_dir, max_steps=100)
	#plot_vehicle_retention(experiment_info["dqn_discreteRGB"], "dqn discrete", log_dir, max_steps=50)
	#plot_vehicle_retention(experiment_info["sac_discreteRGB"], "sac discrete", log_dir, max_steps=100)
	#plot_vehicle_retention(experiment_info["sac_discreteRGB"], "sac discrete", log_dir, max_steps=50)
	#plot_vehicle_retention(experiment_info["ppo_discreteRGB"], "ppo discrete", log_dir, max_steps=100)
	#plot_vehicle_retention(experiment_info["ppo_discreteRGB"], "ppo discrete", log_dir, max_steps=50)

	#plot_mem_usage_graph(experiment_info, log_dir)
	#plot_max_mem_usage_graph(experiment_info, log_dir)
	#extract_args_info(experiment_info)
	#plot_mem_usage_over_time(experiment_info["dqn_discreteRGB"], "DQN (discrete)", log_dir)
	#plot_replay_usage_over_time(experiment_info["dqn_discreteRGB"], "dqn discrete (high dim)", log_dir)