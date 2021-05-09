#!/bin/bash
Algos="sac"
LogDir="rerun_with_spread_vehicles_and_for_longer"
for algo in $Algos; do
 	python marl_scalability/train.py --max-episode-steps 5000 --scenario scenarios/loop_4_lane --policy $algo --headless --memprof --episodes 100 --log-dir $LogDir --profiler cProfile --n-agents 1
	for ((i=70; i<=100; i+=10)); do
    		python marl_scalability/train.py --max-episode-steps 5000 --scenario scenarios/loop_4_lane --policy $algo --headless --memprof --episodes 100 --log-dir $LogDir --profiler cProfile --n-agents $i
	done
done
