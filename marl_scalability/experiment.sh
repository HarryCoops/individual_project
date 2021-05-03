#!/bin/bash
Algos="sac"
LogDir="large_profile"
for algo in $Algos; do
 	python marl_scalability/train.py --scenario scenarios/loop_no_sv --policy $algo --headless --memprof --episodes 1000 --log-dir $LogDir --profiler cProfile --n-agents 1
	for ((i=5; i<=50; i+=5)); do
    		python marl_scalability/train.py --scenario scenarios/loop_no_sv --policy $algo --headless --memprof --episodes 1000 --log-dir $LogDir --profiler cProfile --n-agents $i
	done
done
