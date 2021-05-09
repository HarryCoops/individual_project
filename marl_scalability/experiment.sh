#!/bin/bash
Algos="dqn_discrete"
LogDir="dqn_discrete_experiment_many_step"
for algo in $Algos; do
 	#python marl_scalability/train.py --scenario scenarios/loop_no_sv --policy $algo --memprof --episodes 10 --log-dir $LogDir --profiler pyinstrument --n-agents 1
	for ((i=40; i<=100; i+=20)); do
    		python marl_scalability/train.py --headless --scenario scenarios/loop_4_lane --policy $algo --memprof --episodes 20 --log-dir $LogDir --profiler cProfile --n-agents $i
	done
done
