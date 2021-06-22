#!/bin/bash
Algos="dqn_discreteRGB"
LogDir="tester23"
for algo in $Algos; do
 	python marl_scalability/train.py --profiler pyinstrument --use-marb --compression lz4 --max-episode-steps 500 --scenario scenarios/loop_4_lane --policy $algo --max-steps 1000 --headless --memprof --episodes 100 --log-dir $LogDir --n-agents 2
	 python marl_scalability/train.py --compression lz4 --profiler pyinstrument --max-episode-steps 500 --scenario scenarios/loop_4_lane --policy $algo --max-steps 1000 --headless --memprof --episodes 100 --log-dir $LogDir --n-agents 2
	#for ((i=70; i<=100; i+=10)); do
    #		python marl_scalability/train.py --max-episode-steps 5000 --scenario scenarios/loop_4_lane --policy $algo --headless --memprof --episodes 100 --log-dir $LogDir --profiler cProfile --n-agents $i
	#done
done
