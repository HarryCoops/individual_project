#!/bin/bash
Algos="ppo_discreteRGB"
LogDir="chapter_3_benchmarks"
for algo in $Algos; do
	#python marl_scalability/train.py --batch-size 32 --maintain-agent-numbers --compression lz4 --profiler pyinstrument --use-marb --max-episode-steps 500 --scenario scenarios/big_circle --policy $algo --max-steps 2500 --headless --memprof --episodes 100 --log-dir $LogDir --n-agents 1	
	python marl_scalability/train.py --maintain-agent-numbers --profiler pyinstrument --max-episode-steps 500 --scenario scenarios/big_circle --policy $algo --max-steps 2500 --headless --memprof --episodes 100 --log-dir $LogDir --n-agents 1
	for ((i=5; i<=75; i+=10)); do
	    #python marl_scalability/train.py --maintain-agent-numbers --compression lz4 --profiler pyinstrument --use-marb --max-episode-steps 500 --scenario scenarios/big_circle --policy $algo --max-steps 2500 --headless --memprof --episodes 100 --batch-size 32 --log-dir $LogDir --n-agents $i
	    python marl_scalability/train.py --maintain-agent-numbers --profiler pyinstrument --max-episode-steps 500 --scenario scenarios/big_circle --policy $algo --max-steps 2500 --headless --memprof --episodes 100 --log-dir $LogDir --n-agents $i
	done
done
