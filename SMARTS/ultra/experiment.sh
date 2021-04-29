#!/bin/bash
Algos="dqn ppo sac"
for algo in $Algos; do
	python ultra/train.py --task 0-4agents --level no-traffic --policy $algo,$algo,$algo,$algo --headless True --episodes 10 --eval-rate 100000
	python ultra/train.py --task 0-3agents --level no-traffic --policy $algo,$algo,$algo --headless True --episodes 10 --eval-rate 100000
	python ultra/train.py --task 0-2agents --level no-traffic --policy $algo,$algo --headless True --episodes 10 --eval-rate 100000
	python ultra/train.py --task 0-1agents --level no-traffic --policy $algo --headless True --episodes 10 --eval-rate 100000
done
