#!/bin/bash
Algos="sac ppo dqn bdqn td3"
LogDir="profile_3"
for algo in $Algos; do
	python marl_scalability/train.py --task 0-4agents --level no-traffic --policy $algo,$algo,$algo,$algo --headless True --episodes 10 --eval-rate 1000000 --log-dir $LogDir
	python marl_scalability/train.py --task 0-3agents --level no-traffic --policy $algo,$algo,$algo --headless True --episodes 10 --eval-rate 1000000 --log-dir $LogDir
	python marl_scalability/train.py --task 0-2agents --level no-traffic --policy $algo,$algo --headless True --episodes 10 --eval-rate 1000000 --log-dir $LogDir
	python marl_scalability/train.py --task 0-1agents --level no-traffic --policy $algo --headless True --episodes 10 --eval-rate 1000000 --log-dir $LogDir
done
