#!/bin/bash
time=$(date "+%Y-%m-%d-%H-%M-%S")

mkdir outfile/$time




nohup python -u main.py --algorithm "pFedMe" --epochs 40 --noise_multiplier 0.1 --if_DP False --_running_time $time > ./outfile/$time/running_FedAvg.out 2>&1 &
#wait
nohup python -u main.py --algorithm "pFedMe" --epochs 60 --noise_multiplier 0.1 --if_DP True --_running_time $time > ./outfile/$time/running_FedAvg2.out 2>&1 &
#wait
nohup python -u main.py --algorithm "pFedMe" --epochs 80 --noise_multiplier 0.2 --if_DP True --_running_time $time > ./outfile/$time/running_pFed3.out 2>&1 &







wait


done








