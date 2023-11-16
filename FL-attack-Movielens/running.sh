#!/bin/bash
time=$(date "+%Y-%m-%d-%H-%M-%S")

mkdir outfile/$time



# no dp
nohup python -u main.py --algorithm "pFedMe" --epochs 40 --noise_multiplier 0.05 --if_DP False --attack_rate 0.4 --_running_time $time > ./outfile/$time/running_pFed.out 2>&1 &









wait


done








