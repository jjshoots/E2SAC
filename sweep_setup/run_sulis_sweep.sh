#!/bin/bash

# this file has been automatically generated, do not edit manually

source venv/bin/activate
# pip3 install -e .
# pip3 uninstall gymnasium-robotics -y
# pip3 install -r requirements.txt -U
# rm -rf weights/*
# wingman-compress-weights

declare -a pids=()
wandb agent jjshoots/CCGE2/nc8m5se6 --count 1 & 
pids+=($!)
sleep 20
wandb agent jjshoots/CCGE2/nc8m5se6 --count 1 & 
pids+=($!)
sleep 20
wandb agent jjshoots/CCGE2/nc8m5se6 --count 1 & 
pids+=($!)
sleep 20
wandb agent jjshoots/CCGE2/nc8m5se6 --count 1 & 
pids+=($!)
sleep 20
wandb agent jjshoots/CCGE2/nc8m5se6 --count 1 & 
pids+=($!)
sleep 20
wandb agent jjshoots/CCGE2/nc8m5se6 --count 1 & 
pids+=($!)
sleep 20

for pid in ${pids[*]}; do
    wait $pid
done
