#!/bin/bash

# this file has been automatically generated, do not edit manually

source venv/bin/activate
pip3 install -e .
pip3 uninstall gymnasium-robotics -y
pip3 install -r requirements.txt -U
rm -rf weights/*
wingman-compress-weights

declare -a pids=()
wandb agent jjshoots/CCGE2/xhecg0ph --count 2 & 
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/xhecg0ph --count 2 & 
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/xhecg0ph --count 2 & 
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
