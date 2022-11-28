#!/bin/bash

source venv/bin/activate
pip3 install -r requirements.txt -U

declare -a pids=()
wandb agent jjshoots/CCGE2/el7n6k0m --count 8 & 
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/el7n6k0m --count 8 & 
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/el7n6k0m --count 8 & 
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
