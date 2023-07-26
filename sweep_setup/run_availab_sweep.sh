#!/bin/bash

# this file has been automatically generated, do not edit manually

source venv/bin/activate
pip3 install gym==0.25.1
wingman-compress-weights

declare -a pids=()
wandb agent jjshoots/ccge2_stable_gym/ortymard --count 1 & 
pids+=($!)
sleep 10
wandb agent jjshoots/ccge2_stable_gym/ortymard --count 1 & 
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
