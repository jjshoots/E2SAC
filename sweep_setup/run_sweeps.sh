#!/bin/bash

# this file has been automatically generated, do not edit manually

source venv/bin/activate
pip3 install -r requirements.txt -U
wingman-compress-weights

declare -a pids=()
wandb agent jjshoots/CCGE2_oracle_search/nif3zqbb --count 1 & 
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2_oracle_search/nif3zqbb --count 1 & 
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2_oracle_search/nif3zqbb --count 1 & 
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
