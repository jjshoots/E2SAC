#!/bin/bash

# this file has been automatically generated, do not edit manually

source venv/bin/activate
wingman-compress-weights


declare -a pids=()
wandb agent jjshoots/ccge2_railway_thesis/0l3ww4h3 --count 5 & 
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
