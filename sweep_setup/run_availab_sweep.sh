#!/bin/bash

# this file has been automatically generated, do not edit manually

source venv/bin/activate
wingman-compress-weights


declare -a pids=()
wandb agent jjshoots// --count 1 & 
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
