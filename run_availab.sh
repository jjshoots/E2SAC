#!/bin/bash
source venv/bin/activate

declare -a pids=()
wandb agent jjshoots/DQN3/eo4xye3x --count 10 &
pids+=($!)
wandb agent jjshoots/DQN3/eo4xye3x --count 10 &
pids+=($!)
wandb agent jjshoots/DQN3/eo4xye3x --count 10 &
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done
