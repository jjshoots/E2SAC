#!/bin/bash
source venv/bin/activate

declare -a pids=()
wandb agent jjshoots/DQN3/kmwwbfkg --count 10 &
pids+=($!)
wandb agent jjshoots/DQN3/kmwwbfkg --count 10 &
pids+=($!)
wandb agent jjshoots/DQN3/kmwwbfkg --count 10 &
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done
