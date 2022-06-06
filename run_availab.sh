#!/bin/bash
source venv/bin/activate

declare -a pids=()
wandb agent jjshoots/DQN2/a0gjbznv --count 10 &
pids+=($!)
sleep 10
wandb agent jjshoots/DQN2/a0gjbznv --count 10 &
pids+=($!)
sleep 10
wandb agent jjshoots/DQN2/a0gjbznv --count 10 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

