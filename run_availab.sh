#!/bin/bash
source venv/bin/activate

declare -a pids=()
wandb agent jjshoots/DQN2/u7k2k7qo --count 10 &
pids+=($!)
sleep 10
wandb agent jjshoots/DQN2/u7k2k7qo --count 10 &
pids+=($!)
sleep 10
wandb agent jjshoots/DQN2/u7k2k7qo --count 10 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

declare -a pids=()
wandb agent jjshoots/DQN2/emhvyijs --count 10 &
pids+=($!)
sleep 10
wandb agent jjshoots/DQN2/emhvyijs --count 10 &
pids+=($!)
sleep 10
wandb agent jjshoots/DQN2/emhvyijs --count 10 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

