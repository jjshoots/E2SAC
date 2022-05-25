#!/bin/bash
source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/carracing_sweep2/czd1qkse --count 1 &
pids+=($!)
sleep 10
wandb agent jjshoots/carracing_sweep2/czd1qkse --count 1 &
pids+=($!)
sleep 10
wandb agent jjshoots/carracing_sweep2/czd1qkse --count 1 &
pids+=($!)
sleep 10
wandb agent jjshoots/carracing_sweep2/czd1qkse --count 1 &
pids+=($!)
sleep 10
wandb agent jjshoots/carracing_sweep2/czd1qkse --count 1 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
