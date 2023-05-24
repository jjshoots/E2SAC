#!/bin/bash

source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/CCGE2/vigk09s5 --count 1 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/vigk09s5 --count 1 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/vigk09s5 --count 1 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/vigk09s5 --count 1 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
