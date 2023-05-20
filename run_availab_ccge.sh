#!/bin/bash

source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/CCGE2/i5l7a9wm --count 1 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/i5l7a9wm --count 1 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/i5l7a9wm --count 1 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/i5l7a9wm --count 1 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
