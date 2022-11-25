#!/bin/bash

source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/CCGE2/wate6zrz --count 8 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/wate6zrz --count 8 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/wate6zrz --count 8 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
