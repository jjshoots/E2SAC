#!/bin/bash

source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/CCGE2/ncnzlpx5 --count 12 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/ncnzlpx5 --count 12 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/d0tfkuci --count 12 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/d0tfkuci --count 12 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
