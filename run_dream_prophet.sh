#!/bin/bash
source venv/bin/activate

declare -a pids=()

CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/carracing_sweep2/y25egso1 --count 15 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/carracing_sweep2/y25egso1 --count 15 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
