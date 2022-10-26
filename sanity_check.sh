#!/bin/bash

source venv/bin/activate

declare -a pids=()

python3 src/main.py --train --env_name="Hopper-v4" --wandb --wandb_name="test" --wandb --total_steps=1000 &
pids+=($!)
sleep 4

for pid in ${pids[*]}; do
    wait $pid
done
