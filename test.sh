#!/bin/bash

source venv/bin/activate

declare -a pids=()

python3 src/main.py --train --env_name="Hopper-v4" --wandb_name="CCGE_Hopper_10" --wandb --confidence_lambda=10 &
pids+=($!)
sleep 4

python3 src/main.py --train --env_name="Hopper-v4" --wandb_name="CCGE_Hopper_10" --wandb --confidence_lambda=10 &
pids+=($!)
sleep 4

python3 src/main.py --train --env_name="Hopper-v4" --wandb_name="CCGE_Hopper_100" --wandb --confidence_lambda=100 &
pids+=($!)
sleep 4

python3 src/main.py --train --env_name="Hopper-v4" --wandb_name="CCGE_Hopper_100" --wandb --confidence_lambda=100 &
pids+=($!)
sleep 4

for pid in ${pids[*]}; do
    wait $pid
done
