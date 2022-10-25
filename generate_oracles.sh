#!/bin/bash

source venv/bin/activate

declare -a pids=()

python3 src/mainSAC.py --train --total_steps=300000 --wandb --env_name="Hopper-v4" --wandb_name="Small_Oracle_Hopper" &
pids+=($!)
sleep 4
python3 src/mainSAC.py --train --total_steps=300000 --wandb --env_name="HalfCheetah-v4" --wandb_name="Small_Oracle_HalfCheetah" &
pids+=($!)
sleep 4
python3 src/mainSAC.py --train --total_steps=300000 --wandb --env_name="Walker2d-v4" --wandb_name="Small_Oracle_Walker2d" &
pids+=($!)
sleep 4
python3 src/mainSAC.py --train --total_steps=300000 --wandb --env_name="Ant-v4" --wandb_name="Small_Oracle_Ant" &
pids+=($!)
sleep 4

for pid in ${pids[*]}; do
    wait $pid
done
