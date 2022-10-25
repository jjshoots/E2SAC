#!/bin/bash

source venv/bin/activate

declare -a pids=()

python3 src/mainSAC.py --train --env_name="Hopper-v4" --wandb_name="SAC_Hopper" --wandb &
pids+=($!)
sleep 4
python3 src/mainSAC.py --train --env_name="Hopper-v4" --wandb_name="SAC_Hopper" --wandb &
pids+=($!)
sleep 4
python3 src/main.py --train --env_name="Hopper-v4" --wandb_name="CCGE_Hopper" --wandb &
pids+=($!)
sleep 4
python3 src/main.py --train  --env_name="Hopper-v4" --wandb_name="CCGE_Hopper" --wandb &
pids+=($!)
sleep 4

for pid in ${pids[*]}; do
    wait $pid
done
