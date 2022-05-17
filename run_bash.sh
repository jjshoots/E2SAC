#!/bin/bash

source venv/bin/activate

declare -a pids=()

python3 src/mainSAC.py --train --wandb --wandb_name='SAC_ant' --env_name=AntPyBulletEnv-v0 &
pids+=($!)
python3 src/mainSAC.py --train --wandb --wandb_name='SAC_hopper' --env_name=HopperPyBulletEnv-v0 &
pids+=($!)
python3 src/mainSAC.py --train --wandb --wandb_name='SAC_walker' --env_name=Walker2DPyBulletEnv-v0 &
pids+=($!)
python3 src/mainSAC.py --train --wandb --wandb_name='SAC_cheetah' --env_name=HalfCheetahPyBulletEnv-v0 &
pids+=($!)
for pid in ${pids[*]}; do
    wait $pid
done

