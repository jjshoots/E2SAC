#!/bin/bash

source venv/bin/activate

declare -a pids=()

python3 src/main.py --train --wandb --wandb_name='ant_big' --env_name=AntPyBulletEnv-v0 &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name='hopper_big' --env_name=HopperPyBulletEnv-v0 &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name='walker_big' --env_name=Walker2DPyBulletEnv-v0 &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name='cheetah_big' --env_name=HalfCheetahPyBulletEnv-v0 &
pids+=($!)
for pid in ${pids[*]}; do
    wait $pid
done

