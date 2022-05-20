#!/bin/bash

source venv/bin/activate

# python3 src/main.py --train --wandb --wandb_name='CCGE1.1_ant' --env_name=AntPyBulletEnv-v0 &
# pids+=($!)

# python3 src/main.py --train --wandb --wandb_name='CCGE1.1_hopper' --env_name=HopperPyBulletEnv-v0 &
# pids+=($!)

# python3 src/main.py --train --wandb --wandb_name='CCGE1.1_walker' --env_name=Walker2DPyBulletEnv-v0 &
# pids+=($!)


python3 src/main.py --train --wandb --wandb_name='CCGE1.1_cheetah' --env_name=HalfCheetahPyBulletEnv-v0 --sub_size='norm' &
pids+=($!)

python3 src/main.py --train --wandb --wandb_name='CCGE2.1_hopper' --env_name=HopperPyBulletEnv-v0 --sub_size='smol' &
pids+=($!)

python3 src/main.py --train --wandb --wandb_name='CCGE2.1_hopper' --env_name=HopperPyBulletEnv-v0 --sub_size='smol' &
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done

python3 src/main.py --train --wandb --wandb_name='CCGE2.1_ant' --env_name=AntPyBulletEnv-v0 --sub_size='smol' &
pids+=($!)

python3 src/main.py --train --wandb --wandb_name='CCGE2.1_walker' --env_name=Walker2DPyBulletEnv-v0 --sub_size='smol' &
pids+=($!)

python3 src/main.py --train --wandb --wandb_name='CCGE2.1_cheetah' --env_name=HalfCheetahPyBulletEnv-v0 --sub_size='smol' &
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done

python3 src/main.py --train --wandb --wandb_name='CCGE2.1_ant' --env_name=AntPyBulletEnv-v0 --sub_size='smol' &
pids+=($!)

python3 src/main.py --train --wandb --wandb_name='CCGE2.1_walker' --env_name=Walker2DPyBulletEnv-v0 --sub_size='smol' &
pids+=($!)

python3 src/main.py --train --wandb --wandb_name='CCGE2.1_cheetah' --env_name=HalfCheetahPyBulletEnv-v0 --sub_size='smol' &
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done
