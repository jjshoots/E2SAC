#!/bin/bash

source venv/bin/activate

for i in {0..3}; do
    declare -a pids=()

    CUDA_VISIBLE_DEVICES=0 python3 src/mainSAC.py --train --wandb --wandb_name='SAC_ant' --env_name=AntPyBulletEnv-v0 &
    pids+=($!)
    CUDA_VISIBLE_DEVICES=0 python3 src/mainSAC.py --train --wandb --wandb_name='SAC_ant' --env_name=AntPyBulletEnv-v0 &
    pids+=($!)
    CUDA_VISIBLE_DEVICES=0 python3 src/mainSAC.py --train --wandb --wandb_name='SAC_ant' --env_name=AntPyBulletEnv-v0 &
    pids+=($!)

    CUDA_VISIBLE_DEVICES=0 python3 src/mainSAC.py --train --wandb --wandb_name='SAC_hopper' --env_name=HopperPyBulletEnv-v0 &
    pids+=($!)
    CUDA_VISIBLE_DEVICES=0 python3 src/mainSAC.py --train --wandb --wandb_name='SAC_hopper' --env_name=HopperPyBulletEnv-v0 &
    pids+=($!)
    CUDA_VISIBLE_DEVICES=0 python3 src/mainSAC.py --train --wandb --wandb_name='SAC_hopper' --env_name=HopperPyBulletEnv-v0 &
    pids+=($!)

    CUDA_VISIBLE_DEVICES=1 python3 src/mainSAC.py --train --wandb --wandb_name='SAC_walker' --env_name=Walker2DPyBulletEnv-v0 &
    pids+=($!)
    CUDA_VISIBLE_DEVICES=1 python3 src/mainSAC.py --train --wandb --wandb_name='SAC_walker' --env_name=Walker2DPyBulletEnv-v0 &
    pids+=($!)
    CUDA_VISIBLE_DEVICES=1 python3 src/mainSAC.py --train --wandb --wandb_name='SAC_walker' --env_name=Walker2DPyBulletEnv-v0 &
    pids+=($!)

    CUDA_VISIBLE_DEVICES=1 python3 src/mainSAC.py --train --wandb --wandb_name='SAC_cheetah' --env_name=HalfCheetahPyBulletEnv-v0 &
    pids+=($!)
    CUDA_VISIBLE_DEVICES=1 python3 src/mainSAC.py --train --wandb --wandb_name='SAC_cheetah' --env_name=HalfCheetahPyBulletEnv-v0 &
    pids+=($!)
    CUDA_VISIBLE_DEVICES=1 python3 src/mainSAC.py --train --wandb --wandb_name='SAC_cheetah' --env_name=HalfCheetahPyBulletEnv-v0 &
    pids+=($!)

    for pid in ${pids[*]}; do
        wait $pid
    done
done
