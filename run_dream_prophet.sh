#!/bin/bash

source venv/bin/activate


declare -a pids=()

CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/40efsd2p --count 1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/40efsd2p --count 1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/40efsd2p --count 1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/40efsd2p --count 1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/40efsd2p --count 1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/40efsd2p --count 1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/40efsd2p --count 1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/40efsd2p --count 1 &
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done

# declare -a pids=()

# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/7zrnctkl --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/7zrnctkl --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/7zrnctkl --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/7zrnctkl --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/7zrnctkl --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/7zrnctkl --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/7zrnctkl --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/7zrnctkl --count 1 &
# pids+=($!)

# for pid in ${pids[*]}; do
#     wait $pid
# done

# declare -a pids=()

# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/hvdsmej5 --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/hvdsmej5 --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/hvdsmej5 --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/hvdsmej5 --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/hvdsmej5 --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/hvdsmej5 --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/hvdsmej5 --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/hvdsmej5 --count 1 &
# pids+=($!)

# for pid in ${pids[*]}; do
#     wait $pid
# done

# declare -a pids=()

# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/do1jpm0c --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/do1jpm0c --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/do1jpm0c --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/do1jpm0c --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/do1jpm0c --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/do1jpm0c --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/do1jpm0c --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/do1jpm0c --count 1 &
# pids+=($!)

# for pid in ${pids[*]}; do
#     wait $pid
# done

# declare -a pids=()

# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/yp5pim4q --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/yp5pim4q --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/yp5pim4q --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/yp5pim4q --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/yp5pim4q --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/yp5pim4q --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/yp5pim4q --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/yp5pim4q --count 1 &
# pids+=($!)

# for pid in ${pids[*]}; do
#     wait $pid
# done

# declare -a pids=()

# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/w6k8glav --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/w6k8glav --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/w6k8glav --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/w6k8glav --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/w6k8glav --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/w6k8glav --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/w6k8glav --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/w6k8glav --count 1 &
# pids+=($!)

# for pid in ${pids[*]}; do
#     wait $pid
# done

# declare -a pids=()

# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/j0dvoamd --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/j0dvoamd --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/j0dvoamd --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/j0dvoamd --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/j0dvoamd --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/j0dvoamd --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/j0dvoamd --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/j0dvoamd --count 1 &
# pids+=($!)

# for pid in ${pids[*]}; do
#     wait $pid
# done

# declare -a pids=()

# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/wfvijb3o --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/wfvijb3o --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/wfvijb3o --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/wfvijb3o --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/wfvijb3o --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/wfvijb3o --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/wfvijb3o --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/wfvijb3o --count 1 &
# pids+=($!)

# for pid in ${pids[*]}; do
#     wait $pid
# done

# declare -a pids=()

# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/y9q16c2d --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/y9q16c2d --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/y9q16c2d --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/y9q16c2d --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/y9q16c2d --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/y9q16c2d --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/y9q16c2d --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/y9q16c2d --count 1 &
# pids+=($!)

# for pid in ${pids[*]}; do
#     wait $pid
# done

# declare -a pids=()

# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/femsqxpk --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/femsqxpk --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/femsqxpk --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/femsqxpk --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/femsqxpk --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/femsqxpk --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/femsqxpk --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/femsqxpk --count 1 &
# pids+=($!)

# for pid in ${pids[*]}; do
#     wait $pid
# done

# declare -a pids=()

# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/uapvwtal --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/uapvwtal --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/uapvwtal --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/uapvwtal --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/uapvwtal --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/uapvwtal --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/uapvwtal --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/uapvwtal --count 1 &
# pids+=($!)

# for pid in ${pids[*]}; do
#     wait $pid
# done

# declare -a pids=()

# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/t45aurh0 --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/t45aurh0 --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/t45aurh0 --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/t45aurh0 --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/t45aurh0 --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/t45aurh0 --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/t45aurh0 --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/t45aurh0 --count 1 &
# pids+=($!)

# for pid in ${pids[*]}; do
#     wait $pid
# done

# declare -a pids=()

# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/j7z38a2b --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/j7z38a2b --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/j7z38a2b --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/pybullet3/j7z38a2b --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/j7z38a2b --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/j7z38a2b --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/j7z38a2b --count 1 &
# pids+=($!)
# CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/pybullet3/j7z38a2b --count 1 &
# pids+=($!)
# pybullet3for pid in ${pids[*]}; do
#     wait $pid
# done
