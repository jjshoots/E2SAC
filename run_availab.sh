#!/bin/bash

source venv/bin/activate


declare -a pids=()

wandb agent jjshoots/pybullet3/6t1dtya7 --count 3 &
pids+=($!)
wandb agent jjshoots/pybullet3/6t1dtya7 --count 3 &
pids+=($!)
wandb agent jjshoots/pybullet3/6t1dtya7 --count 3 &
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done

declare -a pids=()

wandb agent jjshoots/pybullet3/6bls9o5w --count 3 &
pids+=($!)
wandb agent jjshoots/pybullet3/6bls9o5w --count 3 &
pids+=($!)
wandb agent jjshoots/pybullet3/6bls9o5w --count 3 &
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done

declare -a pids=()

wandb agent jjshoots/pybullet3/2c7ze04n --count 3 &
pids+=($!)
wandb agent jjshoots/pybullet3/2c7ze04n --count 3 &
pids+=($!)
wandb agent jjshoots/pybullet3/2c7ze04n --count 3 &
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done

declare -a pids=()

wandb agent jjshoots/pybullet3/s7j5w7vg --count 3 &
pids+=($!)
wandb agent jjshoots/pybullet3/s7j5w7vg --count 3 &
pids+=($!)
wandb agent jjshoots/pybullet3/s7j5w7vg --count 3 &
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done
