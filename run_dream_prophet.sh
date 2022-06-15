#!/bin/bash
source venv/bin/activate

declare -a pids=()
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/dotzndpe --count 2 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/dotzndpe --count 2 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/dotzndpe --count 2 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/dotzndpe --count 2 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/dotzndpe --count 2 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/dotzndpe --count 2 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
