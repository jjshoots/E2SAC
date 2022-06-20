#!/bin/bash
source venv/bin/activate

declare -a pids=()
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/z7whmidz --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/z7whmidz --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/z7whmidz --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/z7whmidz --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/z7whmidz --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/z7whmidz --count 10 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

declare -a pids=()
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/nc1ui7ok --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/nc1ui7ok --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/nc1ui7ok --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/nc1ui7ok --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/nc1ui7ok --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/nc1ui7ok --count 10 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

declare -a pids=()
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/7vjjj1qc --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/7vjjj1qc --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/7vjjj1qc --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/7vjjj1qc --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/7vjjj1qc --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/7vjjj1qc --count 10 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
