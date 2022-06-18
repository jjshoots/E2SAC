#!/bin/bash
source venv/bin/activate

declare -a pids=()
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/5bv1o5du --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/5bv1o5du --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/5bv1o5du --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/5bv1o5du --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/5bv1o5du --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/5bv1o5du --count 10 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

declare -a pids=()
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/t3e9smkh --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/t3e9smkh --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/t3e9smkh --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/t3e9smkh --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/t3e9smkh --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/t3e9smkh --count 10 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

declare -a pids=()
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/6ssn48ak --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/6ssn48ak --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/6ssn48ak --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/6ssn48ak --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/6ssn48ak --count 10 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/6ssn48ak --count 10 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
