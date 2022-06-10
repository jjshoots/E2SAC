#!/bin/bash
source venv/bin/activate

declare -a pids=()
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/ns2i31ul --count 5 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/ns2i31ul --count 5 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/ns2i31ul --count 5 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/ns2i31ul --count 5 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/ns2i31ul --count 5 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/ns2i31ul --count 5 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

declare -a pids=()
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/146u4rcg --count 5 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/146u4rcg --count 5 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/146u4rcg --count 5 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/146u4rcg --count 5 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/146u4rcg --count 5 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/146u4rcg --count 5 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

declare -a pids=()
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/0d1c22d0 --count 5 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/0d1c22d0 --count 5 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/DQN2/0d1c22d0 --count 5 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/0d1c22d0 --count 5 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/0d1c22d0 --count 5 &
pids+=($!)
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/DQN2/0d1c22d0 --count 5 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

