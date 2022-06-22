#!/bin/bash
source venv/bin/activate

declare -a pids=()
wandb agent jjshoots/DQN2/xy1blq0i --count 10 &
pids+=($!)
sleep 10
wandb agent jjshoots/DQN2/xy1blq0i --count 10 &
pids+=($!)
sleep 10
wandb agent jjshoots/DQN2/xy1blq0i --count 10 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

declare -a pids=()
wandb agent jjshoots/DQN2/zlqur3uh --count 10 &
pids+=($!)
sleep 10
wandb agent jjshoots/DQN2/zlqur3uh --count 10 &
pids+=($!)
sleep 10
wandb agent jjshoots/DQN2/zlqur3uh --count 10 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

declare -a pids=()
wandb agent jjshoots/DQN2/4zjlky9u --count 10 &
pids+=($!)
sleep 10
wandb agent jjshoots/DQN2/4zjlky9u --count 10 &
pids+=($!)
sleep 10
wandb agent jjshoots/DQN2/4zjlky9u --count 10 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
