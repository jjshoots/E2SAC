#!/bin/bash
source venv/bin/activate

declare -a pids=()
wandb agent jjshoots/DQN2/uccgmxfq --count 10 &
pids+=($!)
sleep 10
wandb agent jjshoots/DQN2/uccgmxfq --count 10 &
pids+=($!)
sleep 10
wandb agent jjshoots/DQN2/uccgmxfq --count 10 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

declare -a pids=()
wandb agent jjshoots/DQN2/2flhn974 --count 10 &
pids+=($!)
sleep 10
wandb agent jjshoots/DQN2/2flhn974 --count 10 &
pids+=($!)
sleep 10
wandb agent jjshoots/DQN2/2flhn974 --count 10 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

declare -a pids=()
wandb agent jjshoots/DQN2/bapn8sg1 --count 10 &
pids+=($!)
sleep 10
wandb agent jjshoots/DQN2/bapn8sg1 --count 10 &
pids+=($!)
sleep 10
wandb agent jjshoots/DQN2/bapn8sg1 --count 10 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

