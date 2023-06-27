#!/bin/bash

source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/CCGE2/3a92kbas --count 7 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/3a92kbas --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/3a92kbas --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/3a92kbas --count 6 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
