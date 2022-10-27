#!/bin/bash

source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/CCGE2/ot1qxm41 --count 4 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/ot1qxm41 --count 4 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/5vzkut2a --count 4 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/5vzkut2a --count 4 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

