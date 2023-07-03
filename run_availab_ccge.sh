#!/bin/bash

source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/CCGE2/wtfin9dd --count 7 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/wtfin9dd --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/wtfin9dd --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/wtfin9dd --count 6 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
