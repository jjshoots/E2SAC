#!/bin/bash

source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/CCGE2/vxtb4692 --count 4 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/vxtb4692 --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/vxtb4692 --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/vxtb4692 --count 3 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
