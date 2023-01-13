#!/bin/bash

source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/CCGE2/44l7ajlp --count 4 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/44l7ajlp --count 4 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/44l7ajlp --count 4 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/44l7ajlp --count 4 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/44l7ajlp --count 4 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/44l7ajlp --count 4 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
