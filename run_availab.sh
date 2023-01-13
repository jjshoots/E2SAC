#!/bin/bash

source venv/bin/activate

wingman-compress-weights

declare -a pids=()

wandb agent jjshoots/CCGE2_oracle_search/grfpun8e --count 1 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2_oracle_search/grfpun8e --count 1 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2_oracle_search/grfpun8e --count 1 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
