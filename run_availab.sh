#!/bin/bash

source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/CCGE2/sinopd9a --count 4 &
pids+=($!)
sleep 5
wandb agent jjshoots/CCGE2/sinopd9a --count 4 &
pids+=($!)
sleep 5
wandb agent jjshoots/CCGE2/sinopd9a --count 4 &
pids+=($!)
sleep 5
wandb agent jjshoots/CCGE2/sinopd9a --count 4 &
pids+=($!)
sleep 5

for pid in ${pids[*]}; do
    wait $pid
done

wandb agent jjshoots/CCGE2/4uvx5qez --count 4 &
pids+=($!)
sleep 5
wandb agent jjshoots/CCGE2/4uvx5qez --count 4 &
pids+=($!)
sleep 5
wandb agent jjshoots/CCGE2/4uvx5qez --count 4 &
pids+=($!)
sleep 5
wandb agent jjshoots/CCGE2/4uvx5qez --count 4 &
pids+=($!)
sleep 5

for pid in ${pids[*]}; do
    wait $pid
done

wandb agent jjshoots/CCGE2/aboikqup --count 4 &
pids+=($!)
sleep 5
wandb agent jjshoots/CCGE2/aboikqup --count 4 &
pids+=($!)
sleep 5
wandb agent jjshoots/CCGE2/aboikqup --count 4 &
pids+=($!)
sleep 5
wandb agent jjshoots/CCGE2/aboikqup --count 4 &
pids+=($!)
sleep 5

for pid in ${pids[*]}; do
    wait $pid
done

wandb agent jjshoots/CCGE2/95qoyb9w --count 4 &
pids+=($!)
sleep 5
wandb agent jjshoots/CCGE2/95qoyb9w --count 4 &
pids+=($!)
sleep 5
wandb agent jjshoots/CCGE2/95qoyb9w --count 4 &
pids+=($!)
sleep 5
wandb agent jjshoots/CCGE2/95qoyb9w --count 4 &
pids+=($!)
sleep 5

for pid in ${pids[*]}; do
    wait $pid
done
