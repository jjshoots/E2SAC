#!/bin/bash

source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/CCGE2/zgq81g05 --count 2 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/zgq81g05 --count 2 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/zgq81g05 --count 2 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/zgq81g05 --count 2 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

wandb agent jjshoots/CCGE2/4uvx5qez --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/4uvx5qez --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/4uvx5qez --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/4uvx5qez --count 3 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

wandb agent jjshoots/CCGE2/aboikqup --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/aboikqup --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/aboikqup --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/aboikqup --count 3 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

# wandb agent jjshoots/CCGE2/95qoyb9w --count 3 &
# pids+=($!)
# sleep 10
# wandb agent jjshoots/CCGE2/95qoyb9w --count 3 &
# pids+=($!)
# sleep 10
# wandb agent jjshoots/CCGE2/95qoyb9w --count 3 &
# pids+=($!)
# sleep 10
# wandb agent jjshoots/CCGE2/95qoyb9w --count 3 &
# pids+=($!)
# sleep 10

# for pid in ${pids[*]}; do
#     wait $pid
# done
