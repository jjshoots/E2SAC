#!/bin/bash

source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/CCGE2/7efxff2l --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/7efxff2l --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/7efxff2l --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/7efxff2l --count 3 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

declare -a pids=()

wandb agent jjshoots/CCGE2/fk1ez9z0 --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/fk1ez9z0 --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/fk1ez9z0 --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/fk1ez9z0 --count 3 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

declare -a pids=()

wandb agent jjshoots/CCGE2/62p2gthd --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/62p2gthd --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/62p2gthd --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/62p2gthd --count 3 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

declare -a pids=()

wandb agent jjshoots/CCGE2/m9ckqm0k --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/m9ckqm0k --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/m9ckqm0k --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/m9ckqm0k --count 3 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
