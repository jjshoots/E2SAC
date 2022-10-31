#!/bin/bash

source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/CCGE2/dp4byhg8 --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/dp4byhg8 --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/dp4byhg8 --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/dp4byhg8 --count 6 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

wandb agent jjshoots/CCGE2/o61ibecs --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/o61ibecs --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/o61ibecs --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/o61ibecs --count 6 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

wandb agent jjshoots/CCGE2/n6ufn853 --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/n6ufn853 --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/n6ufn853 --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/n6ufn853 --count 6 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
