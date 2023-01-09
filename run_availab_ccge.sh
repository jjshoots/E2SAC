#!/bin/bash

source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/CCGE2/fcm6kbat --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/fcm6kbat --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/fcm6kbat --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/fcm6kbat --count 3 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

declare -a pids=()

wandb agent jjshoots/CCGE2/w1poh5sj --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/w1poh5sj --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/w1poh5sj --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/w1poh5sj --count 3 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

declare -a pids=()

wandb agent jjshoots/CCGE2/6htikfc6 --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/6htikfc6 --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/6htikfc6 --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/6htikfc6 --count 3 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

declare -a pids=()

wandb agent jjshoots/CCGE2/3qtvevee --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/3qtvevee --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/3qtvevee --count 3 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/3qtvevee --count 3 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
