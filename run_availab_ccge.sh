#!/bin/bash

source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/CCGE2/18p6t19p --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/18p6t19p --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/18p6t19p --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/18p6t19p --count 6 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

wandb agent jjshoots/CCGE2/f9sntmmm --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/f9sntmmm --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/f9sntmmm --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/f9sntmmm --count 6 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

wandb agent jjshoots/CCGE2/5e3vqsse --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/5e3vqsse --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/5e3vqsse --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/5e3vqsse --count 6 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

wandb agent jjshoots/CCGE2/trgcinhi --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/trgcinhi --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/trgcinhi --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/trgcinhi --count 6 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
