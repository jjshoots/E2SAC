#!/bin/bash

source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/CCGE2/s34yh4a6 --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/s34yh4a6 --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/s34yh4a6 --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/s34yh4a6 --count 6 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

wandb agent jjshoots/CCGE2/62jvni7q --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/62jvni7q --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/62jvni7q --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/62jvni7q --count 6 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

wandb agent jjshoots/CCGE2/b0r4xjqu --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/b0r4xjqu --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/b0r4xjqu --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/b0r4xjqu --count 6 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

wandb agent jjshoots/CCGE2/po10jfqp --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/po10jfqp --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/po10jfqp --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/po10jfqp --count 6 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
