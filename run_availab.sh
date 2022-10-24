#!/bin/bash

source venv/bin/activate


declare -a pids=()

wandb agent jjshoots/CCGE2/edxnip5i --count 4 &
pids+=($!)
wandb agent jjshoots/CCGE2/edxnip5i --count 4 &
pids+=($!)
wandb agent jjshoots/CCGE2/edxnip5i --count 4 &
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done

wandb agent jjshoots/CCGE2/484so2vd --count 4 &
pids+=($!)
wandb agent jjshoots/CCGE2/484so2vd --count 4 &
pids+=($!)
wandb agent jjshoots/CCGE2/484so2vd --count 4 &
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done

wandb agent jjshoots/CCGE2/l96fgdt5 --count 4 &
pids+=($!)
wandb agent jjshoots/CCGE2/l96fgdt5 --count 4 &
pids+=($!)
wandb agent jjshoots/CCGE2/l96fgdt5 --count 4 &
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done

wandb agent jjshoots/CCGE2/itosrdor --count 4 &
pids+=($!)
wandb agent jjshoots/CCGE2/itosrdor --count 4 &
pids+=($!)
wandb agent jjshoots/CCGE2/itosrdor --count 4 &
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done
