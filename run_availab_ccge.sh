#!/bin/bash

source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/CCGE2/fscr3ka0 --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/fscr3ka0 --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/fscr3ka0 --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/fscr3ka0 --count 6 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

wandb agent jjshoots/CCGE2/5ydeyszy --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/5ydeyszy --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/5ydeyszy --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/5ydeyszy --count 6 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

wandb agent jjshoots/CCGE2/3rk3ayzb --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/3rk3ayzb --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/3rk3ayzb --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/3rk3ayzb --count 6 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

wandb agent jjshoots/CCGE2/4de8u2d3 --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/4de8u2d3 --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/4de8u2d3 --count 6 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/4de8u2d3 --count 6 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
