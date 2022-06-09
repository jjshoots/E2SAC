#!/bin/bash
source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/carracing_sweep2/ah9k62um --count 1 &
pids+=($!)

wandb agent jjshoots/carracing_sweep2/ah9k62um --count 1 &
pids+=($!)

wandb agent jjshoots/carracing_sweep2/h6tq8uag --count 1 &
pids+=($!)

wandb agent jjshoots/carracing_sweep2/h6tq8uag --count 1 &
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done

# python3 src/main.py --train --wandb --wandb_name='e2SAC_dr_lessent'
# python3 src/mainSAC.py --train --wandb --wandb_name='SAC_dr_lessent'
