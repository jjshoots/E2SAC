#!/bin/bash
source venv/bin/activate

declare -a pids=()

python3 src/main.py --train --wandb --wandb_name='e2SAC_rand' &
# python3 src/mainSAC.py --train --wandb --wandb_name='SAC' &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done

# python3 src/main.py --train --wandb --wandb_name='e2SAC_dr_lessent'
# python3 src/mainSAC.py --train --wandb --wandb_name='SAC_dr_lessent'
