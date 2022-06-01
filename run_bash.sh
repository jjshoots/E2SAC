#!/bin/bash
source venv/bin/activate

declare -a pids=()

python3 src/main.py --train --wandb --wandb_name='cartpole' --env_name='CartPole-v1' &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name='cartpole' --env_name='CartPole-v1' &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name='cartpole' --env_name='CartPole-v1' &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name='cartpole' --env_name='CartPole-v1' &
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done

