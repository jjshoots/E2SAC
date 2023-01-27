#!/bin/bash

source venv/bin/activate

wingman-compress-weights

pip3 uninstall gymnasium-robotics -y
pip3 install -e . -r requirements.txt -U

declare -a pids=()

wandb agent jjshoots/CCGE2_oracle_search/guq3huro --count 1 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2_oracle_search/guq3huro --count 1 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2_oracle_search/guq3huro --count 1 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2_oracle_search/guq3huro --count 1 &
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2_oracle_search/guq3huro --count 1 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
