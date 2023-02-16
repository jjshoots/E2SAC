#!/bin/bash

# this file has been automatically generated, do not edit manually

source venv/bin/activate
# pip install wandb==0.13.3 --upgrade
# pip3 install -U pip
# pip3 install -e .
# pip3 install -r requirements.txt -U
# wingman-compress-weights
# pip3 uninstall gymnasium-robotics -y
# pip3 install git+https://github.com/jjshoots/Gymnasium-Robotics.git

declare -a pids=()
for i in {1..10}; do
    wandb agent jjshoots/CCGE2/gnrp57gk --count 1 &
    pids+=($!)
    sleep 10
done

for pid in ${pids[*]}; do
    wait $pid
done
