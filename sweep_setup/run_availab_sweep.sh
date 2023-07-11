#!/bin/bash

# this file has been automatically generated, do not edit manually

source venv/bin/activate
pip3 install -e .
rm -rf weights/*
wingman-compress-weights

declare -a pids=()
wandb agent jjshoots/CCGE2/eq71c16h --count 3 & 
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/eq71c16h --count 3 & 
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/eq71c16h --count 3 & 
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/eq71c16h --count 3 & 
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
