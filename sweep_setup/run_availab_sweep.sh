#!/bin/bash

# this file has been automatically generated, do not edit manually

source venv/bin/activate

pip3 uninstall pyflyt_dogfight -y
pip3 install git+https://github.com/jjshoots/pyflyt_dogfight@master --no-cache-dir -U

wingman-compress-weights

declare -a pids=()
wandb agent jjshoots/sac_dogfight/lzekb4tt --count 1 & 
pids+=($!)
sleep 10
wandb agent jjshoots/sac_dogfight/lzekb4tt --count 1 & 
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
