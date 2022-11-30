#!/bin/bash

source venv/bin/activate
pip3 install -r requirements.txt -U
wingman-compress-weights

declare -a pids=()
wandb agent jjshoots/CCGE2/ly5oeqm4 --count 8 & 
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/ly5oeqm4 --count 8 & 
pids+=($!)
sleep 10
wandb agent jjshoots/CCGE2/ly5oeqm4 --count 8 & 
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
