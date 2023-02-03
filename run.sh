#!/bin/bash

# setup the sweep
echo "Generating sweep..."
wandb sweep sweep.yaml &> ./sweep_setup/temp.out

# automatically generate sh file for availab servers
echo "Generating run.sh"
python3 sweep_setup/make_run_sweeps_sh.py

# remove the temp file
rm ./sweep_setup/temp.out

# make executable
chmod +x ./sweep_setup/run_sweeps.sh

# run all files
# ssh arctic-linx 'tmux send-keys -t 0 "git pull origin ccge2_pyflyt" ENTER'
# ssh arctic-linx 'tmux send-keys -t 0 "./sweep_setup/run_sweeps.sh" ENTER'
