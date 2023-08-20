#!/bin/bash

availab_machines=("availab-dl1" "availab-dl2")
total_gpus=2
total_runs=2
runs_per_gpu=1

######################################################################################################
# setup the sweep
######################################################################################################
echo "Generating sweep..."
wandb sweep sweep.yaml &> ./sweep_setup/temp.out

# automatically generate sh file for availab servers
echo "Generating run.sh"
python3 sweep_setup/make_run_sweeps_sh.py $total_gpus $total_runs $runs_per_gpu

# remove the temp file
rm ./sweep_setup/temp.out

# make executable
chmod +x ./sweep_setup/run_availab_sweep.sh

######################################################################################################
# sync the env files
######################################################################################################
echo "Syncing env files..."
declare -a pids=()

for machine in ${availab_machines[@]}; do
  rsync -avr --delete --exclude-from='rsync_ignore_out.txt' ../pyflyt_rail_env/ $machine:~/Sandboxes/pyflyt_rail_env &
  pids+=($!)
done

for pid in ${pids[*]}; do
  wait $pid
done

######################################################################################################
# sync all files out
######################################################################################################
echo "Syncing out..."
declare -a pids=()

for machine in ${availab_machines[@]}; do
  rsync -avr --delete --exclude-from='rsync_ignore_out.txt' ./ $machine:~/Sandboxes/e2SAC/ &
  pids+=($!)
done

for pid in ${pids[*]}; do
  wait $pid
done

######################################################################################################
# run all files
######################################################################################################
declare -a pids=()

for machine in ${availab_machines[@]}; do
  echo "Sending commands to $machine..."
  ssh $machine 'tmux send-keys -t 0 "./sweep_setup/run_availab_sweep.sh" ENTER' &
  pids+=($!)
done

for pid in ${pids[*]}; do
  wait $pid
done
