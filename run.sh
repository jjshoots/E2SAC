#!/bin/bash

availab_machines=("availab-dl1" "availab-dl2" "availab-dl3" "availab-dl4")
dream_prophet_machines=("dream" "prophet")
total_gpus=8

if true; then
  ######################################################################################################
  # setup the sweep
  ######################################################################################################
  echo "Generating sweep..."
  wandb sweep sweep.yaml &> ./sweep_setup/temp.out

  # automatically generate sh file for availab servers
  echo "Generating run.sh"
  python3 sweep_setup/make_run_sweeps_sh.py $total_gpus

  # remove the temp file
  rm ./sweep_setup/temp.out

  # make executable
  chmod +x ./sweep_setup/run_availab_sweep.sh
  chmod +x ./sweep_setup/run_dream_prophet_sweep.sh

  ######################################################################################################
  # sync all files out
  ######################################################################################################
  echo "Syncing out..."
  declare -a pids=()

  for machine in ${availab_machines[@]}; do
    rsync -avr --delete --exclude-from='rsync_ignore_out.txt' ./ $machine:~/Sandboxes/e2SAC/ &
    pids+=($!)
  done

  for machine in ${dream_prophet_machines[@]}; do
    rsync -avr --delete --exclude-from='rsync_ignore_out.txt' ./ $machine:~/e2SAC/ &
    pids+=($!)
  done

  for pid in ${pids[*]}; do
      wait $pid
  done

  ######################################################################################################
  # run all files on availab
  ######################################################################################################
  declare -a pids=()

  for machine in ${availab_machines[@]}; do
    ssh $machine 'tmux send-keys -t 0 "./sweep_setup/run_availab_sweep.sh" ENTER' &
    echo "Sent commands to $machine."
    pids+=($!)
  done

  for machine in ${dream_prophet_machines[@]}; do
    ssh $machine 'tmux send-keys -t 0 "./sweep_setup/run_dream_prophet_sweep.sh" ENTER' &
    echo "Sent commands to $machine."
    pids+=($!)
  done

  for pid in ${pids[*]}; do
      wait $pid
  done

else

  # sync all files out
  echo "Syncing out..."
  ./sync-out.sh

  # run all files
  ssh availab-dl1 'tmux send-keys -t 0 "./run_availab.sh" ENTER'
  echo "Sent commands to dl1."
  ssh availab-dl2 'tmux send-keys -t 0 "./run_availab.sh" ENTER'
  echo "Sent commands to dl2."
  ssh availab-dl3 'tmux send-keys -t 0 "./run_availab.sh" ENTER'
  echo "Sent commands to dl3."
  ssh availab-dl4 'tmux send-keys -t 0 "./run_availab.sh" ENTER'
  echo "Sent commands to dl4."

fi
