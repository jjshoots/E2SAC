#!/bin/bash

if true; then

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

  # sync all files out
  echo "Syncing out..."
  ./sync-out.sh

  # run all files
  ssh availab-dl1 'tmux send-keys -t 0 "./sweep_setup/run_sweeps.sh" ENTER'
  echo "Sent commands to dl1."
  ssh availab-dl2 'tmux send-keys -t 0 "./sweep_setup/run_sweeps.sh" ENTER'
  echo "Sent commands to dl2."
  ssh availab-dl3 'tmux send-keys -t 0 "./sweep_setup/run_sweeps.sh" ENTER'
  echo "Sent commands to dl3."
  ssh availab-dl4 'tmux send-keys -t 0 "./sweep_setup/run_sweeps.sh" ENTER'
  echo "Sent commands to dl4."

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
