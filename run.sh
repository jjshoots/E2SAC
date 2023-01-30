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
  chmod +x ./sweep_setup/run_availab_sweep.sh
  chmod +x ./sweep_setup/run_dream_prophet_sweep.sh

  # sync all files out
  echo "Syncing out..."
  declare -a pids=()
  rsync -avr --progress --stats --delete --exclude-from='rsync_ignore_out.txt' ./ availab-dl1:~/Sandboxes/e2SAC/ &
  pids+=($!)
  rsync -avr --progress --stats --delete --exclude-from='rsync_ignore_out.txt' ./ availab-dl2:~/Sandboxes/e2SAC/ &
  pids+=($!)
  rsync -avr --progress --stats --delete --exclude-from='rsync_ignore_out.txt' ./ availab-dl3:~/Sandboxes/e2SAC/ &
  pids+=($!)
  rsync -avr --progress --stats --delete --exclude-from='rsync_ignore_out.txt' ./ availab-dl4:~/Sandboxes/e2SAC/ &
  pids+=($!)
  rsync -avr --progress --stats --delete --exclude-from='rsync_ignore_out.txt' ./ dream:~/e2SAC/ &
  pids+=($!)
  rsync -avr --progress --stats --delete --exclude-from='rsync_ignore_out.txt' ./ prophet:~/e2SAC/ &
  pids+=($!)
  for pid in ${pids[*]}; do
      wait $pid
  done

  # run all files on availab
  declare -a pids=()

  ssh availab-dl1 'tmux send-keys -t 0 "./sweep_setup/run_availab_sweep.sh" ENTER'
  echo "Sent commands to dl1."
  pids+=($!)
  ssh availab-dl2 'tmux send-keys -t 0 "./sweep_setup/run_availab_sweep.sh" ENTER'
  echo "Sent commands to dl2."
  pids+=($!)
  ssh availab-dl3 'tmux send-keys -t 0 "./sweep_setup/run_availab_sweep.sh" ENTER'
  echo "Sent commands to dl3."
  pids+=($!)
  ssh availab-dl4 'tmux send-keys -t 0 "./sweep_setup/run_availab_sweep.sh" ENTER'
  echo "Sent commands to dl4."
  pids+=($!)

  # run all files on dream prophet
  ssh dream 'tmux send-keys -t 0 "./sweep_setup/run_dream_prophet_sweep.sh" ENTER'
  echo "Sent commands to dl3."
  pids+=($!)
  ssh prophet 'tmux send-keys -t 0 "./sweep_setup/run_dream_prophet_sweep.sh" ENTER'
  echo "Sent commands to dl4."
  pids+=($!)

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
