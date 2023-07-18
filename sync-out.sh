#!/bin/bash

availab_machines=("availab-dl1" "availab-dl2" "availab-dl3" "availab-dl4")

######################################################################################################
# sync the env files
######################################################################################################
echo "Syncing env files..."
pids=()

for machine in ${availab_machines[@]}; do
  rsync -avr --delete --exclude-from='rsync_ignore_out.txt' ../pyflyt_rail_env/ $machine:~/Sandboxes/pyflyt_rail_env &
  pids+=($!)
done

for pid in ${pids[*]}; do
  wait $pid
done

######################################################################################################
# sync the run files
######################################################################################################
echo "Syncing run files..."
pids=()

for machine in ${availab_machines[@]}; do
  rsync -avr --delete --exclude-from='rsync_ignore_out.txt' ./ $machine:~/Sandboxes/e2SAC/ &
  pids+=($!)
done

for pid in ${pids[*]}; do
  wait $pid
done
