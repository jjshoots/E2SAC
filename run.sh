#!/bin/fish
if [ "$hostname" = "snow-fox" ]
  ./sync-out.sh
  # ssh availab-dl1 'tmux send-keys -t 0 "./run_availab_ccge.sh" ENTER'
  # ssh availab-dl2 'tmux send-keys -t 0 "./run_availab_ccge.sh" ENTER'
  ssh availab-dl3 'tmux send-keys -t 0 "./run_availab_ccge.sh" ENTER'
  ssh availab-dl4 'tmux send-keys -t 0 "./run_availab_ccge.sh" ENTER'
else if [ "$hostname" = "availab-dl1" ]
  bash ./run_availab.sh
else if [ "$hostname" = "availab-dl2" ]
  bash ./run_availab.sh
else if [ "$hostname" = "dream" ]
  bash ./run_dream_prophet.sh
else if [ "$hostname" = "prophet" ]
  bash ./run_dream_prophet.sh
end
