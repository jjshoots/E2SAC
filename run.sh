#!/bin/fish
if [ "$hostname" = "arctic-linx" ]
  ssh availab-dl1 'tmux send-keys -t 0 "./run_availab.sh" ENTER'
  ssh availab-dl2 'tmux send-keys -t 0 "./run_availab.sh" ENTER'
else if [ "$hostname" = "availab-dl1" ]
  source venv/bin/activate.fish
  ./run_availab.sh
else if [ "$hostname" = "availab-dl2" ]
  source venv/bin/activate.fish
  ./run_availab.sh
end
