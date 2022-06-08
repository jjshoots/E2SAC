#!/bin/fish
if [ "$hostname" = "arctic-linx" ]
  sleep 2
  ssh availab-dl1 'tmux send-keys -t 0 "./run_availab.sh" ENTER'
  ssh availab-dl2 'tmux send-keys -t 0 "./run_availab.sh" ENTER'
  ssh availab-dl3 'tmux send-keys -t 0 "./run_availab.sh" ENTER'
  ssh availab-dl4 'tmux send-keys -t 0 "./run_availab.sh" ENTER'
else if [ "$hostname" = "availab-dl1" ]
  source venv/bin/activate.fish
  # python3 src/main.py --train --wandb --wandb_name='lunar_lander' &
else if [ "$hostname" = "availab-dl2" ]
  source venv/bin/activate.fish
  # python3 src/main.py --train --wandb --wandb_name='lunar_lander' &
end
