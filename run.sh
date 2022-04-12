#!/bin/fish
if [ "$hostname" = "arctic-linx" ]
  sleep 2
  ssh availab-dl1 'tmux send-keys -t 0 "./run.sh" ENTER'
  ssh availab-dl2 'tmux send-keys -t 0 "./run.sh" ENTER'
else if [ "$hostname" = "availab-dl1" ]
  source venv/bin/activate.fish
  python3 src/mainESDDQN.py --train --wandb --wandb_name='lunarlander_test'
  python3 src/mainESDDQN.py --train --wandb --wandb_name='lunarlander_test'
else if [ "$hostname" = "availab-dl2" ]
  source venv/bin/activate.fish
  python3 src/mainESDDQN.py --train --wandb --wandb_name='lunarlander_test'
  python3 src/mainESDDQN.py --train --wandb --wandb_name='lunarlander_test'
end
