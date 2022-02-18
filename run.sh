#!/bin/fish
if [ "$hostname" = "arctic-linx" ]
  ssh availab-dl1 'tmux send-keys -t 0 "./run.sh" ENTER'
  ssh availab-dl2 'tmux send-keys -t 0 "./run.sh" ENTER'
else if [ "$hostname" = "availab-dl1" ]
  source venv/bin/activate.fish
  # xvfb-run -s '-screen 1 1400x900x24' python3 src/mainSAC.py --train --wandb --name='SAC'
  xvfb-run -s '-screen 1 1400x900x24' python3 src/main.py --train --wandb --name='e2SAC_implicit' --notes='implicit uncertainty with split entropy, target_q from infer'
  # xvfb-run -s '-screen 1 1400x900x24' wandb agent jjshoots/e2SAC/j4au56kz
else if [ "$hostname" = "availab-dl2" ]
  source venv/bin/activate.fish
  # xvfb-run -s '-screen 1 1400x900x24' python3 src/mainSAC.py --train --wandb --name='SAC'
  xvfb-run -s '-screen 1 1400x900x24' python3 src/main.py --train --wandb --name='e2SAC_implicit' --notes='implicit uncertainty with split entropy, target_q from infer, one critic update step'
  # xvfb-run -s '-screen 1 1400x900x24' wandb agent jjshoots/e2SAC/j4au56kz
end
