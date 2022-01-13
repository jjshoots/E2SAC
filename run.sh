#!/bin/fish
if [ "$hostname" = "arctic-linx" ]
  echo hello there!
else if [ "$hostname" = "availab-dl1" ]
  source venv/bin/activate.fish
  # xvfb-run -s '-screen 1 1400x900x24' python3 src/mainSAC.py --train --wandb --name='sac'
  xvfb-run -s '-screen 1 1400x900x24' wandb agent jjshoots/e2SAC/7q91j5gt
else if [ "$hostname" = "availab-dl2" ]
  source venv/bin/activate.fish
  # xvfb-run -s '-screen 1 1400x900x24' python3 src/mainSAC.py --train --wandb --name='sac'
  xvfb-run -s '-screen 1 1400x900x24' wandb agent jjshoots/e2SAC/7q91j5gt
end
