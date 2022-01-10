#!/bin/fish
if [ "$hostname" = "arctic-linx" ]
  echo hello there!
else if [ "$hostname" = "availab-dl1" ]
  source venv/bin/activate.fish
  xvfb-run -s '-screen 1 1400x900x24' python3 src/mainSAC.py --train --wandb --name='sac'
else if [ "$hostname" = "availab-dl2" ]
  source venv/bin/activate.fish
  xvfb-run -s '-screen 1 1400x900x24' python3 src/main.py --train --wandb --name='e2sac'
end
