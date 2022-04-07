#!/bin/fish
if [ "$hostname" = "arctic-linx" ]
  # ssh availab-dl1 'tmux send-keys -t 0 "./run.sh" ENTER'
  ssh availab-dl2 'tmux send-keys -t 0 "./run.sh" ENTER'
else if [ "$hostname" = "availab-dl1" ]
  source venv/bin/activate.fish
  xvfb-run -s '-screen 1 1400x900x24' python3 src/main.py --train --wandb --wandb_name='e2SAC_final'
  xvfb-run -s '-screen 1 1400x900x24' python3 src/mainSAC.py --train --wandb --wandb_name='SAC_final'
  xvfb-run -s '-screen 1 1400x900x24' python3 src/main.py --train --wandb --wandb_name='e2SAC_final'
  xvfb-run -s '-screen 1 1400x900x24' python3 src/mainSAC.py --train --wandb --wandb_name='SAC_final'
  xvfb-run -s '-screen 1 1400x900x24' python3 src/main.py --train --wandb --wandb_name='e2SAC_final'
  xvfb-run -s '-screen 1 1400x900x24' python3 src/mainSAC.py --train --wandb --wandb_name='SAC_final'
  xvfb-run -s '-screen 1 1400x900x24' python3 src/main.py --train --wandb --wandb_name='e2SAC_final'
  xvfb-run -s '-screen 1 1400x900x24' python3 src/mainSAC.py --train --wandb --wandb_name='SAC_final'
  xvfb-run -s '-screen 1 1400x900x24' python3 src/main.py --train --wandb --wandb_name='e2SAC_final'
  xvfb-run -s '-screen 1 1400x900x24' python3 src/mainSAC.py --train --wandb --wandb_name='SAC_final'
else if [ "$hostname" = "availab-dl2" ]
  source venv/bin/activate.fish
  xvfb-run -s '-screen 1 1400x900x24' python3 src/mainSAC.py --train --wandb --wandb_name='SAC_final'
  xvfb-run -s '-screen 1 1400x900x24' python3 src/main.py --train --wandb --wandb_name='e2SAC_final'
  xvfb-run -s '-screen 1 1400x900x24' python3 src/mainSAC.py --train --wandb --wandb_name='SAC_final'
  xvfb-run -s '-screen 1 1400x900x24' python3 src/main.py --train --wandb --wandb_name='e2SAC_final'
  xvfb-run -s '-screen 1 1400x900x24' python3 src/mainSAC.py --train --wandb --wandb_name='SAC_final'
  xvfb-run -s '-screen 1 1400x900x24' python3 src/main.py --train --wandb --wandb_name='e2SAC_final'
  xvfb-run -s '-screen 1 1400x900x24' python3 src/mainSAC.py --train --wandb --wandb_name='SAC_final'
  xvfb-run -s '-screen 1 1400x900x24' python3 src/main.py --train --wandb --wandb_name='e2SAC_final'
  xvfb-run -s '-screen 1 1400x900x24' python3 src/mainSAC.py --train --wandb --wandb_name='SAC_final'
  xvfb-run -s '-screen 1 1400x900x24' python3 src/main.py --train --wandb --wandb_name='e2SAC_final'
end
