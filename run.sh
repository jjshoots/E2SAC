#!/bin/fish
if [ "$hostname" = "arctic-linx" ]
  ssh availab-dl1 'tmux send-keys -t 0 "./run.sh" ENTER'
  ssh availab-dl1 'tmux send-keys -t 1 "./run.sh" ENTER'
  ssh availab-dl1 'tmux send-keys -t 2 "./run.sh" ENTER'
  ssh availab-dl1 'tmux send-keys -t 3 "./run.sh" ENTER'
  ssh availab-dl1 'tmux send-keys -t 4 "./run.sh" ENTER'
  ssh availab-dl2 'tmux send-keys -t 0 "./run.sh" ENTER'
  ssh availab-dl2 'tmux send-keys -t 1 "./run.sh" ENTER'
  ssh availab-dl2 'tmux send-keys -t 2 "./run.sh" ENTER'
  ssh availab-dl2 'tmux send-keys -t 3 "./run.sh" ENTER'
  ssh availab-dl2 'tmux send-keys -t 4 "./run.sh" ENTER'
else if [ "$hostname" = "availab-dl1" ]
  source venv/bin/activate.fish
  xvfb-run -s '-screen 1 1400x900x24' python3 src/mainSAC.py --train --env_name='AntPyBulletEnv-v0' --wandb --wandb_name='SAC_ant' --notes='ant'
  xvfb-run -s '-screen 2 1400x900x24' python3 src/mainSAC.py --train --env_name='HopperPyBulletEnv-v0' --wandb --wandb_name='SAC_hopper' --notes='hopper'
  xvfb-run -s '-screen 3 1400x900x24' python3 src/mainSAC.py --train --env_name='HumanoidPyBulletEnv-v0' --wandb --wandb_name='SAC_humanoid' --notes='humanoid'
  xvfb-run -s '-screen 4 1400x900x24' python3 src/mainSAC.py --train --env_name='HalfCheetahPyBulletEnv-v0' --wandb --wandb_name='SAC_half_cheetah' --notes='half_cheetah'
else if [ "$hostname" = "availab-dl2" ]
  source venv/bin/activate.fish
  xvfb-run -s '-screen 1 1400x900x24' python3 src/mainSAC.py --train --env_name='AntPyBulletEnv-v0' --wandb --wandb_name='SAC_ant' --notes='ant'
  xvfb-run -s '-screen 2 1400x900x24' python3 src/mainSAC.py --train --env_name='HopperPyBulletEnv-v0' --wandb --wandb_name='SAC_hopper' --notes='hopper'
  xvfb-run -s '-screen 3 1400x900x24' python3 src/mainSAC.py --train --env_name='HumanoidPyBulletEnv-v0' --wandb --wandb_name='SAC_humanoid' --notes='humanoid'
  xvfb-run -s '-screen 4 1400x900x24' python3 src/mainSAC.py --train --env_name='HalfCheetahPyBulletEnv-v0' --wandb --wandb_name='SAC_half_cheetah' --notes='half_cheetah'
end
