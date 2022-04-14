#!/bin/fish
if [ "$hostname" = "arctic-linx" ]
  ssh availab-dl1 'tmux send-keys -t 0 "./run.sh" ENTER'
  # ssh availab-dl2 'tmux send-keys -t 0 "./run.sh" ENTER'
else if [ "$hostname" = "availab-dl1" ]
  source venv/bin/activate.fish
  python3 src/main.py --train --env_name='HopperPyBulletEnv-v0' --wandb --wandb_name='e2SAC_hopper' --notes='hopper'
else if [ "$hostname" = "availab-dl2" ]
  source venv/bin/activate.fish
  python3 src/mainSAC.py --train --env_name='HopperPyBulletEnv-v0' --wandb --wandb_name='SAC_hopper' --notes='hopper'
  # python3 src/main.py --train --env_name='HopperPyBulletEnv-v0' --wandb --wandb_name='e2SAC_hopper' --notes='hopper'
end


# xvfb-run -s '-screen 4 1400x900x24' python3 src/mainSAC.py --train --env_name='HalfCheetahPyBulletEnv-v0' --wandb --wandb_name='SAC_half_cheetah' --notes='half_cheetah'
# python3 src/mainSAC.py --train --env_name='BipedalWalker-v3' --wandb --wandb_name='BipedalWalker-v3-suboptimal'
# python3 src/main.py --train --env_name='AntPyBulletEnv-v0' --wandb --wandb_name='e2SAC_ant' --notes='ant'
# python3 src/main.py --train --env_name='HopperPyBulletEnv-v0' --wandb --wandb_name='e2SAC_hopper' --notes='hopper'
# python3 src/main.py --train --env_name='Walker2DPyBulletEnv-v0' --wandb --wandb_name='e2SAC_walker2d' --notes='walker2d'
# python3 src/main.py --train --env_name='HalfCheetahPyBulletEnv-v0' --wandb --wandb_name='e2SAC_half_cheetah' --notes='half_cheetah'
