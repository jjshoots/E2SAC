#!/bin/fish
if [ "$hostname" = "arctic-linx" ]
  sleep 2
  ssh availab-dl1 'tmux send-keys -t 0 "./run.sh" ENTER'
  sleep 2
  ssh availab-dl1 'tmux send-keys -t 1 "./run.sh" ENTER'
  sleep 2
  ssh availab-dl1 'tmux send-keys -t 2 "./run.sh" ENTER'
  sleep 2
  ssh availab-dl1 'tmux send-keys -t 3 "./run.sh" ENTER'
  sleep 2
  # ssh availab-dl2 'tmux send-keys -t 0 "./run.sh" ENTER'
  # sleep 2
  # ssh availab-dl2 'tmux send-keys -t 1 "./run.sh" ENTER'
  # sleep 2
  # ssh availab-dl2 'tmux send-keys -t 2 "./run.sh" ENTER'
  # sleep 2
  # ssh availab-dl2 'tmux send-keys -t 3 "./run.sh" ENTER'
  # sleep 2
else if [ "$hostname" = "availab-dl1" ]
  source venv/bin/activate.fish
  # python3 src/mainSAC.py --train --env_name='AntPyBulletEnv-v0' --wandb --wandb_name='SAC_ant' --notes='ant'
  python3 src/mainSAC.py --train --env_name='HopperPyBulletEnv-v0' --wandb --wandb_name='SAC_hopper' --notes='hopper'
  python3 src/mainSAC.py --train --env_name='HumanoidPyBulletEnv-v0' --wandb --wandb_name='SAC_humanoid' --notes='humanoid'
  python3 src/mainSAC.py --train --env_name='HalfCheetahPyBulletEnv-v0' --wandb --wandb_name='SAC_half_cheetah' --notes='half_cheetah'
else if [ "$hostname" = "availab-dl2" ]
  source venv/bin/activate.fish
  python3 src/mainSAC.py --train --env_name='AntPyBulletEnv-v0' --wandb --wandb_name='SAC_ant' --notes='ant'
  python3 src/mainSAC.py --train --env_name='AntPyBulletEnv-v0' --wandb --wandb_name='SAC_ant' --notes='ant'
  python3 src/mainSAC.py --train --env_name='HopperPyBulletEnv-v0' --wandb --wandb_name='SAC_hopper' --notes='hopper'
  python3 src/mainSAC.py --train --env_name='HopperPyBulletEnv-v0' --wandb --wandb_name='SAC_hopper' --notes='hopper'
  python3 src/mainSAC.py --train --env_name='HumanoidPyBulletEnv-v0' --wandb --wandb_name='SAC_humanoid' --notes='humanoid'
  python3 src/mainSAC.py --train --env_name='HumanoidPyBulletEnv-v0' --wandb --wandb_name='SAC_humanoid' --notes='humanoid'
  python3 src/mainSAC.py --train --env_name='HalfCheetahPyBulletEnv-v0' --wandb --wandb_name='SAC_half_cheetah' --notes='half_cheetah'
  python3 src/mainSAC.py --train --env_name='HalfCheetahPyBulletEnv-v0' --wandb --wandb_name='SAC_half_cheetah' --notes='half_cheetah'
end


# xvfb-run -s '-screen 4 1400x900x24' python3 src/mainSAC.py --train --env_name='HalfCheetahPyBulletEnv-v0' --wandb --wandb_name='SAC_half_cheetah' --notes='half_cheetah'
