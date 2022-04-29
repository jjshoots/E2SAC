#!/bin/fish
if [ "$hostname" = "arctic-linx" ]
  sleep 2
  ssh availab-dl1 'tmux send-keys -t 0 "./run.sh" ENTER'
  ssh availab-dl2 'tmux send-keys -t 0 "./run.sh" ENTER'
else if [ "$hostname" = "availab-dl1" ]
  source venv/bin/activate.fish
  declare -a pids=()

  python3 src/main.py --train --wandb --wandb_name='lunar_lander' &
  pids+=($!)
  python3 src/main.py --train --wandb --wandb_name='lunar_lander' &
  pids+=($!)

  for pid in ${pids[*]}; do
      wait $pid
  done
else if [ "$hostname" = "availab-dl2" ]
  source venv/bin/activate.fish
  declare -a pids=()

  python3 src/main.py --train --wandb --wandb_name='lunar_lander' &
  pids+=($!)
  python3 src/main.py --train --wandb --wandb_name='lunar_lander' &
  pids+=($!)

  for pid in ${pids[*]}; do
      wait $pid
  done
end
