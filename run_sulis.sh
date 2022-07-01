source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/carracing_discrete/9cnmud6a --count 1 &
pids+=($!)

wandb agent jjshoots/carracing_discrete/9cnmud6a --count 1 &
pids+=($!)

wandb agent jjshoots/carracing_discrete/9cnmud6a --count 1 &
pids+=($!)

wandb agent jjshoots/carracing_discrete/9cnmud6a --count 1 &
pids+=($!)

wandb agent jjshoots/carracing_discrete/9cnmud6a --count 1 &
pids+=($!)

wandb agent jjshoots/carracing_discrete/9cnmud6a --count 1 &
pids+=($!)

wandb agent jjshoots/carracing_discrete/9cnmud6a --count 1 &
pids+=($!)

wandb agent jjshoots/carracing_discrete/9cnmud6a --count 1 &
pids+=($!)

wandb agent jjshoots/carracing_discrete/9cnmud6a --count 1 &
pids+=($!)

wandb agent jjshoots/carracing_discrete/9cnmud6a --count 1 &
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done
