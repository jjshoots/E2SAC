source venv/bin/activate

declare -a pids=()

wandb agent --count=1 jjshoots/UA3SAC_gym/yp3kgbaf &
pids+=($!)
wandb agent --count=1 jjshoots/UA3SAC_gym/yp3kgbaf &
pids+=($!)
wandb agent --count=1 jjshoots/UA3SAC_gym/yp3kgbaf &
pids+=($!)
wandb agent --count=1 jjshoots/UA3SAC_gym/yp3kgbaf &
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done
