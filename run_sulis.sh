source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/UA3SAC_gym/s73mfuy2 --count 1 &
pids+=($!--count 1)
wandb agent jjshoots/UA3SAC_gym/s73mfuy2 --count 1 &
pids+=($!)
wandb agent jjshoots/UA3SAC_gym/s73mfuy2 --count 1 &
pids+=($!)
wandb agent jjshoots/UA3SAC_gym/s73mfuy2 --count 1 &
pids+=($!)
wandb agent jjshoots/UA3SAC_gym/s73mfuy2 --count 1 &
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done
