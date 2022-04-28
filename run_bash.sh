source venv/bin/activate

declare -a pids=()

python3 src/main.py --train --wandb --wandb_name='cartpole' &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name='cartpole' &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name='cartpole' &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name='cartpole' &
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done

