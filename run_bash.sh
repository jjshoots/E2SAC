source venv/bin/activate

declare -a pids=()

python3 src/main.py --train --wandb --wandb_name='lunarlander' &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name='lunarlander' &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name='lunarlander' &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name='lunarlander' &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name='lunarlander' &
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done

