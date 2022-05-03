source venv/bin/activate

declare -a pids=()

python3 src/main.py --train --wandb --wandb_name="e2SAC_dr_$1" --confidence_lambda=$1 &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name="e2SAC_dr_$1" --confidence_lambda=$1 &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name="e2SAC_dr_$1" --confidence_lambda=$1 &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name="e2SAC_dr_$1" --confidence_lambda=$1 &
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done

# python3 src/main.py --train --wandb --wandb_name='e2SAC_dr_lessent'
# python3 src/mainSAC.py --train --wandb --wandb_name='SAC_dr_lessent'
