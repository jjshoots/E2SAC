source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/carracing_sac_sweep/s78umd9w
pids+=($!)
sleep 10
wandb agent jjshoots/carracing_sac_sweep/s78umd9w
pids+=($!)
sleep 10
wandb agent jjshoots/carracing_sac_sweep/s78umd9w
pids+=($!)
sleep 10
wandb agent jjshoots/carracing_sac_sweep/s78umd9w
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done

# python3 src/main.py --train --wandb --wandb_name='e2SAC_dr_lessent'
# python3 src/mainSAC.py --train --wandb --wandb_name='SAC_dr_lessent'
