source venv/bin/activate

declare -a pids=()

python3 src/main.py --train --wandb --wandb_name='ant_cf0.1' --env_name=AntPyBulletEnv-v0 --confidence_lambda=0.1 &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name='ant_cf0.5' --env_name=AntPyBulletEnv-v0 --confidence_lambda=0.5 &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name='ant_cf1.0' --env_name=AntPyBulletEnv-v0 --confidence_lambda=1.0 &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name='ant_cf2.0' --env_name=AntPyBulletEnv-v0 --confidence_lambda=2.0 &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name='ant_cf5.0' --env_name=AntPyBulletEnv-v0 --confidence_lambda=5.0 &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name='ant_cf10.0' --env_name=AntPyBulletEnv-v0 --confidence_lambda=10.0 &
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done

# python3 src/main.py --train --wandb --wandb_name='e2SAC_dr_lessent'
# python3 src/mainSAC.py --train --wandb --wandb_name='SAC_dr_lessent'
