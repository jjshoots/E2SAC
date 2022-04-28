source venv/bin/activate

declare -a pids=()

python3 src/main.py --train --wandb --wandb_name='ant_cf0.01' --env_name=AntPyBulletEnv-v0 --confidence_lambda=0.01 &
pids+=($!)
sleep 1
python3 src/main.py --train --wandb --wandb_name='ant_cf0.05' --env_name=AntPyBulletEnv-v0 --confidence_lambda=0.05 &
pids+=($!)
sleep 1
python3 src/main.py --train --wandb --wandb_name='ant_cf0.1' --env_name=AntPyBulletEnv-v0 --confidence_lambda=0.1 &
pids+=($!)
sleep 1
python3 src/main.py --train --wandb --wandb_name='ant_cf0.5' --env_name=AntPyBulletEnv-v0 --confidence_lambda=0.5 &
pids+=($!)
sleep 1
python3 src/main.py --train --wandb --wandb_name='ant_cf1.0' --env_name=AntPyBulletEnv-v0 --confidence_lambda=1.0 &
pids+=($!)
sleep 1
python3 src/main.py --train --wandb --wandb_name='ant_cf0.01' --env_name=AntPyBulletEnv-v0 --confidence_lambda=0.01 &
pids+=($!)
sleep 1
python3 src/main.py --train --wandb --wandb_name='ant_cf0.05' --env_name=AntPyBulletEnv-v0 --confidence_lambda=0.05 &
pids+=($!)
sleep 1
python3 src/main.py --train --wandb --wandb_name='ant_cf0.1' --env_name=AntPyBulletEnv-v0 --confidence_lambda=0.1 &
pids+=($!)
sleep 1
python3 src/main.py --train --wandb --wandb_name='ant_cf0.5' --env_name=AntPyBulletEnv-v0 --confidence_lambda=0.5 &
pids+=($!)
sleep 1
python3 src/main.py --train --wandb --wandb_name='ant_cf1.0' --env_name=AntPyBulletEnv-v0 --confidence_lambda=1.0 &
pids+=($!)
sleep 1

for pid in ${pids[*]}; do
    wait $pid
done

