source venv/bin/activate

declare -a pids=()

python3 src/main.py --train --wandb --wandb_name='ant_cf10.0' --env_name=AntPyBulletEnv-v0 --confidence_lambda=10.0 &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name='ant_cf10.0' --env_name=AntPyBulletEnv-v0 --confidence_lambda=10.0 &
pids+=($!)
sleep 20
python3 src/main.py --train --wandb --wandb_name='ant_cf20.0' --env_name=AntPyBulletEnv-v0 --confidence_lambda=20.0 &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name='ant_cf20.0' --env_name=AntPyBulletEnv-v0 --confidence_lambda=20.0 &
pids+=($!)
sleep 20
python3 src/main.py --train --wandb --wandb_name='ant_cf50.0' --env_name=AntPyBulletEnv-v0 --confidence_lambda=50.0 &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name='ant_cf50.0' --env_name=AntPyBulletEnv-v0 --confidence_lambda=50.0 &
pids+=($!)
sleep 20
python3 src/main.py --train --wandb --wandb_name='ant_cf100.0' --env_name=AntPyBulletEnv-v0 --confidence_lambda=100.0 &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name='ant_cf100.0' --env_name=AntPyBulletEnv-v0 --confidence_lambda=100.0 &
pids+=($!)
sleep 20
python3 src/main.py --train --wandb --wandb_name='ant_cf200.0' --env_name=AntPyBulletEnv-v0 --confidence_lambda=200.0 &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name='ant_cf200.0' --env_name=AntPyBulletEnv-v0 --confidence_lambda=200.0 &
pids+=($!)
sleep 20
python3 src/main.py --train --wandb --wandb_name='ant_cf500.0' --env_name=AntPyBulletEnv-v0 --confidence_lambda=500.0 &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name='ant_cf500.0' --env_name=AntPyBulletEnv-v0 --confidence_lambda=500.0 &
pids+=($!)
sleep 20
python3 src/main.py --train --wandb --wandb_name='ant_cf1000.0' --env_name=AntPyBulletEnv-v0 --confidence_lambda=1000.0 &
pids+=($!)
python3 src/main.py --train --wandb --wandb_name='ant_cf1000.0' --env_name=AntPyBulletEnv-v0 --confidence_lambda=1000.0 &
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done

