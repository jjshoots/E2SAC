#!/bin/bash

source venv/bin/activate

wingman-compress-weights

declare -a pids=()

python3 src/generate_oracle.py --wandb --wandb_name="AdroitHandRelocateOracle" --train --env_name="AdroitHandRelocate-v0" &
pids+=($!)
sleep 10
python3 src/generate_oracle.py --wandb --wandb_name="AdroitHandHammerOracle" --train --env_name="AdroitHandHammer-v0" &
pids+=($!)
sleep 10
python3 src/generate_oracle.py --wandb --wandb_name="AdroitHandDoorOracle" --train --env_name="AdroitHandDoor-v0" &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
