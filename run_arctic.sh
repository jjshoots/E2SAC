#!/bin/sh
source venv/bin/activate

python3 src/mainCCGE2.py --train --env_name="Ant-v4" --target_performance=2000 --wandb --wandb_name="Experimental_Ant-v4"
