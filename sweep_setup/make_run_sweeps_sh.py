import sys

_TOTAL_GPUS = int(sys.argv[1])
_TOTAL_RUNS = int(sys.argv[2])
_RUNS_PER_GPU = int(sys.argv[3])

# read the lines and grab the url
sweep_id = ""
project_id = ""
with open("./sweep_setup/temp.out", "r") as f:
    # read lines
    lines = f.readlines()

    # find the url
    for line in lines:
        if "Run sweep agent with:" in line:
            sweep_id = line.split("/")[-1]
            project_id = line.split("/")[-2]
            sweep_id = sweep_id.replace("\n", "")

top_lines = """#!/bin/bash

# this file has been automatically generated, do not edit manually

source venv/bin/activate

pip3 install pyflyt -U

pip3 uninstall pyflyt_dogfight -y
pip3 install git+https://github.com/jjshoots/pyflyt_dogfight --no-cache-dir -U

wingman-compress-weights

declare -a pids=()
"""

availab_run_line = f"wandb agent jjshoots/{project_id}/{sweep_id} --count {round(_TOTAL_RUNS/_RUNS_PER_GPU/_TOTAL_GPUS)} & "

joining_lines = """
pids+=($!)
sleep 10
"""

end_lines = """
for pid in ${pids[*]}; do
    wait $pid
done
"""

# write for availab machines
with open("./sweep_setup/run_availab_sweep.sh", "w") as f:
    # shebangs
    f.write(top_lines)

    # contents
    for _ in range(_RUNS_PER_GPU):
        f.write(availab_run_line)
        f.write(joining_lines)

    # closing
    f.write(end_lines)
