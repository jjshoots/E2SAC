_RUNS_PER_GPU = 10

# read the lines and grab the url
sweep_id = ""
with open("./sweep_setup/temp.out", "r") as f:
    # read lines
    lines = f.readlines()

    # find the url
    for line in lines:
        if "View sweep at:" in line:
            sweep_id = line.split(" ")[-1]
            sweep_id = line.split("/")[-1]
            sweep_id = sweep_id.replace("\n", "")

top_lines = """#!/bin/bash

# this file has been automatically generated, do not edit manually

source venv/bin/activate
# pip3 install -e .
# pip3 uninstall gymnasium-robotics -y
# pip3 install -r requirements.txt -U
# rm -rf weights/*
# wingman-compress-weights

declare -a pids=()
"""

sulis_run_line = f"wandb agent jjshoots/CCGE2/{sweep_id} --count 1 & "

joining_lines = """
pids+=($!)
sleep 20
"""

end_lines = """
for pid in ${pids[*]}; do
    wait $pid
done
"""

# write for dream prophet
with open("./sweep_setup/run_sulis_sweep.sh", "w") as f:
    # shebangs
    f.write(top_lines)

    # contents
    for _ in range(_RUNS_PER_GPU):
        f.write(sulis_run_line)
        f.write(joining_lines)

    # closing
    f.write(end_lines)
