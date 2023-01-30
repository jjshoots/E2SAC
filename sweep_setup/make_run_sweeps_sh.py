_RUNS_PER_GPU = 3

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
pip3 install -e .
pip3 uninstall gymnasium-robotics -y
pip3 install -r requirements.txt -U
wingman-compress-weights

declare -a pids=()
"""

availab_run_line = f"wandb agent jjshoots/CCGE2/{sweep_id} --count {int(50/_RUNS_PER_GPU/8)} & "
dream_prophet_run_line_0 = f"CUDA_VISIBLE_DEVICES=0 wandb agent jjshoots/CCGE2/{sweep_id} --count {int(50/_RUNS_PER_GPU/8)} & "
dream_prophet_run_line_1 = f"CUDA_VISIBLE_DEVICES=1 wandb agent jjshoots/CCGE2/{sweep_id} --count {int(50/_RUNS_PER_GPU/8)} & "

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

# write for dream prophet
with open("./sweep_setup/run_dream_prophet_sweep.sh", "w") as f:
    # shebangs
    f.write(top_lines)

    # contents
    for _ in range(_RUNS_PER_GPU):
        f.write(dream_prophet_run_line_0)
        f.write(joining_lines)
        f.write(dream_prophet_run_line_1)
        f.write(joining_lines)

    # closing
    f.write(end_lines)
