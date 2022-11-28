_RUNS_PER_MACHINE = 3

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

source venv/bin/activate
pip3 install -r requirements.txt -U

declare -a pids=()
"""

run_line = f"wandb agent jjshoots/CCGE2/{sweep_id} --count 8 & "

joining_lines = """
pids+=($!)
sleep 10
"""

end_lines = """
for pid in ${pids[*]}; do
    wait $pid
done
"""

# start putting things in a run file
with open("./sweep_setup/run_sweeps.sh", "w") as f:
    # shebangs
    f.write(top_lines)

    # contents
    for _ in range(_RUNS_PER_MACHINE):
        f.write(run_line)
        f.write(joining_lines)

    # closing
    f.write(end_lines)
