#!/bin/bash
declare -a pids=()

rsync -avr --progress availab-dl1:~/Sandboxes/e2SAC/weights/ ./weights &
pids+=($!)
rsync -avr --progress availab-dl2:~/Sandboxes/e2SAC/weights/ ./weights &
pids+=($!)

for pid in ${pids[*]}; do
  wait $pid
done
