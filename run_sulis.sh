#!/bin/bash

if [[ $HOSTNAME == *"sulis"* ]]; then
  cd ~/e2SAC/
  for i in {1..5}; do
    sbatch sweep_setup/job_sulis.slurm
    sleep 10
  done
else
  ######################################################################################################
  # setup the sweep
  ######################################################################################################
  echo "Generating sweep..."
  wandb sweep sweep.yaml &> ./sweep_setup/temp.out

  # automatically generate sh file for availab servers
  echo "Generating run.sh"
  python3 sweep_setup/make_run_sulis_sweep_sh.py

  # remove the temp file
  rm ./sweep_setup/temp.out

  # make executable
  chmod +x ./sweep_setup/run_sulis_sweep.sh

  ######################################################################################################
  # sync all files out
  ######################################################################################################
  echo "Syncing out..."
  rsync -avr --delete --exclude-from='rsync_ignore_out.txt' ./ sulis:~/e2SAC/

  ######################################################################################################
  # run all files on sulis
  ######################################################################################################
  echo "Please manually run run_sulis.sh on sulis"
fi
