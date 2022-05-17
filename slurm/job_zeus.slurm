#!/bin/bash

## job name
#SBATCH --job-name="carracing"

## time needed to run job before terminate
#SBATCH --time=72:00:00
## number of CPUs
#SBATCH --ntasks-per-node=4
## amount of RAM in MB
#SBATCH --mem=16000
## add in GPUs
#SBATCH --gres=gpu:K80:1

# Lets generate hosts file (in case we need mpi):
WORKDIR=${SLURM_SUBMIT_DIR}
HOSTS=`srun /bin/hostname|sort`
echo ${HOSTS} | sed 's/ /n/g'>${WORKDIR}/slurm/hostsfile.txt

# Nr of CPU's in total we reserved
NPROCS=$(( $SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE ))

source ~/.bashrc
source venv/bin/activate
wandb agent jjshoots/carracing_sac_sweep/bp3bcsm9 --count 1
