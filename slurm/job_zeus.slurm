#!/bin/bash

## job name
#SBATCH --job-name="carracing"

## time needed to run job before terminate
#SBATCH --time=72:00:00
## number of CPUs
#SBATCH --ntasks-per-node=4
## amount of RAM in MB
#SBATCH --mem=64000
## add in GPUs
#SBATCH --gres=gpu:K80:1

# Lets generate hosts file (in case we need mpi):
WORKDIR=${SLURM_SUBMIT_DIR}
HOSTS=`srun /bin/hostname|sort`
echo ${HOSTS} | sed 's/ /n/g'>${WORKDIR}/slurm/hostsfile.txt

# Nr of CPU's in total we reserved
NPROCS=$(( $SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE ))

source ~/.bashrc

NJOBS=`squeue -h --node=$(hostname -s) --user=$SLURM_JOB_USER | wc -l`
screen="-screen 0 1400x900x24"
if [[ $NJOBS -gt 1 ]]; then
  screen="-screen 1 1400x900x24";
fi

module load python/last
conda activate robocar

xvfb-run -s "$screen" wandb agent --count=1 jjshoots/UA3SAC_gym/yp3kgbaf
