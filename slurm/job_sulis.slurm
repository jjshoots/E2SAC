#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --account=su003-iftc1

module purge
module load GCC/10.2.0  CUDA/11.1.1  OpenMPI/4.0.5
module load PyTorch/1.9.0
module load torchvision/0.10.0-PyTorch-1.9.0

source ~/.bashrc
source /venv/bin/activate

NJOBS=`squeue -h --node=$(hostname -s) --user=$SLURM_JOB_USER | wc -l`
screen="-screen 0 1400x900x24"
if [[ $NJOBS -gt 1 ]]; then
  screen="-screen 1 1400x900x24";
fi

./run_sulis.sh
