#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --account=su404-iftc1-gpu

module purge
module load nvidia/cuda-11.2
module load GCCcore/11.3.0
module load Python/3.10.4

source ~/.bashrc
./sweep_setup/run_sulis_sweep.sh
