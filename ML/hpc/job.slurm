#!/bin/bash
# One stage of ML/hpc/run_jobs.sh as a Slurm job.  Submit from the repo root:
#
#     sbatch --job-name=cab-pilot  ML/hpc/job.slurm pilot
#     sbatch --job-name=cab-b1     ML/hpc/job.slurm b1
#     sbatch --job-name=cab-dagger ML/hpc/job.slurm dagger
#
# Add the partition / account lines your cluster requires, and change the
# module name and environment path below to yours.
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=ML/logs/slurm_%x_%j.out

module load gurobi 2>/dev/null || echo "no 'gurobi' module: make sure gurobi_cl is on PATH"
source .venv/bin/activate 2>/dev/null || echo "no .venv in the repo root: activate your environment here"
bash ML/hpc/run_jobs.sh "$1"
