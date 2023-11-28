#!/bin/bash -l
#SBATCH --job-name=GCVAE
#SBATCH --mail-user=ifeanyi.ezukwoke@emse.fr
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=75G
#SBATCH --time=7-00:00:00
#SBATCH --partition=audace2018
ulimit -l unlimited
unset SLURM_GTIDS

echo -----------------------------------------------
echo SLURM_NNODES: $SLURM_NNODES
echo SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST
echo SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR
echo SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST
echo SLURM_JOB_ID: $SLURM_JOB_ID
echo SLURM_JOB_NAME: $SLURM_JOB_NAME
echo SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION
echo SLURM_NTASKS: $SLURM_NTASKS
echo SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE
echo SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE
echo -----------------------------------------------
echo Run program...
#conda deactivate
#module purge
#module load gcc/8.1.0
#module load python/3.7.1
source ~/meso-env/env.sh
python ./REMEDS/gcvae_fa?py
echo -----------------------------------------------
