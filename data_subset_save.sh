#!/usr/bin/env bash


#SBATCH -o slurm-%j.out
#SBATCH -p all
#SBATCH --output=logs/%u_%x_%A.out
#SBATCH --mail-type=END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=50G
#SBATCH --time=00:10:00 
#SBATCH --job-name=test_data

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

echo "With access to cpu id(s): "
cat /proc/$$/status | grep Cpus_allowed_list

echo "Array Allocation Number: $SLURM_ARRAY_JOB_ID"
echo "Array Index: $SLURM_ARRAY_TASK_ID"

module load anacondapy
source activate towers_glm_hmm

python data_subset_save.py 
source deactivate 