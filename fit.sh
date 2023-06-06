#!/usr/bin/env bash


#SBATCH -o slurm-%j.out
#SBATCH -p all
#SBATCH --output=logs/%u_%x_%A.out
#SBATCH --mail-type=END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=15:00:00
#SBATCH --job-name=hmm_test_fit
#SBATCH --mail-user=yousuf@princeton.edu


echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

echo "With access to cpu id(s): "
cat /proc/$$/status | grep Cpus_allowed_list

echo "Array Allocation Number: $SLURM_ARRAY_JOB_ID"
echo "Array Index: $SLURM_ARRAY_TASK_ID"

module load anacondapy/2022.05
source activate ~/.conda/envs/mybase/envs/towers


python /jukebox/witten/yousuf/ephys_glm_hmm/glm_hmm_fit.py
