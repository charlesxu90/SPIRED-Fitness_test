#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mincpus=30
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --partition=batch
#SBATCH --output=log-%x-%j.out
#SBATCH --error=log-%x-%j.out

module purge
module load gcc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/xux/miniconda3/envs/spired_fitness 

#python run_SPIRED-Stab.py --fasta_file data/example/test.fasta --saved_folder results/example

input_file=$1
echo "input file $input_file"
python run_SPIRED-Stab.py --fasta_file $input_file --saved_folder $input_file.pred 
