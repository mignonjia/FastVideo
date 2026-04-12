#!/bin/bash
#SBATCH --job-name=mg-validate-latent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/home/mhuo/FastVideo-MG/preprocess_mc_output/validate_%j.out
#SBATCH --error=/mnt/home/mhuo/FastVideo-MG/preprocess_mc_output/validate_%j.err

set -e -x
source /opt/conda/etc/profile.d/conda.sh && conda activate mhuo-fv
export PYTHONPATH="/mnt/home/mhuo/FastVideo-MG:${PYTHONPATH:-}"
python3 /mnt/home/mhuo/FastVideo-MG/tests/validate_latent.py
