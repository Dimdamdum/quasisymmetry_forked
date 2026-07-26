#!/bin/bash
#SBATCH --job-name=metrics
#SBATCH --output=cluster_metrics_%j.out
#SBATCH --error=cluster_metrics_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=normal

conda activate quasisym

python cluster_numbers_metrics.py h2o sto3g 1.0 'variance' --cluster-matrix '[[1,1,0,0,0,0,0],[0,0,1,1,0,0,0]]' --max-transfers 2 --bond-angle 104.5
python cluster_numbers_metrics.py h2o sto3g 1.0 'eval_eq' --cluster-matrix '[[1,1,0,0,0,0,0],[0,0,1,1,0,0,0]]' --max-transfers 2 --bond-angle 104.5
python cluster_numbers_metrics.py h2o sto3g 1.0 'extremality' --cluster-matrix '[[1,1,0,0,0,0,0],[0,0,1,1,0,0,0]]' --max-transfers 2 --bond-angle 104.5