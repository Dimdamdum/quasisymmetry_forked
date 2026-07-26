#!/bin/bash

python cluster_numbers_metrics.py h2o sto3g 1.0 'variance' \
    --cluster-matrix '[[1,1,0,0,0,0,0],[0,0,1,1,0,0,0]]' --max-transfers 2 --bond-angle 104.5

python cluster_numbers_metrics.py h2o sto3g 1.0 'eval_eq'\
    --cluster-matrix '[[1,1,0,0,0,0,0],[0,0,1,1,0,0,0]]' --max-transfers 2 --bond-angle 104.5

python cluster_numbers_metrics.py h2o sto3g 1.0 'extremality'\
    --cluster-matrix '[[1,1,0,0,0,0,0],[0,0,1,1,0,0,0]]' --max-transfers 2 --bond-angle 104.5