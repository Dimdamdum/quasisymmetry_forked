#!/bin/bash

python cluster_numbers_metrics.py h2o 6-31g 1.0 \
    --cluster-matrix '[[1,1,0,0,0,0,0,1,0,1,0,0,0],[0,0,1,1,0,0,0,0,1,0,1,0,0],[0,0,0,0,1,1,0,0,0,0,0,0,1]]' --max-transfers 2 --bond-angle 104

python cluster_numbers_metrics.py h2o 6-31g 1.0 \
    --cluster-matrix '[[1,0,1,0,0,0,0,0,1,0,1,0,0],[0,1,0,1,0,0,0,1,0,1,0,0,0]]' --max-transfers 2 --bond-angle 104

python cluster_numbers_metrics.py h2o 6-31g 1.0 \
    --cluster-matrix '[[1,1,1,1,0,0,0,0,1,0,1,0,0]]' --max-transfers 2 --bond-angle 104

python cluster_numbers_metrics.py h2o 6-31g 1.0 \
    --cluster-matrix '[[1,1,1,1,0,0,0,0,1,0,1,0,0]]' --max-transfers 3 --bond-angle 104


python cluster_numbers_metrics.py h2o 6-31g 1.0 \
    --cluster-matrix '[[1,1,0,0,0,0,0,1,0,1,0,0,0],[0,0,1,1,0,0,0,0,1,0,1,0,0],[0,0,0,0,1,1,0,0,0,0,0,0,1]]' --max-transfers 2 --bond-angle 50

python cluster_numbers_metrics.py h2o 6-31g 1.0 \
    --cluster-matrix '[[1,0,1,0,0,0,0,0,1,0,1,0,0],[0,1,0,1,0,0,0,1,0,1,0,0,0]]' --max-transfers 2 --bond-angle 50

python cluster_numbers_metrics.py h2o 6-31g 1.0 \
    --cluster-matrix '[[1,1,1,1,0,0,0,0,1,0,1,0,0]]' --max-transfers 2 --bond-angle 50

python cluster_numbers_metrics.py h2o 6-31g 1.0 \
    --cluster-matrix '[[1,1,1,1,0,0,0,0,1,0,1,0,0]]' --max-transfers 3 --bond-angle 50