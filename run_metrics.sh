#! /bin/bash

#your folder name goes here; assuming all of them are jsons that metrics.py wants
for file in ./oo_h4/*;
do 
  echo "Processing: $file"
  python metrics.py $file
done
