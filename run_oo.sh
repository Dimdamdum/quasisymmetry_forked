#! /bin/bash

h="hamiltonians/H4/H4_linear_d1.0890.chk"
parity_matrices=( "hamiltonians/H4/parity_4_sens.txt" "hamiltonians/H4/LNE_2_out_of_8_so.txt" )
ref_states=( "hf" "fci" )
cost_functions=( "NC" "variance" "decoupled" "fixed_sector" "switching_sector" )
# cost_functions=( "switching_sector" )


for parity_sen in "${parity_matrices[@]}"
do 
  for ref in "${ref_states[@]}"
  do
    for cost in "${cost_functions[@]}"
    do
      echo "$parity_sen $ref $cost"
      python optimize_symmetries.py $h $parity_sen --reference $ref --cost_function $cost
    done
  done
done
