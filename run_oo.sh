#! /bin/bash

hamiltonians=( "hamiltonians/H4/H4_linear_d2.5560.chk" "hamiltonians/H4/H4_linear_d2.0670.chk" "hamiltonians/H4/H4_linear_d1.0890.chk")
parity_matrices=( "hamiltonians/H4/parity_4_sens.txt" "hamiltonians/H4/LNE_2_out_of_8_so.txt" )
ref_states=( "hf" "fci" )
ref_cost_functions=( "NC" "variance" )
non_ref_cost_functions=( "decoupled" "fixed_sector" "switching_sector" )
# cost_functions=( "switching_sector" )

for h in "${hamiltonians[@]}"
do
	for parity_sen in "${parity_matrices[@]}"
	do 
		for ref in "${ref_states[@]}"
		do
		  for cost in "${ref_cost_functions[@]}"
		  do
		    echo "$parity_sen $ref $cost"
		    python optimize_symmetries.py $h $parity_sen --reference $ref --cost_function $cost
#		    python optimize_symmetries.py $h $parity_sen --reference $ref --cost_function $cost --optimizer_maxiter 0
		  done
		  for cost in "${non_ref_cost_functions[@]}"
		  do
		    echo "$parity_sen $ref $cost"
		    python optimize_symmetries.py $h $parity_sen --cost_function $cost
		  done
		  python optimize_symmetries.py $h $parity_sen --cost_function fixed_sector --optimizer_maxiter 0
		done
	done
done
