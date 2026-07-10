#! /bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --time=1:00:00


module load python/3.11.5

source ../test_env_3/bin/activate


#hamiltonians=( "hamiltonians/H4/H4_linear_d2.5560.chk" "hamiltonians/H4/H4_linear_d2.0670.chk" "hamiltonians/H4/H4_linear_d1.0890.chk")
#parity_matrices=( "hamiltonians/H4/parity_4_sens.txt" "hamiltonians/H4/LNE_2_out_of_8_so.txt" )

hamiltonians=( "hamiltonians/water/H2O_OH0.9580_104.5000.chk" "hamiltonians/water/H2O_OH1.6430_104.5000.chk" "hamiltonians/water/H2O_OH2.0000_104.5000.chk" "hamiltonians/water/H2O_OH2.5000_104.5000.chk")
parity_matrices_1=( "hamiltonians/water/parity_7_sens.txt" "hamiltonians/water/parity_water_0.9580.txt" "hamiltonians/water/parity_core_numbers_0.9580.txt")
parity_matrices_2=( "hamiltonians/water/parity_7_sens.txt" "hamiltonians/water/parity_water_1.643.txt" "hamiltonians/water/parity_core_numbers_1.643.txt")
parity_matrices_3=( "hamiltonians/water/parity_7_sens.txt" "hamiltonians/water/parity_water_2.000.txt" "hamiltonians/water/parity_core_numbers_2.000.txt")
parity_matrices_4=( "hamiltonians/water/parity_7_sens.txt" "hamiltonians/water/parity_water_2.500.txt" "hamiltonians/water/parity_core_numbers_2.500.txt")
ref_states=( "hf" "fci" )
ref_cost_functions=( "NC" "variance" )
non_ref_cost_functions=( "decoupled" "fixed_sector" "switching_sector" )
# cost_functions=( "switching_sector" )

for i in {0..3};
do
  h=${hamiltonians[i]}
  case "$i" in
    0) declare -n parity_matrices=parity_matrices_1 ;;
    1) declare -n parity_matrices=parity_matrices_2 ;;
    2) declare -n parity_matrices=parity_matrices_3 ;;
    3) declare -n parity_matrices=parity_matrices_4 ;;
    *) echo "Invalid value of i" >&2; exit 1 ;;
  esac

	for parity_sen in "${parity_matrices[@]}"
	do 
		for ref in "${ref_states[@]}"
		do
		  for cost in "${ref_cost_functions[@]}"
		  do
		    echo "$h $parity_sen $ref $cost"
		    python optimize_symmetries.py $h $parity_sen --reference $ref --cost_function $cost
		    python optimize_symmetries.py $h $parity_sen --reference $ref --cost_function $cost --optimizer_maxiter 0
		  done
		done
		for cost in "${non_ref_cost_functions[@]}"
		  do
#		    echo "$h $parity_sen $cost"
		    python optimize_symmetries.py $h $parity_sen --cost_function $cost
		  done
		  python optimize_symmetries.py $h $parity_sen --cost_function fixed_sector --optimizer_maxiter 0
	done
done
