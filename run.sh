#! /bin/bash

python new_optimize.py hamiltonians/N2/N2_1.5000_D2h.chk hamiltonians/N2/parity_1.5_D2h_sen_quart.txt --verbose
python new_optimize.py hamiltonians/N2/N2_1.5000_D2h.chk hamiltonians/N2/parity_1.5_D2h_sen_quart_ax.txt --verbose

python new_optimize.py hamiltonians/N2/N2_2.0000_D2h.chk hamiltonians/N2/parity_10_sens.txt --verbose
python new_optimize.py hamiltonians/N2/N2_2.0000_D2h.chk hamiltonians/N2/parity_8_core_number.txt --verbose
python new_optimize.py hamiltonians/N2/N2_2.000_D2h.chk hamiltonians/N2/parity_2.0_D2h_sen_quart.txt --verbose
python new_optimize.py hamiltonians/N2/N2_2.000_D2h.chk hamiltonians/N2/parity_2.0_D2h_sen_quart_ax.txt --verbose
