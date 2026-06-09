import argparse
import pyscf
import ffsim
import numpy as np
import scipy
import json
from tqdm import tqdm
from pathlib import Path
from itertools import product

from chemistry import load_moldata, fcidump_data
from new_optimize import parity_matrix_to_quasisymmetries


def symmetry_sectors(symmetries, norb, nelec):
    pass


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Calculate the metrics")
    parser.add_argument("molpath",
        help="path to the Hamiltonian (PySCF .chk or .FCIDUMP)")
    parser.add_argument("parity_matrix",
                        help="path to the incidence matrix of symmetries")
    parser.add_argument("--U", help="x as orbital rotation",
                        default=None)
    args = parser.parse_args()

    moldata = load_moldata(args.molpath)
    dumpdata = fcidump_data(args.molpath)

    parity_matrix = np.loadtxt(args.parity_matrix, dtype=int)
    symmetries = parity_matrix_to_quasisymmetries(parity_matrix,
                                                  moldata.norb,
                                                  moldata.nelec)