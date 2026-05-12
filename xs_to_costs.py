import argparse
import pyscf
import ffsim
import numpy as np

from optimize import commutator_cost_fci, commutator_cost_hf, variance_cost


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("molpath",
                        help="path to the Hamiltonian (PySCF checkfile)")
    parser.add_argument("xs",
                        help="path to file with data points (one line = one point)")
    args = parser.parse_args()

    mol = pyscf.lib.chkfile.load_mol(args.molpath)
    mf = pyscf.scf.RHF(mol)
    mf.update_from_chk(args.molpath)
    moldata = ffsim.MolecularData.from_scf(mf)

    commutator_fci = commutator_cost_fci(moldata)
    commutator_hf = commutator_cost_hf(moldata)
    variance_fci = variance_cost(moldata, "fci")
    variance_hf = variance_cost(moldata, "hf")

    xs = np.loadtxt(args.xs, skiprows=1)
    n_points = xs.shape[0]

    data_filename = args.xs + "_results.txt"
    fieldnames = ["V_fci", "V_hf", "C_fci", "C_hf", "b", "c"]

    with open(data_filename,
              "a", newline="") as fp:
        fp.write(" ".join(fieldnames) + "\n")

    for i in range(n_points):
        x = xs[i, :]
        phi1, phi2 = x[-2], x[-1]
        a_opt = np.sin(phi1) * np.cos(phi2)
        b_opt = np.sin(phi1) * np.sin(phi2)
        c_opt = np.cos(phi1)
        costs_and_bc = np.array([variance_fci(x), variance_hf(x),
                                 commutator_fci(x), commutator_hf(x),
                                 b_opt / a_opt, c_opt / a_opt])
        print(costs_and_bc)

        with open(data_filename, "ab") as fp:
            np.savetxt(fp, costs_and_bc.reshape(1, costs_and_bc.shape[0]))