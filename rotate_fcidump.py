import argparse
import numpy as np
import ffsim

from chemistry import load_moldata, fcidump_data
from optimize_symmetries import x_to_rotation

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("molpath",
                        help="path to the Hamiltonian (PySCF checkfile)")
    parser.add_argument("rot",
                        help="path to the rotation")
    parser.add_argument("outname")
    args = parser.parse_args()

    moldata = load_moldata(args.molpath)
    h = ffsim.linear_operator(moldata.hamiltonian,
                  norb=moldata.norb, nelec=moldata.nelec)

    rot = np.loadtxt(args.rot, comments="{")
    U = x_to_rotation(rot, moldata.norb)

    rotated_hamiltonian = moldata.hamiltonian.rotated(U)

    rot_moldata = ffsim.MolecularData(
        atom=moldata.atom,
        basis=moldata.basis,
        spin=moldata.spin,
        nelec=moldata.nelec,
        hf_energy=moldata.hf_energy,
        one_body_integrals=rotated_hamiltonian.one_body_tensor,
        two_body_integrals=rotated_hamiltonian.two_body_tensor,
        norb=moldata.norb,
        core_energy=moldata.core_energy
    )

    rot_moldata.to_fcidump(args.outname)
