"""Run SCF for a given geometry and save a PySCF checkpoint file"""

import pyscf
import argparse

from chemistry import  get_geometry_and_description

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mol",
                        help="one of the following: lih, h2o, h4_linear, h4_square, h4_rectangle, h2, n2")
    parser.add_argument("bond", type=float, help="bond")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--mol_parameter_2", type=float,
                        help="Additional geometry parameter of the molecule (if any)")

    args = parser.parse_args()

    if args.mol=="h2o":
        geometry, description = get_geometry_and_description(args.mol, args.bond,
                                                             hoh_angle_deg=args.mol_parameter_2)
    else:
        geometry, description = get_geometry_and_description(args.mol, args.bond)

    mol = pyscf.M()
    mol.build(atom=geometry, basis=args.basis)

    mf = pyscf.scf.RHF(mol)
    mf.chkfile = "hamiltonians/" + description + ".chk"
    mf.kernel()




