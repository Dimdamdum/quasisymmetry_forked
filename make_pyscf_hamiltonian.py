"""Run SCF for a given geometry and save a PySCF checkpoint file.

Example::

    python make_pyscf_hamiltonian.py h2o 0.958 --basis sto-3g
    python make_pyscf_hamiltonian.py h2o 0.958 --basis sto-3g --point_group C2v
    python make_pyscf_hamiltonian.py n2 1.1 --basis 6-31g --point_group D2h

``--point_group`` enables PySCF molecular symmetry so the checkpoint carries
irrep-adapted MOs (needed for ``optimize_*.py --orbital_rotation irrep``).
Incompatible with ``--localized`` (Pipek–Mezey), which breaks irrep blocks.
"""

import argparse

import matplotlib.pyplot as plt
import pyscf
import pyscf.mcscf
import pyscf.tools
from pyscf import lo

from chemistry import get_geometry_and_description

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RHF and write a PySCF .chk under hamiltonians/"
    )
    parser.add_argument(
        "mol",
        help="one of: lih, h2o, h4_linear, h4_square, h4_rectangle, h2, n2",
    )
    parser.add_argument("bond", type=float, help="bond length (angstrom)")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument(
        "--mol_parameter_2",
        type=float,
        help="Additional geometry parameter (e.g. H2O H–O–H angle in degrees)",
    )
    parser.add_argument("--localized", action="store_true")
    parser.add_argument(
        "--point_group",
        default=None,
        metavar="GROUP",
        help=(
            "Enable PySCF point-group symmetry (e.g. C2v, D2h, or auto). "
            "Produces symmetry-adapted MOs for irrep-restricted orbital rotations. "
            "Incompatible with --localized."
        ),
    )
    parser.add_argument("--scf_maxiter", type=int, default=100,
                        help="Number of iterations for the SCF solver.")
    parser.add_argument("--active_space", nargs=2, type=int, default=None,
                        help="Number of orbitals and electrons in the active space. "
                             "If specified, saves an active-space FCIDUMP in addition to the"
                             "full-space PySCF checkfile.")

    args = parser.parse_args()

    if args.localized and args.point_group is not None:
        raise SystemExit(
            "--localized (Pipek–Mezey) breaks irrep blocks; "
            "do not combine it with --point_group"
        )

    if args.mol == "h2o":
        geometry, description = get_geometry_and_description(
            args.mol, args.bond, hoh_angle_deg=args.mol_parameter_2
        )
    else:
        geometry, description = get_geometry_and_description(args.mol, args.bond)

    mol = pyscf.M()
    if args.point_group is not None:
        group = args.point_group.strip()
        mol.symmetry = True if group.lower() == "auto" else group
    mol.build(atom=geometry, basis=args.basis)

    mf = pyscf.scf.RHF(mol)
    mf.max_cycle = args.scf_maxiter
    if not args.localized:
        mf.chkfile = "hamiltonians/" + description + "_" + str(args.basis) + ".chk"
        mf.kernel()
        if not mf.converged:
            raise ValueError("SCF didn't converge! Try increasing --scf_maxiter")

        if args.point_group is not None:
            print("point group:", mol.groupname)
            if hasattr(mf, "get_orbsym"):
                print("MO irreps:", mf.get_orbsym())
    else:
        mf.chkfile = "hamiltonians/" + description + "_" + str(args.basis) + "_Pipek.chk"
        mf.kernel()
        localizer = lo.PipekMezey(mol, mf.mo_coeff[:, mf.mo_occ > 0])
        loc_orbs_occ = localizer.kernel()

        mf.mo_coeff[:, mf.mo_occ > 0] = loc_orbs_occ
        print(mf.mo_coeff)
        plt.imshow(mf.mo_coeff, cmap="PuOr", vmin=-1, vmax=1)
        plt.yticks(range(mf.mo_coeff.shape[0]), mol.ao_labels())
        plt.show()

    if args.active_space is not None:
        mycas = pyscf.mcscf.CASCI(mf, *args.active_space)
        cas_filename = mf.chkfile[:-4] + "_CAS({0:}o{1:}e).FCIDUMP".format(*args.active_space)
        pyscf.tools.fcidump.from_mcscf(mycas, cas_filename)