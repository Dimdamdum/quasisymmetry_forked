"""Apply an orbital rotation to a Hamiltonian and write a rotated FCIDUMP.

``rot`` may be an OO JSON (uses stored ``rotation`` / ``orbital_rotation`` /
``irreps``) or a raw parameter vector. For a raw vector against a
symmetry-adapted Hamiltonian, pass ``--orbital_rotation irrep``.

Example::

    python rotate_fcidump.py mol.chk oo.json rotated.FCIDUMP
    python rotate_fcidump.py mol.chk x_opt.txt rotated.FCIDUMP --orbital_rotation irrep
"""

import argparse
import json

import ffsim
import numpy as np

from chemistry import load_moldata
from optimize_symmetries import x_to_rotation
from src.orbital_rotation import pairs_from_oo_data, resolve_orbital_rotation
from src.workflow_cli import add_orbital_rotation_arg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rotate a molecular Hamiltonian and write an FCIDUMP"
    )
    parser.add_argument(
        "molpath",
        help="path to the Hamiltonian (PySCF checkfile or FCIDUMP)",
    )
    parser.add_argument(
        "rot",
        help="path to rotation parameters (or OO JSON with a rotation field)",
    )
    parser.add_argument("outname")
    add_orbital_rotation_arg(parser)
    args = parser.parse_args()

    moldata = load_moldata(args.molpath)

    pairs = None
    if str(args.rot).endswith(".json"):
        with open(args.rot) as fp:
            oo = json.load(fp)
        rot = np.asarray(oo["rotation"], dtype=float)
        pairs = pairs_from_oo_data(oo, moldata.norb)
    else:
        rot = np.loadtxt(args.rot, comments="{")
        pairs, _ = resolve_orbital_rotation(
            args.orbital_rotation, args.molpath, moldata.norb
        )

    U = x_to_rotation(rot, moldata.norb, pairs)
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
        core_energy=moldata.core_energy,
    )

    rot_moldata.to_fcidump(args.outname)
