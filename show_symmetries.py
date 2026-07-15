"""Visualize quasisymmetry NC scores as a heatmap.

Optional ``--U`` applies orbital-rotation parameters before computing local
parities. Use ``--orbital_rotation irrep`` when those parameters came from an
irrep-restricted optimization.
"""

import argparse
import ffsim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from matplotlib.colors import LogNorm

from optimize_symmetries import parities, get_fci, x_to_rotation
from chemistry import load_moldata, fcidump_data
from src.orbital_rotation import resolve_orbital_rotation
from src.workflow_cli import add_orbital_rotation_arg



if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Visualize the quasisymmetries")
    parser.add_argument("molpath",
        help="path to the Hamiltonian (PySCF .chk or .FCIDUMP)")
    parser.add_argument("--reference",
        help="reference state (default: fci)",
        default="fci")
    parser.add_argument("--U", help="x as orbital rotation",
                        default=None)
    add_orbital_rotation_arg(parser)
    args = parser.parse_args()

    moldata = load_moldata(args.molpath)
    dumpdata = fcidump_data(args.molpath)

    if args.U is not None:
        x = np.loadtxt(args.U)
        pairs, _ = resolve_orbital_rotation(
            args.orbital_rotation, args.molpath, moldata.norb
        )
        U = x_to_rotation(x, moldata.norb, pairs)
    else:
        U = np.eye(moldata.norb)

    if args.reference == "fci":
        _, state = get_fci(dumpdata)
    elif args.reference == "hf":
        state = ffsim.hartree_fock_state(moldata.norb, moldata.nelec)
    elif args.reference == "cisd":
        raise NotImplementedError()
    else:
        raise ValueError("reference must be fci or hf")

    state = ffsim.apply_orbital_rotation(state, U,
                                         moldata.norb, moldata.nelec)

    local_parities = parities(moldata.norb, moldata.nelec)
    quartets = []
    for i in range(moldata.norb):
        for j in range(i + 1, moldata.norb):
            quartets.append(local_parities[i] @ local_parities[j])
    iu = np.triu_indices(moldata.norb, k=1)

    h = ffsim.linear_operator(moldata.hamiltonian.rotated(U),
                              moldata.norb,
                              moldata.nelec)

    nc_scores = np.zeros((moldata.norb, moldata.norb))
    for i in range(moldata.norb):
        c = h @ local_parities[i] - local_parities[i] @ h
        nc_scores[i, i] = np.linalg.norm(c @ state)**2

    for i in range(iu[0].shape[0]):
        c = h @ quartets[i] - quartets[i] @ h
        nc_scores[iu[0][i], iu[1][i]] = np.linalg.norm(c @ state) ** 2

    p = Path(args.molpath)

    outname = "symmetry_scores_" + p.parts[-1] + ".txt"
    np.savetxt(outname, nc_scores)
    plt.figure()
    plt.imshow(nc_scores, norm=LogNorm(vmin=1e-4, vmax=1))
    plt.colorbar()
    plt.xticks(range(moldata.norb), range(moldata.norb))
    plt.yticks(range(moldata.norb), range(moldata.norb))
    if args.reference == "fci":
        plt.title("Quartet noncommutativity norm $||[H, s_{pq}]|FCI\\rangle||^2$ \n" + args.molpath
                  + "\n Diagonal entries are $||[H, s_{p}]|FCI\\rangle||^2$")
    elif args.reference == "hf":
        plt.title("Quartet noncommutativity norm $||[H, s_{pq}]|HF\\rangle||^2$ \n" + args.molpath
                  + "\n Diagonal entries are $||[H, s_{p}]|HF\\rangle||^2$")
    plt.tight_layout()
    plt.show()




