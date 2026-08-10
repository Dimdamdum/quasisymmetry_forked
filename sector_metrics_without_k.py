import argparse
import json

import numpy as np
import pyscf

import chemistry
from metrics import *
from src.orbital_rotation import pairs_from_oo_data
from math import comb


def get_anti_fci(dumpdata: dict, flatten: bool = True) -> tuple[float, np.ndarray]:
    cisolver = pyscf.fci.direct_spin1.FCI()
    cisolver.max_cycle = 10000
    cisolver.conv_tol = 1e-10
    e_fci, fcivec = cisolver.kernel(
        -dumpdata["H1"],
        -dumpdata["H2"],
        dumpdata["NORB"],
        dumpdata["NELEC"],
        ecore=0,
    )
    if not cisolver.converged:
        import warnings
        warnings.warn(
            f"FCI didn't converge (conv_tol={cisolver.conv_tol}); using best available result.",
            RuntimeWarning,
            stacklevel=2,
        )
    if flatten:
        return -e_fci + dumpdata["ECORE"], np.array(fcivec.reshape((-1,)), dtype="complex")
    else:
        return -e_fci + dumpdata["ECORE"], fcivec


def det_energy(h1e, eri, occ):
    # occ: list of occupied orbital indices (alpha=beta assumed for simplicity)
    e1 = 2 * sum(h1e[i, i] for i in occ)
    e2 = 0
    for i in occ:
        for j in occ:
            e2 += 2*eri[i,i,j,j] - eri[i,j,j,i]
    return e1 + e2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Sector metrics (E_decoupled, sector_count, sectors)."
            "Either needs a JSON with orbital optimization stuff"
            "or separately a molecule (chk or fcidump), a parity matrix,"
            "and an orbital rotation (optional)."
        ),
    )
    parser.add_argument(
        "--oo_json", help="JSON you got from optimize_symmetries.py"
    )
    parser.add_argument("--molpath", help="path to either chk or fcidump")
    parser.add_argument("--symmetries",
                        help=".npy file, (k,norb) or (k,2*norb) 0/1 array")
    parser.add_argument("--rotation", default=None,
                        help=".npy file of upper-triangular generator entries")
    parser.add_argument("--upper_method", default="fci", help="how to calculate the Eckart bound")
    args = parser.parse_args()

    if args.oo_json is not None:
        with open(args.oo_json, "r") as fp:
            input_data = json.load(fp)
        molpath = input_data["molpath"]
        parity_path = input_data["parity"]

    elif args.molpath is not None and args.symmetries is not None:
        molpath = args.molpath
        parity_path = args.symmetries
    else:
        raise SystemExit("Either supply a JSON or --molpath and --symmetries")



    moldata = load_moldata(molpath)
    dumpdata = fcidump_data(molpath)

    parity_matrix = np.loadtxt(parity_path, dtype=int)
    symmetries = parity_matrix_to_quasisymmetries(
        parity_matrix, moldata.norb, moldata.nelec
    )

    print(parity_matrix)

    sectors = symmetry_sectors(parity_matrix, moldata.norb, moldata.nelec)

    if args.molpath is not None and args.symmetries is not None:
        x = np.loadtxt(args.rotation) if args.rotation is not None else np.zeros(comb(moldata.norb, 2))
        U = x_to_rotation(x, moldata.norb)
    else:
        x = np.array(input_data["rotation"])
        U = x_to_rotation(x, moldata.norb, pairs_from_oo_data(input_data, moldata.norb))

    rotated_h = moldata.hamiltonian.rotated(U)
    rotated_h_linop = ffsim.linear_operator(
        rotated_h, norb=moldata.norb, nelec=moldata.nelec
    )

    e_ref, fcivec = get_fci(dumpdata)
    print("FCI ", e_ref)

    try:
        nocc = sum(dumpdata["NELEC"]) // 2
    except TypeError:
        nocc = dumpdata["NELEC"] // 2 # sometimes it's an integer, sometimes a tuple

    if args.upper_method == "aufbau":
        occ_top = list(range(dumpdata["NORB"] - nocc, dumpdata["NORB"]))  # highest MOs instead of lowest
        e_max_approx = (det_energy(dumpdata["H1"],
                                   pyscf.ao2mo.restore(1, dumpdata["H2"], dumpdata["NORB"]),
                                   occ_top)
                        + dumpdata["ECORE"])
        print("Anti-Aufbau energy ", e_max_approx)
    elif args.upper_method == "fci":
        e_max_approx, _ = get_anti_fci(dumpdata)
        print("Anti-FCI energy", e_max_approx)
    else:
        raise ValueError("--upper_method can be 'aufbau' or 'fci'")

    approx_eckart_epsilon = chemistry.CHEMICAL_PRECISION / (e_max_approx - e_ref)

    print("Tolerance from the Eckart bound {0:2.2e}".format(approx_eckart_epsilon))


    rotated_refvec = ffsim.apply_orbital_rotation(
        fcivec, U, norb=moldata.norb, nelec=moldata.nelec
    )

    sector_weights = {}

    for sector_label, sector_bitstrings in sectors.items():
        weight = np.linalg.norm(rotated_refvec[sector_bitstrings])**2
        # print(sector_label, weight)
        sector_weights[sector_label] = weight

    weights = np.array(list(sector_weights.values()))
    order = np.argsort(weights)[::-1]
    cumulative_weights = np.cumsum(weights[order])
    K = np.nan
    for i, cumweight in enumerate(cumulative_weights):
        if cumweight > 1 - approx_eckart_epsilon:
            print("Sector count", i)
            K = i + 1
            break

    used_sector_indices = order[:K]
    used_sectors_data = {}
    for i, (k, v) in enumerate(sector_weights.items()):
        if i in used_sector_indices:
            # print(k, v)
            used_sectors_data[str(k)] = (v, len(sectors[k]))

    out_data = {"args": vars(args),
                # "OO_data": input_data,
                "E_FCI": e_ref,
                "Eckart": approx_eckart_epsilon,
                "used_sectors_weights_dims": used_sectors_data}

    p = Path(molpath)
    outname = "sector_metrics_" + p.parts[-1] + "_" + str(uuid4())[:8] + ".json"

    with open(outname, "a") as fp:
        json.dump(out_data, fp, indent=2)

