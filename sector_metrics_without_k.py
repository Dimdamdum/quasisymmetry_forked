import argparse
import json

import numpy as np
import pyscf
from plotly.validators.histogram import cumulative

import chemistry
from metrics import *


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
        return e_fci + dumpdata["ECORE"], fcivec


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
            "Sector metrics (E_decoupled, K) from an OO JSON. "
            "Use --backend for the sector eigensolver; "
            "--coupled_energy_method reference uses a DMRG wavefunction for "
            "overlap ordering (PT needs no overlap reference). "
            "Rebuilds U from JSON rotation using orbital_rotation/irreps "
            "when present. See --help epilog."
        ),
    )
    parser.add_argument(
        "input_data", help="JSON you got from optimize_symmetries.py"
    )
    args = parser.parse_args()


    with open(args.input_data, "r") as fp:
        input_data = json.load(fp)

    moldata = load_moldata(input_data["molpath"])
    dumpdata = fcidump_data(input_data["molpath"])

    parity_matrix = np.loadtxt(input_data["parity"], dtype=int)
    symmetries = parity_matrix_to_quasisymmetries(
        parity_matrix, moldata.norb, moldata.nelec
    )

    print(parity_matrix)

    sectors = symmetry_sectors(parity_matrix, moldata.norb, moldata.nelec)

    x = np.array(input_data["rotation"])
    from src.orbital_rotation import pairs_from_oo_data

    U = x_to_rotation(x, moldata.norb, pairs_from_oo_data(input_data, moldata.norb))

    rotated_h = moldata.hamiltonian.rotated(U)
    rotated_h_linop = ffsim.linear_operator(
        rotated_h, norb=moldata.norb, nelec=moldata.nelec
    )

    e_ref, fcivec = get_fci(dumpdata)
    print("FCI ", e_ref)

    nocc = sum(dumpdata["NELEC"]) // 2
    occ_top = list(range(dumpdata["NORB"] - nocc, dumpdata["NORB"]))  # highest MOs instead of lowest
    e_max_approx = (det_energy(dumpdata["H1"],
                               pyscf.ao2mo.restore(1, dumpdata["H2"], dumpdata["NORB"]),
                               occ_top)
                    + dumpdata["ECORE"])
    print("Anti-Aufbau energy ", e_max_approx)

    approx_eckart_epsilon = chemistry.CHEMICAL_PRECISION / (e_max_approx - e_ref)

    print("Tolerance from the approximate Eckart bound {0:2.2e}".format(approx_eckart_epsilon))

    # e_max, _ = get_anti_fci(dumpdata)
    # print("max_energy", e_max)

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
                "OO_data": input_data,
                "E_FCI": e_ref,
                "Eckart": approx_eckart_epsilon,
                "used_sectors_weights_dims": used_sectors_data}

    p = Path(input_data["molpath"])
    outname = "sector_metrics_" + p.parts[-1] + "_" + str(uuid4())[:8] + ".json"

    with open(outname, "a") as fp:
        json.dump(out_data, fp, indent=2)

