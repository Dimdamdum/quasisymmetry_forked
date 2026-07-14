"""Davidson eigensolver for symmetry-sector Hamiltonian blocks.

Uses ``pyscf.lib.davidson1`` (already a project dependency) on the same dense
sector matrices built by ``subspace_matrix`` / ``metrics.solve_eigs``.
"""

from __future__ import annotations

import time

import numpy as np
from pyscf import lib


def solve_sector_davidson(
    h_subspace,
    nroots=1,
    tol=1e-12,
    max_cycle=50,
    max_space=12,
):
    """Lowest roots of a Hermitian sector matrix via PySCF Davidson.

    Small blocks (``dim <= 2`` or ``nroots >= dim - 1``) fall back to dense
    ``eigh``, matching ``metrics.solve_eigs``. Complex matrices with negligible
    imaginary part are cast to real; genuinely complex blocks also use dense
    ``eigh`` because PySCF's Davidson path is real-oriented.

    Returns
    -------
    energies : ndarray, shape (nroots,)
    vectors : ndarray, shape (dim, nroots)
    meta : dict
        ``solver``, ``converged``, ``converged_per_root``, ``n_cycle``,
        ``elapsed_seconds``, ``dimension``, ``nroots``.
    """
    h = np.asarray(h_subspace)
    h = 0.5 * (h + h.conj().T)
    dim = int(h.shape[0])
    nroots = min(max(1, int(nroots)), dim)

    start = time.perf_counter()

    if dim == 0:
        raise ValueError("empty sector matrix")

    use_dense = dim <= 2 or nroots >= dim - 1
    if np.max(np.abs(np.asarray(h.imag, dtype=float))) >= 1e-10:
        use_dense = True
        h_work = h
    else:
        h_work = np.asarray(h.real, dtype=np.float64)

    if use_dense:
        energies, vectors = np.linalg.eigh(h_work)
        energies = np.asarray(np.real_if_close(energies[:nroots]), dtype=float)
        vectors = np.asarray(vectors[:, :nroots])
        return energies, vectors, {
            "solver": "dense",
            "converged": True,
            "converged_per_root": [True] * nroots,
            "n_cycle": 0,
            "elapsed_seconds": time.perf_counter() - start,
            "dimension": dim,
            "nroots": nroots,
        }

    diag = np.diag(h_work).copy()

    def aop(xs):
        return [h_work @ np.asarray(x, dtype=np.float64) for x in xs]

    def precond(dx, e, _x0):
        denom = diag - e
        denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
        return np.asarray(dx, dtype=np.float64) / denom

    order = np.argsort(diag)
    x0 = []
    for i in range(nroots):
        guess = np.zeros(dim, dtype=np.float64)
        guess[int(order[i])] = 1.0
        x0.append(guess)

    cycle_state = {"n_cycle": 0}

    def callback(envs):
        # pyscf passes locals() each cycle; icyc is 0-based.
        cycle_state["n_cycle"] = int(envs.get("icyc", cycle_state["n_cycle"])) + 1

    conv, energies, vectors = lib.davidson1(
        aop,
        x0,
        precond,
        tol=tol,
        max_cycle=max_cycle,
        max_space=max_space,
        nroots=nroots,
        verbose=0,
        callback=callback,
    )

    if isinstance(conv, (bool, np.bool_)):
        conv_list = [bool(conv)] * nroots
    else:
        conv_arr = np.asarray(conv, dtype=bool).ravel()
        if conv_arr.size == 1:
            conv_list = [bool(conv_arr[0])] * nroots
        else:
            conv_list = [bool(c) for c in conv_arr[:nroots]]

    energies = np.atleast_1d(np.asarray(energies, dtype=float)).ravel()[:nroots]
    if isinstance(vectors, (list, tuple)):
        vectors = np.column_stack(
            [np.asarray(v, dtype=float).ravel() for v in vectors]
        )
    else:
        vectors = np.asarray(vectors, dtype=float)
        if vectors.ndim == 1:
            vectors = vectors.reshape(dim, 1)
    if vectors.shape[1] > nroots:
        vectors = vectors[:, :nroots]

    sort = np.argsort(energies)
    energies = energies[sort]
    vectors = vectors[:, sort]
    if len(conv_list) == nroots:
        conv_list = [conv_list[i] for i in sort]

    return energies, vectors, {
        "solver": "davidson",
        "converged": bool(all(conv_list)),
        "converged_per_root": conv_list,
        "n_cycle": int(cycle_state["n_cycle"]),
        "elapsed_seconds": time.perf_counter() - start,
        "dimension": dim,
        "nroots": nroots,
    }
