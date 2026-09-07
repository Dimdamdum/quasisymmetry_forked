from __future__ import annotations

"""
Cluster Number Sector Search -- Exact Weights

Ranks candidate cluster-number sectors of a decomposition deco = (partition, U)
by their EXACT weight w(N) = <psi|Pi_N|psi> in the ground-state DMRG
wavefunction, instead of cluster_number_sector_search.py's cheap tier-0/1/2
heuristics (a Gaussian/max-entropy weight estimate, optionally refined by an
RDM-contraction-based coupling-strength score). w(N) is computed via a direct
environment sweep on the block-sparse tensors of the ground-state MPS after
it has been rotated into deco's own orbital basis, exploiting the MPS's
built-in U(1) particle-number conservation (block2/pyblock2's native
quantum-number-block structure). No MPO is built and no iterative fitting/
projection is done for the sweep itself, and the sweep's OWN answer is exact
to floating-point precision for whatever state the rotated MPS actually
holds -- see decode_mps/sector_weight_native's docstrings and this module's
test suite, which validate this independently of DMRG/rotation convergence
(including against a deliberately non-converged, truncated-bond-dimension
MPS). The one place finite-precision DOES enter is the rotation step itself:
expressing the existing ground-state MPS in deco's basis is a short td_dmrg
(TDVP) run, not an exact operation, so the OVERALL weight (rotate, then
sweep) carries whatever residual error that run left behind -- shrinking
roughly linearly in n_td_steps (SectorSearchConfig.n_td_steps /
--n-td-steps), and typically much smaller in practice than a synthetic
worst-case rotation shows, since a beam-search trajectory's own deco.U is
usually a much smaller, more localized rotation than an arbitrary one (see
this module's test suite for both the worst-case convergence curve and a
real-trajectory-rotation measurement). If exact weight VALUES (not just
their relative ranking) matter for a downstream use, increase n_td_steps
and confirm the answer stops moving.

Why this is worth the (still modest) extra cost over the tier heuristics:
tier-0's Gaussian weight is a maximum-entropy estimate from the first two
RDM-derived moments only -- it can misrank sectors whenever the true joint
distribution isn't well approximated by a Gaussian (e.g. multimodal or
strongly skewed distributions, which do arise for stretched-geometry or
otherwise strongly-correlated systems). Tier-1/2's coupling-strength score
adds a direct RDM-contraction estimate of "how much does H couple this
sector to the main one", which helps but is still a perturbative estimate,
not the sector's actual weight. This module instead spends a bounded,
onetime cost (the rotation) to get the exact answer, and found -- see
tests/test_cluster_number_sector_search_exact_weights.py -- that the per-
label cost of the sweep ITSELF is cheap and essentially independent of
bond dimension, cluster size, and how many labels are swept per rotated
MPS: ranking is therefore no more expensive, in the parameters that matter
for the search (partition granularity, number of candidate sectors), than
the tier-0-only heuristic once the rotation is paid for. See
cluster_number_sector_search.py's own module docstring for the parts of the
algorithm this module reuses UNCHANGED (Steps 0, C, E below) since they have
nothing to do with how a candidate sector is SCORED.

Algorithm (see rank_relevant_sectors for the orchestration):
  0. normalize_cluster_family (reused from cluster_number_sector_search.py,
     unchanged): tolerate messy input partitions.
  A. cluster_number_moments (reused, unchanged): mean vector mu of the
     cluster number operators, from the 1-/2-RDM alone -- used only to seed
     main_sector_label below (the covariance it also computes is unused
     here: it fed tier-0's Gaussian score, which no longer exists).
  B. main_sector_label (reused, unchanged): a cheap, closed-form estimate of
     the main sector holding most of the state's weight -- still the
     natural center to search outward from, even though nothing here scores
     sectors by distance from it anymore.
  C. enumerate_candidate_labels (reused, unchanged): which OTHER sectors are
     even worth exact-weighing -- a bounded graph search (BFS over
     elementary single-electron cluster-to-cluster moves) out to
     max_elec_transfer, not an exhaustive sweep over every sector (which,
     for a fine partition, is still combinatorially many even though this
     module could in principle exact-weigh any of them cheaply once the
     rotation is paid for -- the search still needs SOME bound on how far
     from the main sector to look).
  D. rotate_mps_to_basis / decode_mps / sector_weight_native: the exact
     replacement for tier-0/1/2's scores. Rotates the ground-state MPS into
     deco's basis (handling det(U) = -1 via a single-mode reflection
     factoring, exactly as validated for proper AND improper rotations --
     see this module's own test suite), decodes its block-sparse tensor
     structure ONCE per decomposition, then sweeps the resulting
     environment once per candidate label (cheap: the decode is the only
     nontrivial cost; see the module-level NOTE below on the decoder fix).
  E. sector_dimension (reused, unchanged): the exact size of a sector's
     Hilbert-space dimension.
  F. rank_relevant_sectors: ties 0-E together into a single ranked,
     greedily-truncated (num_sectors_to_retain / max_cum_dim_to_retain)
     list of SectorRelevance entries, sorted by weight alone (no more
     tier-priority logic needed: an exact weight is already the single
     right ranking key for "how much of the wavefunction lives here").

NOTE on pyblock2.algebra.io.MPSTools.from_block2: that library function is
missing dispatch branches for a dot=2 MPS whose orthogonality center touches
a chain boundary (confirmed to arise from td_dmrg's own TDVP sweep schedule
at ordinary, unremarkable bond dimensions -- not just at pathological ones --
across every system/partition this module's test suite covers). This module
carries a patched copy (_patched_from_block2 below) fixing both boundary
configurations that were ever empirically observed to occur (see that
function's own docstring for the full derivation); an internally consistent
but genuinely novel third configuration would raise a clear AssertionError
rather than silently mis-decoding, so a failure here is always loud, never
silent.

CLI usage (mirrors cluster_number_sector_search.py, minus the now-meaningless
--force-h1e/--force-full-rdms tier flags):
    python cluster_number_sector_search_exact_weights.py h2o sto-3g 2.0 commutator --bond-angle 104.5
    python cluster_number_sector_search_exact_weights.py h4_square 6-31g 1.0 variance --analyze-num-clusters 3
    python cluster_number_sector_search_exact_weights.py h2o sto-3g 2.0 commutator --K-sector-analysis

Library usage (bring your own Decomposition + RDMData + solver, e.g. from
cluster_number_decomposition_optimization.py's own trajectory):
    from cluster_number_sector_search_exact_weights import rank_relevant_sectors, SectorSearchConfig
    entries = rank_relevant_sectors(deco, rdm_data, nelec, solver, SectorSearchConfig(max_elec_transfer=2))

See tests/test_cluster_number_sector_search_exact_weights.py for correctness
validation, including extensive comparisons against
cluster_number_decomposition_optimization.py's / cluster_numbers_metrics.py's
own full-state route (get_sector_weight, get_ordered_state_projections_in_sectors).
"""

import argparse
import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg

from pyblock2.algebra.io import TensorTools
from pyblock2.algebra.core import MPS as AlgebraMPS

from cluster_number_decomposition_optimization import (
    Decomposition,
    RDMData,
    rotate_rdm_data,
)
from cluster_number_sector_search import (
    KSectorAnalysisState,
    cluster_number_moments,
    enumerate_candidate_labels,
    get_relevance_ranked_K_sectors_values_energies,
    main_sector_label,
    normalize_cluster_family,
    sector_dimension,
    _run_K_sector_analysis_for_entry,
    _setup_K_sector_analysis,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class SectorSearchConfig:
    """Hyperparameters for rank_relevant_sectors."""

    num_sectors_to_retain: int | None = None  # (max) number of sectors to retain; None = unlimited
    max_cum_dim_to_retain: int | None = None  # max sum of retained sectors' dimensions; None = unlimited
    max_elec_transfer: int = 2  # candidate labels are generated out to this many electrons moved
    n_td_steps: int = 20  # TDVP steps for the one-time orbital rotation (see rotate_mps_to_basis)


@dataclass
class SectorRelevance:
    """One ranked candidate sector."""

    label: tuple[int, ...]  # (N_0, ..., N_{K-1}), K = len(deco.partition) (ghost cluster, if any, stripped)
    weight: float  # exact <psi|Pi_label|psi> in [0, 1] -- sums to 1 over the full sector decomposition
    elec_transfer: int  # t = (1/2) * sum(|label - main_label|)
    dimension: int  # exact size of this sector's determinant support


# =============================================================================
# Method 3 core: exact sector weights from the MPS's own block-sparse
# structure, via a patched pyblock2 decoder + a direct environment sweep.
# =============================================================================


def _patched_from_block2(bmps):
    """Patched copy of pyblock2.algebra.io.MPSTools.from_block2, fixing two
    missing dispatch branches for a dot=2 MPS whose orthogonality center
    touches a chain boundary (confirmed, by direct empirical inspection
    across LiH/H2O/N2 at many bond dimensions and both td_dmrg step-count
    parities, that the wavefunction tensor always sits at the literal
    boundary site itself -- 0 or n_sites-1 -- never at its neighbour, for
    every boundary-touching MPS td_dmrg's own TDVP sweep produces; the two
    cases below are therefore the complete set actually reachable this way,
    not an arbitrary subset -- see tests/test_cluster_number_sector_search_
    exact_weights.py for the stress test covering both, at many bond
    dimensions/seeds/systems, for both proper and improper rotations).

    RIGHT boundary (bmps.center == bmps.n_sites - 2): the wavefunction sits
    at site i = center+1 = n_sites-1. This matches none of stock
    from_block2's dispatch conditions and hits its `assert False` fallback:
    tensors[center+1] here has info.is_wavefunction=True and
    n_states_ket[i]==1 for every block (a trivial right boundary), i.e. it
    is structurally identical to the already-handled case "i == n_sites-1
    and i == center and dot == 1" (single boundary site, decoded via
    TensorTools.from_block2_no_fused, which already applies the
    is_wavefunction sign flip internally). Fix: also route this site to
    from_block2_no_fused. Its own neighbour, site `center` itself
    (n_sites-2), already has a dedicated stock dispatch condition
  ("i == center and i == n_sites-2 and dot == 2" -> from_block2_left_fused)
    and needs no change.

    LEFT boundary (bmps.center == 0): the wavefunction sits at site i =
    center = 0. Stock's own "i == center and i == 0 and dot == 2" condition
    already fires here, but routes it to from_block2_right_fused using a
    recipe ("borrow site 1's basis to fuse into the ket") that assumes site
    1 has NO independent tensor of its own -- wrong whenever
    bmps.tensors[1] is not None (the case in every empirically observed
    instance: it crashes with "ValueError: cannot reshape array of size
    0..."). Fix: detect this via bmps.tensors[1] is not None and route site
    0 to from_block2_no_fused instead (mirroring the right-boundary fix).
    Site 0's neighbour, site 1, then has NO stock dispatch condition that
    reaches it at all (it satisfies neither "i < center" -- 1 < 0 is false
    -- nor "i >= center + dot" -- 1 >= 2 is false -- nor
    canonical_form[i] == "S"): it needs a genuinely new branch, decoded via
    from_block2_right_fused with the ordinary same-site recipe (m=basis[1],
    r=right_dims[2]) -- exactly the recipe stock already uses for
    i >= center+dot, just one site short of that threshold here.

    A second, easy-to-miss consequence of the left-boundary fix concerns
    the QUANTUM-NUMBER CONVENTION, not just which decode function to call.
    block2's own from_block2 has TWO different bond-label conventions live
    at once: ordinary ("L"-canonicalized) tensors carry plain
    left-cumulative particle counts, but the tensor at the orthogonality
    center itself, and everything from_block2 decodes via the right-fused
    recipe, comes out in a "right-block" convention (effectively target
    minus the true cumulative count) that has to be converted with a single
    target-minus-q pass to match. from_block2's own post-loop already does
    this unconditionally for tensors[center] and for the whole tail
    range(center+dot, n_sites) -- correct for the ordinary, non-boundary
    case, where sector_weight_native's own further "if i == center:
    qr = target - qr" step composes with that pass to undo it, recovering
    the tensor's TRUE left-cumulative label (this was this project's
    earlier, validated fix for the interior case). For a LEFT-boundary
    wavefunction sitting at the center itself, though, that same composition
    is WRONG: the two conversions cancel back to a raw, non-cumulative
    value that happens to still chain correctly bond-to-bond (so a
    normalization check alone -- summing with no sector cut at all --
    stayed exactly 1.0 and gave no hint anything was wrong) but silently
    breaks any cluster-boundary CUT that lands exactly at this bond, e.g.
    for an all-singleton partition (confirmed: those specific cases came
    back with sector weights identically 0.0, not merely inaccurate). Fixed
    by skipping from_block2's own unconditional center-relabeling pass in
    exactly this one case (left_boundary_wf below), leaving
    sector_weight_native's existing, unmodified center-correction to
    provide the single conversion this case actually needs; site 1's own
    two labels get that same single conversion applied explicitly here
    instead, since it sits one index outside the pre-existing
    range(center+dot, n_sites) pass.
    """
    tensors = [None] * bmps.n_sites

    left_boundary_wf = (
        bmps.dot == 2
        and bmps.center == 0
        and bmps.tensors[0] is not None
        and bmps.tensors[1] is not None
    )
    if left_boundary_wf:
        bmps.load_tensor(0)
        left_boundary_wf = bmps.tensors[0].info.is_wavefunction
        bmps.unload_tensor(0)

    right_boundary_wf = (
        bmps.dot == 2
        and bmps.center == bmps.n_sites - 2
        and bmps.tensors[bmps.center + 1] is not None
    )
    if right_boundary_wf:
        bmps.load_tensor(bmps.center + 1)
        right_boundary_wf = bmps.tensors[bmps.center + 1].info.is_wavefunction
        bmps.unload_tensor(bmps.center + 1)

    for i in range(0, bmps.n_sites):
        if bmps.tensors[i] is None:
            continue

        if i == 0 and left_boundary_wf:
            bmps.load_tensor(i)
            tensors[i] = TensorTools.from_block2_no_fused(bmps.tensors[i])
            bmps.unload_tensor(i)
        elif i == 1 and left_boundary_wf:
            bmps.info.load_right_dims(i + 1)
            m = bmps.info.basis[i]
            r = bmps.info.right_dims[i + 1]
            mr = m.__class__.tensor_product_ref(m, r, bmps.info.right_dims_fci[i])
            cmr = m.__class__.get_connection_info(m, r, mr)
            bmps.load_tensor(i)
            tensors[i] = TensorTools.from_block2_right_fused(bmps.tensors[i], m, r, mr, cmr)
            bmps.unload_tensor(i)
            mr.deallocate()
            r.deallocate()
            for block in tensors[i].blocks:
                block.q_labels = (
                    bmps.info.target - block.q_labels[0],
                    block.q_labels[1],
                    bmps.info.target - block.q_labels[2],
                )
        elif (
            (i == 0 and i < bmps.center)
            or (i == bmps.n_sites - 1 and i >= bmps.center + bmps.dot)
            or (i == 0 and i == bmps.center and bmps.dot == 1)
            or (
                i == 0
                and i == bmps.center
                and i == bmps.n_sites - 2
                and bmps.dot == 2
            )
            or (i == bmps.n_sites - 1 and bmps.dot == 2 and right_boundary_wf)
        ):
            bmps.load_tensor(i)
            tensors[i] = TensorTools.from_block2_no_fused(bmps.tensors[i])
            bmps.unload_tensor(i)
        elif (
            i < bmps.center
            or (i == bmps.center and i == bmps.n_sites - 2 and bmps.dot == 2)
            or (
                i == bmps.center and bmps.dot == 1 and bmps.canonical_form[i] != "S"
            )
        ):
            bmps.info.load_left_dims(i)
            l = bmps.info.left_dims[i]
            m = bmps.info.basis[i]
            lm = m.__class__.tensor_product_ref(
                l, m, bmps.info.left_dims_fci[i + 1]
            )
            clm = m.__class__.get_connection_info(l, m, lm)
            bmps.load_tensor(i)
            if i == bmps.n_sites - 1 and i == bmps.center and bmps.dot == 1:
                if (
                    bmps.tensors[i].info.n == 1
                    and bmps.tensors[i].info.quanta[0].get_ket()
                    == -bmps.info.target
                ):
                    tensors[i] = TensorTools.from_block2_fused(
                        bmps.tensors[i], l, m, lm, clm
                    )
                else:
                    tensors[i] = TensorTools.from_block2_no_fused(bmps.tensors[i])
            else:
                tensors[i] = TensorTools.from_block2_left_fused(
                    bmps.tensors[i], l, m, lm, clm
                )
            bmps.unload_tensor(i)
            lm.deallocate()
            l.deallocate()
        elif (
            i >= bmps.center + bmps.dot
            or (i == bmps.center and i == 0 and bmps.dot == 2)
            or bmps.canonical_form[i] == "S"
        ):
            if i >= bmps.center + bmps.dot or bmps.canonical_form[i] == "S":
                bmps.info.load_right_dims(i + 1)
                m = bmps.info.basis[i]
                r = bmps.info.right_dims[i + 1]
                mr = m.__class__.tensor_product_ref(
                    m, r, bmps.info.right_dims_fci[i]
                )
            else:
                bmps.info.load_right_dims(i + 2)
                m = bmps.info.basis[i + 1]
                r = bmps.info.right_dims[i + 2]
                mr = m.__class__.tensor_product_ref(
                    m, r, bmps.info.right_dims_fci[i + 1]
                )
            cmr = m.__class__.get_connection_info(m, r, mr)
            bmps.load_tensor(i)
            tensors[i] = TensorTools.from_block2_right_fused(
                bmps.tensors[i], m, r, mr, cmr
            )
            bmps.unload_tensor(i)
            mr.deallocate()
            r.deallocate()
        elif (
            i == bmps.center and i != 0 and i != bmps.n_sites - 2 and bmps.dot == 2
        ):
            bmps.info.load_left_dims(i)
            bmps.info.load_right_dims(i + 2)
            l = bmps.info.left_dims[i]
            ma = bmps.info.basis[i]
            mb = bmps.info.basis[i + 1]
            r = bmps.info.right_dims[i + 2]
            lm = ma.__class__.tensor_product_ref(
                l, ma, bmps.info.left_dims_fci[i + 1]
            )
            mr = ma.__class__.tensor_product_ref(
                mb, r, bmps.info.right_dims_fci[i + 1]
            )
            clm = ma.__class__.get_connection_info(l, ma, lm)
            cmr = ma.__class__.get_connection_info(mb, r, mr)
            bmps.load_tensor(i)
            tensors[i] = TensorTools.from_block2_left_and_right_fused(
                bmps.tensors[i], l, ma, mb, r, lm, clm, mr, cmr
            )
            bmps.unload_tensor(i)
            mr.deallocate()
            lm.deallocate()
            r.deallocate()
            l.deallocate()
        else:
            raise AssertionError(
                f"_patched_from_block2: no matching dispatch branch for site {i} "
                f"(center={bmps.center}, dot={bmps.dot}, n_sites={bmps.n_sites}, "
                f"canonical_form={bmps.canonical_form}) -- a genuinely new "
                f"boundary configuration, not one of the two this patch targets "
                f"(see this function's own docstring). Please report the "
                f"(molecule, bond_dim, seed) that triggered this so the fix can "
                f"be extended."
            )

    if bmps.center != bmps.n_sites - 1 and not left_boundary_wf:
        for block in tensors[bmps.center].blocks:
            block.q_labels = block.q_labels[:-1] + (
                bmps.info.target - block.q_labels[-1],
            )
    for i in range(bmps.center + bmps.dot, bmps.n_sites):
        for block in tensors[i].blocks:
            if block.rank == 3:
                block.q_labels = (
                    bmps.info.target - block.q_labels[0],
                    block.q_labels[1],
                    bmps.info.target - block.q_labels[2],
                )
            elif block.rank == 2:
                block.q_labels = (
                    bmps.info.target - block.q_labels[0],
                    block.q_labels[1],
                )
            else:
                raise AssertionError("unexpected tensor rank in post-loop relabeling")
    return AlgebraMPS(tensors=tensors)


def _qkey(q) -> tuple:
    """Hashable dict key for a block2 quantum-number object (particle
    number, 2*Sz, point-group irrep)."""
    return (q.n, q.twos, q.pg)


def decode_mps(mps) -> tuple[list, int, Any]:
    """(tensors, center, target) for one rotated MPS, decoded once via
    _patched_from_block2 -- reuse the result across every candidate label's
    sector_weight_native call (the decode is the only nontrivial per-MPS
    cost; the sweep itself is cheap and label-count-independent, see this
    module's test suite)."""
    decoded = _patched_from_block2(mps)
    return decoded.tensors, mps.center, mps.info.target


def _orbital_to_cluster_map(clean_partition: list[list[int]]) -> dict[int, int]:
    """orbital index -> cluster index, from a normalize_cluster_family-clean
    partition (disjoint, covers every orbital)."""
    mapping: dict[int, int] = {}
    for k, cluster in enumerate(clean_partition):
        for orb in cluster:
            mapping[orb] = k
    return mapping


def _full_environment_sweep(
    decoded: tuple[list, int, Any], orbital_to_cluster: dict[int, int], num_clusters: int
) -> dict[tuple[int, ...], np.ndarray]:
    """{per_cluster_counts: matrix} at the end of the chain, from one full
    left-to-right environment sweep over decode_mps's output.

    Tracks, at every bond, BOTH the MPS's own (bond-native) particle/spin/
    irrep quantum number -- needed to only ever contract mutually-compatible
    blocks, exactly as the MPS's block-sparse structure requires -- AND a
    length-num_clusters tuple of how many electrons have been assigned to
    each cluster so far. The second part is NOT redundant with the first: a
    cluster's own orbitals can be scattered anywhere along the chain, in any
    order (a real cluster_number_decomposition_optimization.py trajectory's
    partition is generally NOT a set of contiguous, chain-order blocks --
    e.g. a singleton-cluster decomposition typically lists clusters in an
    order unrelated to orbital order), so no single "cut point" can isolate
    one cluster's running count from another's the way it could for the
    special case of a partition whose clusters DO happen to be contiguous.
    Tracking the full per-cluster breakdown throughout, and filtering only
    once at the very end (see sector_weight_native / native_sweep_all_labels),
    handles both cases uniformly with no special-casing of contiguous
    partitions needed.

    Each site's own physical (electron-count) contribution is read directly
    off qp -- block2's own local physical basis quantum number, always
    unambiguous and never touched by the target-minus-q relabeling that
    ql/qr sometimes need (see _patched_from_block2's docstring) -- so this
    is correct regardless of which decode branch a given site went through.
    """
    tensors, center, target = decoded
    n_sites = len(tensors)
    zero_counts = (0,) * num_clusters
    env: dict[tuple, dict[tuple[int, ...], np.ndarray]] = {(): {zero_counts: np.array([[1.0]])}}
    for i, t in enumerate(tensors):
        k = orbital_to_cluster[i]
        new_env: dict[tuple, dict[tuple[int, ...], np.ndarray]] = {}
        for b in t.blocks:
            if b.rank == 3:
                ql, qp, qr = b.q_labels
            elif b.rank == 2 and i == 0:
                qp, qr = b.q_labels
                ql = None
            elif b.rank == 2 and i == n_sites - 1:
                ql, qp = b.q_labels
                qr = None
            else:
                raise AssertionError(f"site {i}: unexpected block rank {b.rank}")

            if i == center and qr is not None:
                qr = target - qr

            left_bond_key = () if ql is None else _qkey(ql)
            bucket = env.get(left_bond_key)
            if bucket is None:
                continue
            right_bond_key = () if qr is None else _qkey(qr)
            electrons_here = qp.n
            arr = b.reduced
            if arr.ndim == 2:
                arr = (
                    arr.reshape(arr.shape[0], 1, arr.shape[1])
                    if i == 0
                    else arr.reshape(arr.shape[0], arr.shape[1], 1)
                )
            new_bucket = new_env.setdefault(right_bond_key, {})
            for counts, e in bucket.items():
                new_counts = counts[:k] + (counts[k] + electrons_here,) + counts[k + 1 :]
                for p in range(arr.shape[1]):
                    block = arr[:, p, :]
                    contrib = block.conj().T @ e @ block
                    new_bucket[new_counts] = new_bucket.get(new_counts, 0.0) + contrib
        env = new_env
    return env.get((), {})


def sector_weight_native(
    decoded: tuple[list, int, Any], orbital_to_cluster: dict[int, int], num_clusters: int,
    target_label,
) -> float:
    """<psi|Pi_target_label|psi> for ONE label (a length-num_clusters tuple
    of per-cluster particle counts), via one full environment sweep (see
    _full_environment_sweep). Prefer native_sweep_all_labels when scoring
    many candidate labels against the same decoded MPS: the sweep itself
    doesn't depend on which label(s) are ultimately wanted, so it only ever
    needs to run once."""
    final = _full_environment_sweep(decoded, orbital_to_cluster, num_clusters)
    v = final.get(tuple(target_label))
    if v is None:
        return 0.0
    return max(0.0, float(np.real(np.trace(v))))  # max(0, .) guards O(1e-16) noise only


def native_sweep_all_labels(mps, clean_partition: list[list[int]], labels=None) -> dict[tuple[int, ...], float]:
    """{label: weight} for every label in `labels` (or every label reachable
    at all, if labels is None), label = (N_0,...,N_{K-1}) per-cluster tuple
    matching clean_partition -- clusters need NOT be contiguous or
    chain-ordered (see _full_environment_sweep). Decodes the MPS and sweeps
    it exactly once, however many labels are requested."""
    decoded = decode_mps(mps)
    orbital_to_cluster = _orbital_to_cluster_map(clean_partition)
    num_clusters = len(clean_partition)
    final = _full_environment_sweep(decoded, orbital_to_cluster, num_clusters)

    if labels is None:
        return {counts: max(0.0, float(np.real(np.trace(v)))) for counts, v in final.items()}
    out = {}
    for label in labels:
        key = tuple(label[:num_clusters])
        v = final.get(key)
        out[tuple(label)] = max(0.0, float(np.real(np.trace(v)))) if v is not None else 0.0
    return out


# =============================================================================
# Orbital rotation: express the existing ground-state MPS in a new basis
# =============================================================================


def rotate_mps_to_basis(solver: Any, U: np.ndarray, n_td_steps: int = 20, ket0=None):
    """Rotate solver's ground-state MPS by the orthogonal matrix U (any
    det) into deco's own basis, returning the rotated block2 MPS.

    Proper rotations (det(U) = +1) are applied directly via a short td_dmrg
    run with generator A = logm(U) (antisymmetrized to remove floating-point
    residue). Improper rotations (det(U) = -1) are factored as U = R @ F,
    with F a single-mode reflection on the last orbital (det(F) = -1, so
    det(R) = +1): F is applied first via the exact, MPO-based
    solver.apply_rotated_parity (a single diagonal factor for a one-orbital
    parity row, so this is exact, not fit/truncated beyond the MPS's own
    bond dimension), then R via td_dmrg exactly as in the proper case. This
    composition order (reflect first, then rotate) is the one validated
    against the full-state ground truth for improper rotations -- see this
    module's test suite.

    ket0 defaults to solver.get_mps() (the ground state in its original,
    reference basis -- the basis deco.U is defined relative to). Temporary
    MPS tags created here (the reflection intermediate, if any) are deleted
    before returning; the final rotated MPS's own tag is left for the
    caller to consume and is the caller's responsibility to delete once
    done with it (e.g. after decode_mps has extracted what it needs).

    Calls solver._activate() up front: block2 keeps one global scratch
    frame per process (see Block2DMRGSolver._activate's own docstring), and
    this function -- unlike every OTHER Block2DMRGSolver method it's built
    alongside -- talks to solver.driver directly (expr_builder/get_mpo/
    td_dmrg) rather than through a wrapper that already guarantees
    reactivation. Skipping this is silently wrong, not loudly: with a
    second solver active, solver.driver still exists and superficially
    works, it just quietly points at the OTHER solver's scratch directory,
    surfacing (if at all) as a confusing "file not found" deep inside
    block2 rather than as an obviously-solver-related error -- confirmed
    directly during this module's own test development, when reusing one
    solver across multiple test functions in the same process hit exactly
    this.
    """
    solver._activate()
    norb = solver.n_sites
    U = np.asarray(U, dtype=np.float64)
    if ket0 is None:
        ket0 = solver.get_mps()
    run_tag = uuid.uuid4().hex[:8]

    det_U = np.linalg.det(U)
    if det_U < 0:
        parity_row = np.zeros(norb)
        parity_row[-1] = 1.0
        reflected = solver.apply_rotated_parity(
            parity_row, np.eye(norb), ket=ket0, tag=f"EW_REFL_{run_tag}", n_sweeps=8, tol=1e-12,
        )
        F = np.eye(norb)
        F[-1, -1] = -1.0
        R = U @ F
        ket0 = reflected
    else:
        R = U
        reflected = None

    A = scipy.linalg.logm(R)
    if np.linalg.norm(A.imag) > 1e-8:
        raise ValueError(f"logm(R) has a non-negligible imaginary part: {np.linalg.norm(A.imag):.3e}")
    A = 0.5 * (np.real(A) - np.real(A).T)

    if np.linalg.norm(A) < 1e-12:
        # R is (numerically) the identity: no further rotation needed. Note
        # this can return the CALLER's own ket0 unchanged (proper-U case,
        # reflected is None) -- callers must not delete_mps_tag() the
        # result without first checking it isn't their own input (see
        # rank_relevant_sectors for why this matters).
        return ket0

    builder = solver.driver.expr_builder()
    builder.add_sum_term("cd", A)
    builder.add_sum_term("CD", A)
    mpo_A = solver.driver.get_mpo(builder.finalize(), iprint=0)
    max_bd = max(int(ket0.info.get_max_bond_dimension()), 200)
    mps_rot = solver.driver.td_dmrg(
        mpo_A, ket0, delta_t=-1.0 / n_td_steps, target_t=-1.0, n_steps=n_td_steps,
        te_type="tdvp", bond_dims=[max_bd] * n_td_steps, final_mps_tag=f"EW_ROT_{run_tag}",
        normalize_mps=False, hermitian=False, iprint=0,
    )
    if reflected is not None:
        solver.delete_mps_tag(reflected.info.tag)
    return mps_rot


# =============================================================================
# Step F: orchestration
# =============================================================================


def rank_relevant_sectors(
    deco: Decomposition,
    rdm_data: RDMData,
    nelec: tuple[int, int],
    solver: Any,
    config: SectorSearchConfig | None = None,
    mps: Any = None,
) -> list[SectorRelevance]:
    """Rank candidate cluster-number sectors of `deco` by their EXACT weight
    (see module docstring for the algorithm). `rdm_data` is in the same
    reference basis deco.U is relative to (e.g. MOs); `mps` must hold the
    ground state in that SAME reference basis -- the one rank_relevant_sectors
    rotates into deco's own basis to compute exact weights. `nelec` must be
    given explicitly -- RDMData has no nelec field.

    mps defaults to solver.get_mps() (the production pipeline's own
    convention: the ground state saved under the solver's default MPS tag).
    Passing it explicitly is mainly useful for tests that build a solver/MPS
    pair without persisting it under that tag (e.g. a random MPS built via
    solver.driver.get_random_mps -- see this module's own test suite).
    """
    if config is None:
        config = SectorSearchConfig()
    if mps is None:
        mps = solver.get_mps()

    norb = rdm_data.norb
    clean_partition, _cluster_map, had_ghost = normalize_cluster_family(deco.partition, norb)
    if had_ghost:
        logger.warning(
            "rank_relevant_sectors: deco.partition does not cover all orbitals; an "
            "implicit ghost cluster was added for internal bookkeeping and its "
            "coordinate is stripped from every label returned here."
        )
    cluster_sizes = [len(c) for c in clean_partition]
    K_real = len(clean_partition) - 1 if had_ghost else len(clean_partition)

    rdm_data_cur = rotate_rdm_data(rdm_data, deco.U)
    mu, _sigma = cluster_number_moments(rdm_data_cur, clean_partition)
    main_label = main_sector_label(mu, nelec, cluster_sizes)

    candidates = enumerate_candidate_labels(main_label, cluster_sizes, config.max_elec_transfer)

    t0 = time.time()
    mps_rot = rotate_mps_to_basis(solver, deco.U, n_td_steps=config.n_td_steps, ket0=mps)
    logger.debug("rank_relevant_sectors: rotation took %.3fs", time.time() - t0)

    t0 = time.time()
    weights = native_sweep_all_labels(mps_rot, clean_partition, list(candidates.keys()))
    logger.debug(
        "rank_relevant_sectors: swept %d candidate labels in %.4fs", len(candidates), time.time() - t0
    )
    # Only clean up an MPS rotate_mps_to_basis actually created: for a
    # (near-)identity deco.U -- a real, common case, e.g. the plain-MOs
    # trajectory seed -- it short-circuits and returns `mps` itself
    # unchanged (see its own docstring). Deleting mps_rot's tag
    # unconditionally would then delete the CALLER's own input MPS -- the
    # ground state itself, in main()'s own usage -- silently destroying it
    # on disk for the rest of the pipeline (any later trajectory entry, or
    # the K-sector-analysis step after this function returns). Confirmed
    # to actually happen this way during this module's own test development.
    if mps_rot is not mps:
        solver.delete_mps_tag(mps_rot.info.tag)

    results: list[SectorRelevance] = []
    for label, t in candidates.items():
        reported_label = label[:K_real] if had_ghost else label
        results.append(
            SectorRelevance(
                label=reported_label,
                weight=weights[tuple(label)],
                elec_transfer=t,
                dimension=sector_dimension(label, cluster_sizes, nelec),
            )
        )

    results.sort(key=lambda r: r.weight, reverse=True)

    retained: list[SectorRelevance] = []
    cum_dim = 0
    for r in results:
        if config.num_sectors_to_retain is not None and len(retained) >= config.num_sectors_to_retain:
            break
        if (
            config.max_cum_dim_to_retain is not None
            and retained
            and cum_dim + r.dimension > config.max_cum_dim_to_retain
        ):
            break
        if (
            config.max_cum_dim_to_retain is not None
            and not retained
            and r.dimension > config.max_cum_dim_to_retain
        ):
            logger.warning(
                "rank_relevant_sectors: the top-ranked sector's dimension (%d) alone "
                "exceeds max_cum_dim_to_retain (%d); retaining it anyway so the result "
                "isn't empty.",
                r.dimension,
                config.max_cum_dim_to_retain,
            )
        retained.append(r)
        cum_dim += r.dimension

    return retained


# =============================================================================
# Sector analysis using rank_relevant_sectors output (for flag --K-sector-analysis)
#
# get_relevance_ranked_K_sectors_values_energies / KSectorAnalysisState /
# _setup_K_sector_analysis / _run_K_sector_analysis_for_entry are reused
# UNCHANGED from cluster_number_sector_search.py (imported above): none of
# them read SectorRelevance.weight_score/.energy_score/.energy_tier -- only
# .label and .dimension, which this module's own SectorRelevance still has.
# Only the plot-filename tier prefix (there: 0/1/2 for which energy tier was
# available) needs a replacement here, since there is no tier any more.
# =============================================================================


def _finalize_K_sector_analysis(
    state: KSectorAnalysisState,
    args: argparse.Namespace,
    dmrg_energy: float,
    norb: int,
    timestamp: str,
    git_hash: str,
) -> None:
    if not state.data_label_list:
        logger.info("No K-sector-analysis curves to plot.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.K_sectors_plots import plot_energy_vs_K

    plots_dir = _plots_dir_for(args)
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_energy_vs_K(
        state.data_label_list, state.K_values_list, state.energies_list, state.retained_dims_list,
        dmrg_energy,
        molecule=args.molecule, basis_set=args.basis_set, norb=norb,
        cluster_sizes="varies per curve -- see data_label",
        max_elec_transfers=args.max_elec_transfer, cost=args.cost_function,
        sectors_or_states="sectors", from_beam_search=True,
        min_child_cluster_size=args.min_child_cluster_size,
        target_num_clusters=args.target_num_clusters,
        initial_basis=args.initial_basis,
    )
    filename = f"exact_K_sector_analysis_{args.cost_function}_{timestamp}_{git_hash}.png"
    filepath = plots_dir / filename
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"K-sector-analysis plot saved to {filepath}")

# =============================================================================
# CLI Interface
# =============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Argument parser mirroring cluster_number_sector_search.py's own
    create_parser() (this module runs the same beam search to obtain a
    trajectory to search), minus --force-h1e/--force-full-rdms (meaningless
    here: exact weights need the ground-state MPS, not extra RDM orders),
    plus --n-td-steps for the one-time per-decomposition orbital rotation."""
    parser = argparse.ArgumentParser(
        description="Exact-weight cluster-number sector relevance ranking "
        "for a beam-search decomposition's trajectory."
    )
    parser.add_argument(
        "molecule", type=str,
        help="Molecule to analyze (one of h2, h2o, n2, lih, h4_linear, h4_square, h4_rectangle), "
        "or a free-form label when --fcidump is given",
    )
    parser.add_argument(
        "basis_set", type=str, help="Basis set (e.g., sto-3g, 6-31g), or a free-form label when --fcidump is given"
    )
    parser.add_argument(
        "bond_length", type=float, help="Bond length in Angstrom (unused, but still required, when --fcidump is given)"
    )
    parser.add_argument(
        "cost_function", type=str,
        help="Cost function type for the beam search (variance, eval_eq, extremality, mixed, commutator)",
    )

    parser.add_argument("--bond-angle", type=float, default=None, help="Bond angle in degrees (for H2O)")
    parser.add_argument(
        "--fcidump", type=str, default=None,
        help="Path to an FCIDUMP file with a precomputed Hamiltonian, as in cluster_number_decomposition_optimization.py",
    )

    # DMRG parameters
    parser.add_argument("--bond-dim", type=int, default=150, help="DMRG bond dimension (default: 150)")
    parser.add_argument("--n-sweeps", type=int, default=50, help="Number of DMRG sweeps (default: 50)")

    # Cost function parameters
    parser.add_argument("--var-exponent", type=int, default=1, help="Variance exponent (default: 1)")

    # Beam search hyperparameters
    parser.add_argument("--num-decos", type=int, default=4, help="Beam width (default: 4)")
    parser.add_argument("--num-subdecos", type=int, default=4, help="Splits attempted per deco per round (default: 4)")
    parser.add_argument("--min-parent-cluster-size", type=int, default=1)
    parser.add_argument("--min-child-cluster-size", type=int, default=1)
    parser.add_argument("--no-naturalize-children", action="store_true")
    parser.add_argument("--target-num-clusters", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--no-orb-opt-in-beam-search", action="store_true")

    # Initial basis
    parser.add_argument("--initial-basis", type=str, choices=["MOs", "NatOs", "both"], default="both")
    parser.add_argument("--fiedler-reorder", action="store_true")

    # Polish
    parser.add_argument("--no-polish", action="store_true")
    parser.add_argument("--polish-maxiter", type=int, default=500)

    # Sector search (new)
    parser.add_argument(
        "--analyze-num-clusters", type=int, nargs="+", default=None,
        help="Which trajectory cluster counts to run the sector search on (default: all with >= 2 clusters)",
    )
    parser.add_argument("--num-sectors-to-retain", type=int, default=None)
    parser.add_argument("--max-cum-dim-to-retain", type=int, default=None)
    parser.add_argument("--max-elec-transfer", type=int, default=2)
    parser.add_argument(
        "--n-td-steps", type=int, default=20,
        help="TDVP steps for the one-time per-decomposition orbital rotation (default: 20)",
    )
    parser.add_argument(
        "--K-sector-analysis", action="store_true",
        help="Produce one combined double plot (energy vs K, cumulative retained dimension vs K), "
        "covering every selected trajectory entry, where K is the number of top-ranked sectors "
        "retained per rank_relevant_sectors's own ranking (exact weight, not psi-weight): at each "
        "K, the energy is <psi'|H|psi'> for psi' = the normalized projection of the true "
        "wavefunction onto the direct sum of the top-K sectors' determinant supports. A horizontal "
        "chemical-accuracy line (from the DMRG energy) is drawn, and each entry's curve stops once "
        "it first crosses that line. Off by default.",
    )

    # Output options
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--plots-dir", type=str, default=None, help="Plots directory (for --K-sector-analysis)")
    parser.add_argument(
        "--wavefunction-dir", type=str, default="wavefunctions", help="MPS wavefunction directory (for input and output)"
    )

    # HPC options
    parser.add_argument("--n-threads", type=int, default=1, help="Number of threads (default: 1)")
    parser.add_argument("--no-reuse", action="store_true", help="Don't reuse existing wavefunction")

    # Logging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser


def _geometry_output_subpath(args: argparse.Namespace) -> Path:
    molecule = args.molecule.lower()
    basis_set = args.basis_set.lower()
    if args.fcidump is not None:
        return Path(molecule) / basis_set / f"fcidump_{Path(args.fcidump).stem}"
    bond_length_str = f"{args.bond_length:.4f}".replace(".", "_")
    path = Path(molecule) / basis_set / f"bond_{bond_length_str}"
    if args.bond_angle is not None:
        angle_str = f"{args.bond_angle:.4f}".replace(".", "_")
        path = path / f"angle_{angle_str}"
    return path


def _output_dir_for(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)
    return Path("outputs_") / "cluster_number_sector_search" / _geometry_output_subpath(args)


def _plots_dir_for(args: argparse.Namespace) -> Path:
    if args.plots_dir is not None:
        return Path(args.plots_dir)
    return Path("plots") / "cluster_number_sector_search" / _geometry_output_subpath(args)


def _sector_relevance_to_json(entries: list[SectorRelevance]) -> list[dict]:
    return [
        {
            "label": list(r.label),
            "weight": r.weight,
            "elec_transfer": r.elec_transfer,
            "dimension": r.dimension,
        }
        for r in entries
    ]


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    from cluster_number_decomposition_optimization import (
        DecompositionOptimizerConfig,
        _build_cost_function_constructor,
        _build_initial_bases,
        _run_dmrg_and_build_rdm_data,
        polish_decomposition,
        run_decomposition_optimizer,
    )
    from cluster_numbers_metrics import get_git_hash, get_timestamp

    _known_molecules = {"h2", "h2o", "n2", "lih", "h4_linear", "h4_square", "h4_rectangle"}
    if args.fcidump is None and args.molecule.lower() not in _known_molecules:
        logger.error(f"Unsupported molecule: {args.molecule}. Supported: {sorted(_known_molecules)}")
        exit(1)

    logger.info(f"Running beam search for {args.molecule}/{args.basis_set} to feed the sector search...")
    rdm_data, dmrg_metadata, solver, dmrg_energy, h1e, g2e, ecore, nelec = _run_dmrg_and_build_rdm_data(
        args, force_full_rdms=False,
    )
    norb = rdm_data.norb

    # See cluster_number_sector_search.py's own main() for why beam_rdm_data is
    # stripped down whenever the cost function doesn't need rdm3/rdm4/h1e/g2e_full
    # (rotate_rdm_data rotates every populated field unconditionally, and those
    # tensors are O(norb^6)/O(norb^8) -- expensive and wasted if unused). Unlike
    # that module, there is no separate --force-full-rdms/--force-h1e need here:
    # exact weights come from the MPS directly, not from any RDM order.
    if args.cost_function == "commutator":
        beam_rdm_data = rdm_data
    else:
        beam_rdm_data = RDMData(D=rdm_data.D, Gamma=rdm_data.Gamma)

    cost_function_constructor = _build_cost_function_constructor(args.cost_function, args.var_exponent)
    initial_bases = _build_initial_bases(args, beam_rdm_data, norb)

    opt_config = DecompositionOptimizerConfig(
        num_decos=args.num_decos, num_subdecos=args.num_subdecos,
        min_parent_cluster_size=args.min_parent_cluster_size,
        min_child_cluster_size=args.min_child_cluster_size,
        naturalize_children=not args.no_naturalize_children,
        target_num_clusters=args.target_num_clusters, max_rounds=args.max_rounds, maxiter=args.maxiter,
        optimize_rotation_in_beam_search=not args.no_orb_opt_in_beam_search,
    )
    trajectory = run_decomposition_optimizer(cost_function_constructor, beam_rdm_data, opt_config, initial_bases)

    if not args.no_polish:
        # Full joint-rotation reoptimization per fixed partition, exactly as
        # cluster_number_decomposition_optimization.py's own main() does --
        # rdm_data (not beam_rdm_data) since polishing runs once per
        # trajectory entry after the beam search entirely, not per-split, so
        # the O(norb^6)/O(norb^8) rdm3/rdm4 cost beam_rdm_data-stripping
        # avoids is not a concern here (and cost_function_constructor may
        # need them regardless, e.g. for "commutator").
        logger.info("Polishing each trajectory entry with a full joint-rotation reoptimization...")
        trajectory = [
            polish_decomposition(deco, rdm_data, cost_function_constructor, maxiter=args.polish_maxiter)
            for deco in trajectory
        ]

    if args.analyze_num_clusters is not None:
        wanted = set(args.analyze_num_clusters)
        entries = [d for d in trajectory if d.num_clusters in wanted]
        missing = wanted - {d.num_clusters for d in entries}
        if missing:
            logger.warning(f"--analyze-num-clusters {sorted(missing)} not present in trajectory; skipping.")
    else:
        entries = [d for d in trajectory if d.num_clusters >= 2]

    if not entries:
        logger.info("No trajectory entries selected for sector search (see --analyze-num-clusters).")
        return

    search_config = SectorSearchConfig(
        num_sectors_to_retain=args.num_sectors_to_retain,
        max_cum_dim_to_retain=args.max_cum_dim_to_retain,
        max_elec_transfer=args.max_elec_transfer,
        n_td_steps=args.n_td_steps,
    )

    output_dir = _output_dir_for(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = get_timestamp()
    git_hash = get_git_hash()

    if args.K_sector_analysis:
        K_sector_state = _setup_K_sector_analysis(solver, h1e, g2e, ecore, norb)

    for deco in entries:
        logger.info(f"Ranking sectors for {deco.num_clusters}-cluster decomposition (sizes={[len(c) for c in deco.partition]})...")
        ranked = rank_relevant_sectors(deco, rdm_data, nelec, solver, search_config)
        for r in ranked[:10]:
            logger.info(f"  label={r.label} weight={r.weight:.6f} t={r.elec_transfer} dim={r.dimension}")

        k_sector_summary = None
        if args.K_sector_analysis:
            k_sector_summary = _run_K_sector_analysis_for_entry(K_sector_state, deco, ranked, norb, nelec, dmrg_energy)

        metadata = {
            "molecule": args.molecule, "basis_set": args.basis_set,
            "bond_length": args.bond_length if args.fcidump is None else None,
            "bond_angle": args.bond_angle if args.fcidump is None else None,
            "fcidump": args.fcidump, "timestamp": timestamp, "git_hash": git_hash,
            "norb": norb, "nelec": [int(x) for x in nelec], "dmrg_energy": float(dmrg_energy),
            "cost": args.cost_function, "num_clusters": deco.num_clusters,
            "cluster_sizes": [len(c) for c in deco.partition],
            "max_elec_transfer": args.max_elec_transfer,
            "n_td_steps": args.n_td_steps,
            "num_sectors_to_retain": args.num_sectors_to_retain,
            "max_cum_dim_to_retain": args.max_cum_dim_to_retain,
            "polished": not args.no_polish,
        }
        output = {"metadata": metadata, "ranked_sectors": _sector_relevance_to_json(ranked)}
        if k_sector_summary is not None:
            output["K_sector_analysis"] = k_sector_summary
        filename = f"sectors_{deco.num_clusters}clusters_{args.cost_function}_{timestamp}_{git_hash}.json"
        filepath = output_dir / filename
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {filepath}")

    if args.K_sector_analysis:
        _finalize_K_sector_analysis(K_sector_state, args, dmrg_energy, norb, timestamp, git_hash)

    logger.info("Computation completed successfully!")


if __name__ == "__main__":
    main()
