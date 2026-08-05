import numpy
import ffsim
from pyscf import ao2mo, lib
from pyscf.lib import logger
from pyscf.fci import direct_spin1
from pyscf.fci.selected_ci import (
    _unpack, _all_linkstr_index, _as_SCIvector, SCIvector
)


def kernel_fixed_list(myci, h1e, eri, norb, nelec, det_list, ci0=None,
                       tol=None, lindep=None, max_cycle=None, max_space=None,
                       nroots=None, max_memory=None, verbose=None, ecore=0,
                       **kwargs):
    '''Like selected_ci.kernel_fixed_space, but `det_list` is the literal
    list of determinants to keep -- a list/array of (stra, strb) integer
    pairs -- rather than the full direct product of separate alpha and
    beta string lists.
    '''
    log = logger.new_logger(myci, verbose)
    if tol is None: tol = myci.conv_tol
    if lindep is None: lindep = myci.lindep
    if max_cycle is None: max_cycle = myci.max_cycle
    if max_space is None: max_space = myci.max_space
    if max_memory is None: max_memory = myci.max_memory
    if nroots is None: nroots = myci.nroots

    nelec = direct_spin1._unpack_nelec(nelec, myci.spin)

    det_list = numpy.asarray(det_list, dtype=numpy.int64).reshape(-1, 2)
    strsa = numpy.asarray(sorted(set(det_list[:, 0])), dtype=numpy.int64)
    strsb = numpy.asarray(sorted(set(det_list[:, 1])), dtype=numpy.int64)
    ci_strs = (strsa, strsb)
    na, nb = len(strsa), len(strsb)

    # mask over the na x nb bounding box: True only at requested determinants
    ia = numpy.searchsorted(strsa, det_list[:, 0])
    ib = numpy.searchsorted(strsb, det_list[:, 1])
    mask = numpy.zeros((na, nb), dtype=bool)
    mask[ia, ib] = True

    def project(c):
        c = c.reshape(na, nb).copy()
        c[~mask] = 0
        return c.ravel()

    h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec, .5)
    h2e = ao2mo.restore(1, h2e, norb)
    link_index = _all_linkstr_index(ci_strs, norb, nelec)

    hdiag = myci.make_hdiag(h1e, eri, ci_strs, norb, nelec, compress=True)
    hdiag = hdiag.reshape(na, nb)
    hdiag[~mask] = 1e9          # keep solver/init-guess away from non-list cells
    hdiag = hdiag.ravel()

    if isinstance(ci0, SCIvector):
        ci0 = [ci0.ravel()] if ci0.size == na*nb else [x.ravel() for x in ci0]
    elif ci0 is None:
        ci0 = myci.get_init_guess(ci_strs, norb, nelec, nroots, hdiag)
        ci0 = [x.ravel() for x in ci0]
    ci0 = [project(x) for x in ci0]

    cpu0 = [logger.process_clock(), logger.perf_counter()]
    def hop(c):
        c = project(c)
        hc = myci.contract_2e(h2e, _as_SCIvector(c.reshape(na, nb), ci_strs),
                               norb, nelec, link_index)
        cpu0[:] = log.timer_debug1('contract_2e', *cpu0)
        return project(hc.reshape(-1))

    precond = lambda x, e, *args: x/(hdiag - e + 1e-4)

    e, c = myci.eig(hop, ci0, precond, tol=tol, lindep=lindep,
                     max_cycle=max_cycle, max_space=max_space, nroots=nroots,
                     max_memory=max_memory, verbose=log, **kwargs)

    if nroots > 1:
        return e + ecore, [_as_SCIvector(ci.reshape(na, nb), ci_strs)[ia, ib] for ci in c]
    else:
        return e + ecore, _as_SCIvector(c.reshape(na, nb), ci_strs)[ia, ib]


def fci_addresses_to_sci_dets(bitstrings, norb, nelec):
    alphas, betas = ffsim.addresses_to_strings(bitstrings, norb, nelec,
                                               concatenate=False)

    return [[a, b] for a, b in zip(alphas, betas)]