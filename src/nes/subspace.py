from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Union

import numpy as np
import scipy as sp
import quantax as qtx
from quantax.utils import LogArray, PsiArray

from ..quantumhall_samplers import GetLzSymmetryProjector
from ..quantumhall_userbasis import LzUserBasisSymmetry
from .state_set import NaturalStateSet


def dense_reduced_matrices(
    state_set: NaturalStateSet,
    operator: qtx.operator.Operator,
    lz: Optional[int] = None,
    z2: Optional[int] = 0, 
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact dense ``(S, O)`` matrices in the span of ``states``,
    where S is the overlap matrix and O is a given operator."""
    dense_symm = LzUserBasisSymmetry(lz, z2)
    Nstates = state_set.Nstates

    eigs, _ = operator.diagonalize(symm=dense_symm, k = Nstates)
    print("Target eigenvalues (ED): ", np.sort(eigs))
    dense_states = [state.todense(dense_symm) for state in state_set.states]

    psi = []
    Opsi = []
    for dense in dense_states:
        psi.append(dense.psi)
        Opsi.append((operator @ dense).psi)

    overlap = np.empty((Nstates, Nstates))
    operator_matrix = np.empty((Nstates, Nstates))

    for i in range(Nstates):
        for j in range(Nstates):
            overlap[i, j] = np.vdot(psi[i], psi[j])
            operator_matrix[i, j] = np.vdot(psi[i], Opsi[j])

    return overlap, operator_matrix


def _slot_amplitudes(psi_matrix: PsiArray, weights: Optional[np.ndarray]):
    r"""The tuples as one row per slot: ``X[r, a]`` is :math:`\psi_a(x_r)` rescaled by a factor per row (its largest
    amplitude) and one per column, the same for every row (the median over the rows), in float64; ``w`` the tuple's
    reweight factor repeated over its slots, and ``ok`` the rows with a finite amplitude and a positive weight."""
    A = LogArray.from_value(psi_matrix)
    nsamples, K = psi_matrix.shape[0], psi_matrix.shape[-1]
    logabs = np.asarray(A.logabs, np.float64).reshape(-1, K)
    sign = np.asarray(A.sign).reshape(-1, K)
    w = np.repeat(np.ones(nsamples) if weights is None else np.asarray(weights, np.float64), K)

    with np.errstate(invalid="ignore"):         # a row of zero amplitudes gives -inf - -inf; `ok` drops it
        rel = logabs - np.max(logabs, axis=1, keepdims=True)
    ok = np.isfinite(rel).any(axis=1) & np.isfinite(w) & (w > 0)
    rel = np.where(np.isfinite(rel), rel, -np.inf)
    col = np.array([np.median(c[np.isfinite(c)]) for c in rel[ok].T])
    return sign * np.exp(rel - col), w, ok


def _tyler_scatter(X: np.ndarray, w: np.ndarray, iters: int, tol: float) -> np.ndarray:
    r"""The fixed point :math:`S = K\,E_w[X^* X^T / \rho]`, :math:`\rho = X^T S^{-1} X^*`, from :math:`S = 1`. Only
    its scale is free; the trace is kept at K while iterating, and at the fixed point the equation holds with no extra
    factor (:math:`\mathrm{tr}(S^{-1} K E[X^* X^T/\rho]) = K` term by term)."""
    K = X.shape[1]
    S = np.eye(K)
    for _ in range(iters):
        rho = np.real(np.einsum("ra,ab,rb->r", X, np.linalg.inv(S), X.conj()))
        S_new = K * np.einsum("r,ra,rb->ab", w / rho, X.conj(), X) / w.sum()
        S_new = S_new / (np.real(np.trace(S_new)) / K)
        converged = np.max(np.abs(S_new - S)) < tol
        S = S_new
        if converged:
            break
    return S


def _normalised(S: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.real(np.diag(S)))
    return np.real_if_close(S / np.outer(d, d))


def sampled_overlap(
    psi_matrix: PsiArray,
    weights: Optional[np.ndarray] = None,
    iters: int = 200,
    tol: float = 1e-10,
) -> np.ndarray:
    r"""The members' normalised overlap matrix :math:`G_{ab} = S_{ab} / \sqrt{S_{aa} S_{bb}}` from NES tuples, with
    no evaluation of its own: ``psi_matrix[t, i, a]`` is :math:`\psi_a(x_{ti})`, the matrix the local energies are
    built from, and ``weights`` the reweight factors that take the tuples to :math:`|\det A|^2` (soft-min's
    :math:`1/\sum_i u_i^{-p}` as much as ``reweight``'s :math:`|\det A|^{2-r}`).

    Under :math:`|\det A|^2` each slot of a tuple is distributed as :math:`\rho(x)/K`, with
    :math:`\rho(x) = \psi(x)^T S^{-1} \psi(x)^*` the one-body density of a determinant of non-orthogonal orbitals.
    Hence :math:`S = K\,E_\mathrm{slots}[\psi^* \psi^T / \rho]`: Tyler's M-estimator of scatter, solved by fixed-point
    iteration from :math:`S = 1`, unique up to a scale that :math:`G` drops. Each term has
    :math:`\psi^T S^{-1} \psi^* / \rho = 1`, so its noise comes from the weights alone. Each term is of degree 0 in its
    row, so the rows are rescaled freely; the columns get one factor each, the same for every tuple (not the
    per-tuple column shift of the local energies, which would mix the members' scales between tuples).
    """
    X, w, ok = _slot_amplitudes(psi_matrix, weights)
    return _normalised(_tyler_scatter(X[ok], w[ok], iters, tol))


class SubspaceEstimate(NamedTuple):
    """`sampled_subspace`: the members' normalised overlaps, and the Ritz states' energies (ascending), energy
    variances, effective numbers of slots and the part of each variance the tail sampling added (zeros without
    ``tail_g``): ``variances - tail_var`` is the variance of the state itself."""
    overlap: np.ndarray
    energies: np.ndarray
    variances: np.ndarray
    n_eff: np.ndarray
    tail_var: np.ndarray


def sampled_subspace(
    psi_matrix: PsiArray,
    local_matrix: np.ndarray,
    weights: Optional[np.ndarray] = None,
    tail_g: Optional[np.ndarray] = None,
    iters: int = 200,
    tol: float = 1e-10,
) -> SubspaceEstimate:
    r"""The Ritz states of the span -- the eigenstates of H projected on it, :math:`\phi_n = \sum_a c_{an}\psi_a` --
    with their energies and energy variances, from the NES tuples and nothing else: ``psi_matrix[t, i, a]`` is
    :math:`\psi_a(x_{ti})`, ``local_matrix[t, i, a]`` the local energy :math:`(H\psi_a)(x_{ti})/\psi_a(x_{ti})` the
    trace was built from, ``weights`` the reweight factors to :math:`|\det A|^2`. No amplitude is evaluated.

    As in `sampled_overlap`, the slots are distributed as :math:`\rho/K`, :math:`\rho = \psi^T S^{-1}\psi^*`, with S
    Tyler's estimate. On the same slots and weights :math:`H_{ab} = K\,E[\psi_a^* (H\psi_b)/\rho]`, and
    :math:`H c = E S c` gives the Ritz values and vectors. Since :math:`\rho = \sum_n |\phi_n|^2` for the
    S-orthonormal :math:`\phi_n`, reweighting a slot to :math:`|\phi_n|^2` takes the factor
    :math:`|\phi_n|^2/\rho \le 1`, so

    .. math::

        \mathrm{Var}_n = \frac{E[\,|(H\phi_n) - E_n\phi_n|^2/\rho\,]}{E[\,|\phi_n|^2/\rho\,]},

    written without dividing by :math:`\phi_n`, which may vanish on a slot. Both are of degree 0 in each row, so the
    row scaling of `_slot_amplitudes` drops out, and :math:`H\psi` takes the same scalings as :math:`\psi`. If the
    span is invariant under H, :math:`H\psi = \psi M` on every slot, so :math:`H = SM` holds exactly on the sample
    and every variance vanishes: the zero-variance property, state by state. ``n_eff`` is Kish's count of the slot
    weights :math:`w|\phi_n|^2/\rho`, at most ``K * nsamples``; slots of one tuple are not independent, so it
    is an upper bound.

    With the tail sampled the local energies are noisy, and each variance includes what the sampling adds. The draw
    on a slot is shared by the members, so :math:`(H\phi_n)(x)` is the sampled estimator applied to
    :math:`\phi_n`, and its added variance averages, on relabelling, to the :math:`|\phi_n|^2`-weighted mean of
    the per-slot g (``tail_g[t, i]``, the H entry of `PooledOperator`'s ``tail_variance``): returned as
    ``tail_var``, with the same slot weights :math:`w|\phi_n|^2/\rho`.
    """
    X, w, ok = _slot_amplitudes(psi_matrix, weights)
    K = X.shape[1]
    local = np.asarray(local_matrix).reshape(-1, K)
    with np.errstate(invalid="ignore", over="ignore"):
        Y = X * local                               # (H psi_a)(x_r), scaled as X; nan where psi_a vanishes
    ok &= np.isfinite(Y).all(axis=1)
    X, Y, w = X[ok], Y[ok], w[ok]
    g = None if tail_g is None else np.real(np.asarray(tail_g)).astype(np.float64).reshape(-1)[ok]

    S = _tyler_scatter(X, w, iters, tol)
    rho = np.real(np.einsum("ra,ab,rb->r", X, np.linalg.inv(S), X.conj()))
    H = K * np.einsum("r,ra,rb->ab", w / rho, X.conj(), Y) / w.sum()
    # not symmetrised: on a finite sample S M is not Hermitian, and symmetrising would break H = S M on an invariant
    # span. Unsymmetrised, H c = E S c makes E_n exactly the slot-weighted mean of phi_n's local energy.
    energies, C = sp.linalg.eig(H, S)
    order = np.argsort(energies.real)
    energies, C = energies[order], C[:, order]
    phi, Hphi = X @ C, Y @ C
    v = w[:, None] * np.abs(phi) ** 2 / rho[:, None]
    variances = np.sum(w[:, None] * np.abs(Hphi - energies * phi) ** 2 / rho[:, None], axis=0) / v.sum(axis=0)
    n_eff = v.sum(axis=0) ** 2 / np.sum(v ** 2, axis=0)
    tail_var = np.zeros(K) if g is None else g @ v / v.sum(axis=0)
    return SubspaceEstimate(_normalised(S), energies.real, variances, n_eff, tail_var)