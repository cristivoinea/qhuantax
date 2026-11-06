from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
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
    A = LogArray.from_value(psi_matrix)
    nsamples, K = psi_matrix.shape[0], psi_matrix.shape[-1]
    logabs = np.asarray(A.logabs, np.float64).reshape(-1, K)
    sign = np.asarray(A.sign).reshape(-1, K)
    w = np.repeat(np.ones(nsamples) if weights is None else np.asarray(weights, np.float64), K)

    rel = logabs - np.max(logabs, axis=1, keepdims=True)
    ok = np.isfinite(rel).any(axis=1) & np.isfinite(w) & (w > 0)
    rel = np.where(np.isfinite(rel), rel, -np.inf)
    col = np.array([np.median(c[np.isfinite(c)]) for c in rel[ok].T])
    X = sign[ok] * np.exp(rel[ok] - col)
    w = w[ok]

    S = np.eye(K)
    for _ in range(iters):
        rho = np.real(np.einsum("ra,ab,rb->r", X, np.linalg.inv(S), X.conj()))
        S_new = K * np.einsum("r,ra,rb->ab", w / rho, X.conj(), X) / w.sum()
        S_new = S_new / (np.real(np.trace(S_new)) / K)
        converged = np.max(np.abs(S_new - S)) < tol
        S = S_new
        if converged:
            break
    d = np.sqrt(np.real(np.diag(S)))
    return np.real_if_close(S / np.outer(d, d))