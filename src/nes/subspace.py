from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import quantax as qtx

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