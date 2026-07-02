from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import quantax as qtx
from quantax.utils import LogArray, PsiArray, ScaleArray


def _same_symmetry(left: qtx.symmetry.Symmetry, right: qtx.symmetry.Symmetry) -> bool:
    if type(left) is not type(right):
        return False

    attrs = (
        "Nsites",
        "Nmodes",
        "Nparticles",
        "particle_type",
        "double_occ",
        "Z2_inversion",
    )
    if any(getattr(left, attr) != getattr(right, attr) for attr in attrs):
        return False

    if tuple(left._sector) != tuple(right._sector):
        return False

    array_attrs = ("_generator", "_generator_sign", "_perm", "_character", "_perm_sign")
    if any(
        not np.array_equal(
            np.asarray(getattr(left, attr)), np.asarray(getattr(right, attr))
        )
        for attr in array_attrs
    ):
        return False

    return getattr(left, "_ph_sign", None) == getattr(right, "_ph_sign", None)


def _stack_psi_arrays(values: Sequence[PsiArray], axis: int = -1) -> PsiArray:
    """Stack Quantax wavefunction arrays without unnecessarily materializing them."""
    # TODO: Revisit this representation once NES determinant/logdet helpers exist.
    # We may want determinant code to own the conversion from per-state PsiArrays to
    # a stabilized matrix representation.
    if any(isinstance(value, ScaleArray) for value in values):
        values = [ScaleArray.from_value(value) for value in values]
        return ScaleArray(
            jnp.stack([value.significand for value in values], axis=axis),
            jnp.stack(
                [
                    jnp.broadcast_to(value.exponent, value.significand.shape)
                    for value in values
                ],
                axis=axis,
            ),
        )

    if any(isinstance(value, LogArray) for value in values):
        values = [LogArray.from_value(value) for value in values]
        return LogArray(
            jnp.stack([value.sign for value in values], axis=axis),
            jnp.stack([value.logabs for value in values], axis=axis),
        )

    return jnp.stack([jnp.asarray(value) for value in values], axis=axis)


class NaturalStateSet:
    r"""
    Container for the states used to build a natural excited state ansatz.

    Contained states may be any Quantax/qhuantax state, for example ``Variational``,
    ``DenseState``, ``OperatedState``, or future subclasses.

    For configurations ``s_i`` and member states ``psi_a``, the central object is the
    matrix

    ``A[i, a] = psi_a(s_i)``.

    """

    def __init__(self, states: Iterable[qtx.state.State]):
        self._states = tuple(states)
        if len(self._states) <= 1:
            raise ValueError("NaturalStateSet requires at least two states.")

        symm = self._states[0].symm
        for index, state in enumerate(self._states[1:], start=1):
            if not _same_symmetry(symm, state.symm):
                raise ValueError(
                    "All states in a NaturalStateSet must have the same symmetry; "
                    f"state {index} differs from state 0."
                )

    @property
    def states(self) -> tuple[qtx.state.State, ...]:
        return self._states

    @property
    def Nstates(self) -> int:
        return len(self._states)

    @property
    def Nsites(self) -> int:
        return self._states[0].Nsites

    @property
    def Nmodes(self) -> int:
        return self._states[0].Nmodes

    @property
    def Nparticles(self):
        return self._states[0].Nparticles

    @property
    def symm(self) -> qtx.symmetry.Symmetry:
        """Reference symmetry, taken from the first state."""
        return self._states[0].symm

    @property
    def nparams_per_state(self) -> tuple[Optional[int], ...]:
        return tuple(getattr(state, "nparams", None) for state in self._states)

    @property
    def nparams(self) -> Optional[int]:
        counts = self.nparams_per_state
        if any(count is None for count in counts):
            return None
        return int(sum(counts))

    def amplitudes(self, spins: Union[jnp.ndarray, jax.Array]) -> PsiArray:
        r"""
        Evaluate every member state on the same configurations.

        Parameters
        ----------
        spins:
            Fock states with shape ``(..., Nmodes)``.

        Returns
        -------
        PsiArray
            Wavefunction values with shape ``(..., Nstates)``.
        """
        spins = jnp.asarray(spins)
        batch_shape = spins.shape[:-1]
        flat_spins = spins.reshape(-1, self.Nmodes)
        values = [state(flat_spins) for state in self._states]
        values = _stack_psi_arrays(values, axis=-1)
        return values.reshape((*batch_shape, self.Nstates))

    __call__ = amplitudes

    def psi_matrix(self, tuple_spins: Union[jnp.ndarray, jax.Array]) -> PsiArray:
        r"""
        Evaluate the NES state-value matrices.

        Parameters
        ----------
        tuple_spins:
            Tuples of configurations with shape ``(..., Nstates, Nmodes)``.

        Returns
        -------
        PsiArray
            Matrices with shape ``(..., Nstates, Nstates)`` where the row index is
            the configuration inside the tuple and the column index is the member
            state.
        """
        tuple_spins = jnp.asarray(tuple_spins)
        batch_shape = tuple_spins.shape[:-2]
        values = self.amplitudes(tuple_spins.reshape(-1, self.Nmodes))
        return values.reshape((*batch_shape, self.Nstates, self.Nstates))

    def jacobians(self, spins: Union[jnp.ndarray, jax.Array]) -> tuple[jax.Array, ...]:
        r"""
        Evaluate per-state logarithmic Jacobians on the same configurations.

        This is available only for member states that implement Quantax's
        ``jacobian`` method, such as ``qtx.state.Variational`` and compatible
        qhuantax subclasses.
        """
        spins = jnp.asarray(spins)
        flat_spins = spins.reshape(-1, self.Nmodes)
        return tuple(state.jacobian(flat_spins) for state in self._states)

    def update(self, steps: Sequence[jax.Array]) -> None:
        """Apply one parameter update to each updatable member state."""
        for state, step in zip(self._states, steps):
            state.update(step)

    def split_step(self, step: jax.Array) -> tuple[jax.Array, ...]:
        """Split a concatenated NES update into per-state parameter updates."""
        counts = self.nparams_per_state
        if any(count is None for count in counts):
            raise TypeError(
                "Cannot split a global step because at least one state has no "
                "`nparams` attribute."
            )

        step = jnp.asarray(step)
        total = sum(counts)
        if step.shape[-1] != total:
            raise ValueError(
                f"Expected a flat update with length {total}, got {step.shape[-1]}."
            )

        split_points = jnp.cumsum(jnp.asarray(counts[:-1])).tolist()
        return tuple(jnp.split(step, split_points, axis=-1))
