from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Optional, Union

import jax
import jax.numpy as jnp
import quantax as qtx
from quantax.utils import LogArray, PsiArray, ScaleArray


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

    Individual member states may be marked non-trainable via ``trainable``. A frozen
    state remains a full member of the ansatz -- it contributes to `amplitudes`,
    `psi_matrix`, and hence to the local-energy trace -- but its parameters are excluded
    from `jacobians`, `nparams`, `split_step`, and `update`. The optimizer therefore
    never allocates buffers for or solves for them. This is the intended way to reuse a
    converged single-state result (loaded via ``param_file`` on
    `quantax.state.Variational`) as a fixed member of the set.
    """

    def __init__(
        self,
        states: Iterable[qtx.state.State],
        trainable: Optional[Sequence[bool]] = None,
    ):
        r"""
        Parameters
        ----------
        states:
            The member states. At least two are required, and all must share the same
            symmetry.

        trainable:
            One flag per member state, marking whether its parameters are optimized.
            Defaults to all states being trainable. At least one state must be
            trainable.
        """
        self._states = tuple(states)
        if len(self._states) <= 1:
            raise ValueError("NaturalStateSet requires at least two states.")

        if trainable is None:
            self._trainable = (True,) * len(self._states)
        else:
            trainable = tuple(bool(flag) for flag in trainable)
            if len(trainable) != len(self._states):
                raise ValueError(
                    f"Expected one `trainable` flag per state ({len(self._states)}), "
                    f"got {len(trainable)}."
                )
            if not any(trainable):
                raise ValueError(
                    "At least one state in a NaturalStateSet must be trainable."
                )
            self._trainable = trainable

    @property
    def states(self) -> tuple[qtx.state.State, ...]:
        return self._states

    @property
    def trainable(self) -> tuple[bool, ...]:
        """Whether each member state's parameters are optimized."""
        return self._trainable

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
    def dtype(self):
        return self._states[0].dtype

    @property
    def vs_type(self):
        return self._states[0].vs_type

    @property
    def nparams_per_state(self) -> tuple[Optional[int], ...]:
        """Parameter count of every member state, trainable or not."""
        return tuple(getattr(state, "nparams", None) for state in self._states)

    def _trainable_counts(self) -> tuple[Optional[int], ...]:
        """Parameter counts of the trainable member states, in member-state order."""
        return tuple(
            count
            for count, flag in zip(self.nparams_per_state, self._trainable)
            if flag
        )

    @property
    def nparams(self) -> Optional[int]:
        r"""
        Total number of *optimized* parameters, i.e. summed over trainable states only.

        This is the width of the NES update vector: it sets the column count of the
        Jacobian assembled from `jacobians`, the length accepted by `split_step`, and
        the size of the optimizer buffers allocated by Quantax's ``QNGD``. Frozen states
        are excluded; use `nparams_per_state` for the per-state totals.
        """
        counts = self._trainable_counts()
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

    def jacobians(self, spins: Union[jnp.ndarray, jax.Array]) -> Iterator[jax.Array]:
        r"""
        Evaluate logarithmic Jacobians of the trainable states on the same
        configurations.

        This is available only for member states that implement Quantax's
        ``jacobian`` method, such as ``qtx.state.Variational`` and compatible
        qhuantax subclasses.

        Yields
        ------
        jax.Array
            One Jacobian per trainable state, in member-state order. Frozen states are
            skipped, so the yield position is *not* the state's index within the full
            set -- callers that need to address the corresponding row/column of the NES
            matrix must map it back through `trainable`.

        Notes
        -----
        Yielded lazily: each Jacobian is ``(nsamples * Nstates, nparams)``, so
        returning them as a tuple keeps all ``Nstates`` of them resident at once.
        """
        spins = jnp.asarray(spins)
        flat_spins = spins.reshape(-1, self.Nmodes)
        for state, flag in zip(self._states, self._trainable):
            if flag:
                yield state.jacobian(flat_spins)

    def update(self, steps: Sequence[jax.Array]) -> None:
        """Apply one parameter update to each trainable member state."""
        steps = tuple(steps)
        ntrainable = sum(self._trainable)
        if len(steps) != ntrainable:
            raise ValueError(
                f"Expected one update per trainable state ({ntrainable}), "
                f"got {len(steps)}."
            )

        step_iter = iter(steps)
        for state, flag in zip(self._states, self._trainable):
            if flag:
                state.update(next(step_iter))

    def split_step(self, step: jax.Array) -> tuple[jax.Array, ...]:
        """Split a concatenated NES update into per-trainable-state parameter updates."""
        counts = self._trainable_counts()
        if any(count is None for count in counts):
            raise TypeError(
                "Cannot split a global step because at least one trainable state has "
                "no `nparams` attribute."
            )

        step = jnp.asarray(step)
        total = sum(counts)
        if step.shape[-1] != total:
            raise ValueError(
                f"Expected a flat update with length {total}, got {step.shape[-1]}."
            )

        split_points = jnp.cumsum(jnp.asarray(counts[:-1])).tolist()
        return tuple(jnp.split(step, split_points, axis=-1))
