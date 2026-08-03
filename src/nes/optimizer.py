from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Callable, Optional, Union

import jax
import jax.numpy as jnp
import quantax as qtx
from quantax.global_defs import get_default_dtype
from quantax.sampler import Samples
from quantax.utils import LogArray

from .state_set import NaturalStateSet


def _scaled_psi_matrix(
    states: NaturalStateSet,
    tuple_spins: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    r"""
    Materialize the NES matrix with per-row stabilization.

    If ``A[i, a] = psi_a(s_i)`` and ``D[i, i] = exp(row_shift[i])``, this returns
    ``A_scaled = D^{-1} A``. The same row scaling must be used for ``B`` in the
    local-energy trace, so that ``solve(A_scaled, D^{-1} B) = A^{-1} B``.
    """
    psi_matrix = LogArray.from_value(states.psi_matrix(tuple_spins))
    row_shift = jnp.max(psi_matrix.logabs, axis=-1)
    row_shift = jnp.where(jnp.isfinite(row_shift), row_shift, 0.0)
    scaled = psi_matrix.sign * jnp.exp(psi_matrix.logabs - row_shift[..., None])
    return scaled, row_shift


def _pinv_scaled_matrix(A_scaled: jax.Array) -> jax.Array:
    real_dtype = jnp.finfo(A_scaled.real.dtype).dtype
    rtol = jnp.asarray(jnp.finfo(real_dtype).eps ** 0.5, dtype=real_dtype)
    return jnp.linalg.pinv(A_scaled, rtol=rtol)


class NaturalTraceEnergyGrad(qtx.optimizer.EnergyGrad):
    """"""
    def local_energy(self, states: NaturalStateSet, samples: Samples) -> jax.Array:
        tuple_spins = jnp.asarray(samples.spins)
        nsamples, Nstates, Nmodes = tuple_spins.shape
        if Nstates != states.Nstates or Nmodes != states.Nmodes:
            raise ValueError(
                "Expected sample spins with shape "
                f"(nsamples, {states.Nstates}, {states.Nmodes}), got "
                f"{tuple_spins.shape}."
            )

        flat_spins = tuple_spins.reshape(nsamples * Nstates, Nmodes)
        S_scaled, row_shift = _scaled_psi_matrix(states, tuple_spins)

        H_columns = []
        for state in states.states:
            psi = jnp.asarray(state(flat_spins))
            Eloc = self.hamiltonian.Oloc(state, flat_spins).astype(get_default_dtype())
            H_columns.append(Eloc * psi)

        H = jnp.stack(H_columns, axis=-1).reshape(nsamples, Nstates, Nstates)
        H_scaled = H * jnp.exp(-row_shift[..., None])
        Sinv_H = jnp.linalg.pinv(S_scaled) @ H_scaled
        return jnp.trace(Sinv_H, axis1=-2, axis2=-1).astype(get_default_dtype())

    def ebar(self, states: NaturalStateSet, samples: Samples) -> jax.Array:
        Eloc = self.local_energy(states, samples)

        if samples.reweight_factor is None:
            reweight_factor = jnp.ones(samples.nsamples)
        else:
            reweight_factor = samples.reweight_factor

        Emean = jnp.mean(Eloc * reweight_factor)
        self._energy = Emean.real
        Evar = jnp.abs(Eloc - Emean) ** 2
        self._VarE = jnp.mean(Evar * reweight_factor).real

        Eloc -= jnp.mean(Eloc)
        Eloc *= jnp.sqrt(reweight_factor / samples.nsamples)
        return Eloc


class NaturalExcitedAdamSR(qtx.optimizer.StochasticQNGD):
    r"""
    AdamSR optimizer for natural excited state determinants.

    The Quantax stochastic optimizer stack still owns the SR equation solve, Adam
    buffers, solver defaults, checkpointing, and non-holomorphic bookkeeping. This
    class provides only the NES-specific centered local-energy vector and
    determinant logarithmic Jacobian.
    """

    def __init__(
        self,
        states: NaturalStateSet,
        hamiltonian: qtx.operator.Operator,
        imag_time: bool = True,
        solver: Optional[Callable[[jax.Array, jax.Array], jax.Array]] = None,
        mu: float = 0.95,
        beta: float = 0.995,
        norm_clip: Optional[float] = None,
        file: Union[None, str, Path, BinaryIO] = None,
    ):
        if not isinstance(states, NaturalStateSet):
            raise TypeError("NaturalExcitedAdamSR expects a NaturalStateSet.")
        if states.nparams is None:
            raise TypeError(
                "NaturalExcitedAdamSR requires all trainable member states to expose "
                "`nparams`."
            )
        for index, state in enumerate(states.states):
            # Frozen states never contribute a Jacobian block, so they are allowed to
            # be any state type; everything else must be consistent across the set,
            # since all members share the psi matrix.
            if states.trainable[index] and not hasattr(state, "jacobian"):
                raise TypeError(
                    "NaturalExcitedAdamSR requires trainable member states with "
                    f"`jacobian`; state {index} has none."
                )
            if not hasattr(state, "vs_type"):
                raise TypeError(
                    "NaturalExcitedAdamSR requires member states with Quantax "
                    f"`vs_type`; state {index} has none."
                )
            if state.vs_type != states.vs_type:
                raise ValueError(
                    "All member states must have the same `vs_type`; "
                    f"state {index} differs from state 0."
                )
            if state.dtype != states.dtype:
                raise ValueError(
                    "All member states must have the same parameter dtype; "
                    f"state {index} differs from state 0."
                )

        grad = NaturalTraceEnergyGrad(hamiltonian)
        updater = qtx.optimizer.Adam(mu, beta, norm_clip)
        super().__init__(
            states,
            grad,
            imag_time=imag_time,
            solver=solver,
            updater=updater,
            file=file,
        )
        self._Omean = None

    @property
    def states(self) -> NaturalStateSet:
        return self.state

    @property
    def Omean(self) -> Optional[jax.Array]:
        return self._Omean

    def get_Ebar(self, samples: Samples) -> jax.Array:
        r"""Compute Quantax-style centered/scaled local energies for the NES determinant."""
        return self._grad.ebar(self.states, samples)

    def get_Obar(self, samples: Samples) -> jax.Array:
        r"""
        Compute determinant logarithmic Jacobians in member-state parameter order.

        For :math:`A[i, a] = \psi_a(s_i)`, the derivative with respect to state ``a``'s
        parameters is :math:`\sum_i (A^{-1})_{ai} A_{ia} O_a(s_i)`, with :math:`O_a` the
        state's own logarithmic Jacobian. Only trainable states contribute a column
        block, but the :math:`A^{-1}` prefactor is built from the *full*
        ``Nstates`` x ``Nstates`` matrix -- the members are coupled through the
        determinant, so a frozen state still shapes the coefficients of the trainable
        blocks. Freezing therefore drops columns from :math:`\bar O` without altering
        the ansatz or approximating the remaining gradients.
        """
        tuple_spins = jnp.asarray(samples.spins)
        nsamples, Nstates, Nmodes = tuple_spins.shape
        if Nstates != self.states.Nstates or Nmodes != self.states.Nmodes:
            raise ValueError(
                "Expected sample spins with shape "
                f"(nsamples, {self.states.Nstates}, {self.states.Nmodes}), got "
                f"{tuple_spins.shape}."
            )

        A_scaled, _ = _scaled_psi_matrix(self.states, tuple_spins)
        Ainv = _pinv_scaled_matrix(A_scaled)

        # `jacobians` skips frozen states, so the yield order is not the member-state
        # order: pair each Jacobian with its own index in the full set.
        trainable_indices = [
            index for index, flag in enumerate(self.states.trainable) if flag
        ]

        blocks = []
        for index, jacobian in zip(trainable_indices, self.states.jacobians(tuple_spins)):
            coeff = Ainv[:, index, :] * A_scaled[:, :, index]
            blocks.append(
                jnp.einsum(
                    "ni,nip->np", coeff, jacobian.reshape(nsamples, Nstates, -1)
                )
            )
            # Drop the reference before the generator computes the next Jacobian,
            # otherwise two of them overlap on device.
            del jacobian

        Omat = jnp.concatenate(blocks, axis=-1)
        del blocks
        if samples.reweight_factor is None:
            reweight = jnp.ones(samples.nsamples)
        else:
            reweight = samples.reweight_factor
        self._Omean = jnp.mean(Omat * reweight[:, None], axis=0)
        factor = jnp.sqrt(reweight / samples.nsamples)[:, None]
        return (Omat - jnp.mean(Omat, axis=0, keepdims=True)) * factor
