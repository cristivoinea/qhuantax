from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Callable, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import quantax as qtx
from quantax.global_defs import get_default_dtype
from quantax.optimizer.solver import auto_pinv_eig
from quantax.sampler import Samples
from quantax.state import VS_TYPE
from quantax.utils import LogArray, get_replicate_sharding

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


class NaturalExcitedAdamSR:
    r"""
    AdamSR optimizer for natural excited state determinants.

    This class follows Quantax's ``AdamSR`` solve pattern, but computes the SR
    right-hand side and logarithmic Jacobian for
    ``Psi_NES(S) = det(psi_a(s_i))`` instead of a single Quantax state.
    """

    def __init__(
        self,
        states: NaturalStateSet,
        hamiltonian: qtx.operator.Operator,
        imag_time: bool = True,
        solver: Optional[Callable[[jax.Array, jax.Array], jax.Array]] = None,
        mu: float = 0.95,
        beta: float = 0.995,
        file: Union[None, str, Path, BinaryIO] = None,
    ):
        if not isinstance(states, NaturalStateSet):
            raise TypeError("NaturalExcitedAdamSR expects a NaturalStateSet.")
        if states.nparams is None:
            raise TypeError(
                "NaturalExcitedAdamSR requires all member states to expose `nparams`."
            )
        for index, state in enumerate(states.states):
            if not hasattr(state, "jacobian"):
                raise TypeError(
                    "NaturalExcitedAdamSR requires variational member states with "
                    f"`jacobian`; state {index} has none."
                )
            if not hasattr(state, "vs_type"):
                raise TypeError(
                    "NaturalExcitedAdamSR requires member states with Quantax "
                    f"`vs_type`; state {index} has none."
                )

        first = states.states[0]
        self._vs_type = first.vs_type
        self._dtype = first.dtype
        for index, state in enumerate(states.states[1:], start=1):
            if state.vs_type != self._vs_type:
                raise ValueError(
                    "All member states must have the same `vs_type`; "
                    f"state {index} differs from state 0."
                )
            if state.dtype != self._dtype:
                raise ValueError(
                    "All member states must have the same parameter dtype; "
                    f"state {index} differs from state 0."
                )

        self._states = states
        self._hamiltonian = hamiltonian
        self._imag_time = imag_time
        self._solver = auto_pinv_eig() if solver is None else solver
        self._mu = mu
        self._beta = beta

        sharding = get_replicate_sharding()
        self._m = jnp.zeros(states.nparams, self._dtype, device=sharding)
        real_dtype = jnp.finfo(self._dtype).dtype
        self._v = jnp.zeros(states.nparams, real_dtype, device=sharding)
        self._t = 0

        self._energy = None
        self._VarE = None
        self._Omean = None

        if file is not None:
            val = eqx.tree_deserialise_leaves(
                file, (self._mu, self._beta, self._m, self._v, self._t)
            )
            self._mu, self._beta, self._m, self._v, self._t = val

    @property
    def states(self) -> NaturalStateSet:
        return self._states

    @property
    def hamiltonian(self) -> qtx.operator.Operator:
        return self._hamiltonian

    @property
    def vs_type(self) -> VS_TYPE:
        return self._vs_type

    @property
    def imag_time(self) -> bool:
        return self._imag_time

    @property
    def energy(self) -> Optional[float]:
        return self._energy

    @property
    def VarE(self) -> Optional[float]:
        return self._VarE

    @property
    def Omean(self) -> Optional[jax.Array]:
        return self._Omean

    def _reweight(self, samples: Samples) -> jax.Array:
        if samples.reweight_factor is None:
            return jnp.ones(samples.nsamples)
        return samples.reweight_factor

    def _determinant_local_energy(self, samples: Samples) -> jax.Array:
        tuple_spins = jnp.asarray(samples.spins)
        nsamples, Nstates, Nmodes = tuple_spins.shape
        if Nstates != self.states.Nstates or Nmodes != self.states.Nmodes:
            raise ValueError(
                "Expected sample spins with shape "
                f"(nsamples, {self.states.Nstates}, {self.states.Nmodes}), got "
                f"{tuple_spins.shape}."
            )

        flat_spins = tuple_spins.reshape(nsamples * Nstates, Nmodes)
        A_scaled, row_shift = _scaled_psi_matrix(self.states, tuple_spins)

        B_columns = []
        for state in self.states.states:
            psi = jnp.asarray(state(flat_spins))
            Eloc = self.hamiltonian.Oloc(state, flat_spins).astype(get_default_dtype())
            B_columns.append(Eloc * psi)

        B = jnp.stack(B_columns, axis=-1).reshape(nsamples, Nstates, Nstates)
        B_scaled = B * jnp.exp(-row_shift[..., None])
        Ainv_B = jnp.linalg.solve(A_scaled, B_scaled)
        return jnp.trace(Ainv_B, axis1=-2, axis2=-1).astype(get_default_dtype())

    def get_Ebar(self, samples: Samples) -> jax.Array:
        r"""Compute Quantax-style centered/scaled local energies for the NES determinant."""
        Eloc = self._determinant_local_energy(samples)
        reweight = self._reweight(samples)

        Emean = jnp.mean(Eloc * reweight)
        self._energy = Emean.real
        Evar = jnp.abs(Eloc - Emean) ** 2
        self._VarE = jnp.mean(Evar * reweight).real

        return (Eloc - jnp.mean(Eloc)) * jnp.sqrt(reweight / samples.nsamples)

    def get_Obar(self, samples: Samples) -> jax.Array:
        r"""Compute determinant logarithmic Jacobians in member-state parameter order."""
        tuple_spins = jnp.asarray(samples.spins)
        nsamples, Nstates, Nmodes = tuple_spins.shape
        if Nstates != self.states.Nstates or Nmodes != self.states.Nmodes:
            raise ValueError(
                "Expected sample spins with shape "
                f"(nsamples, {self.states.Nstates}, {self.states.Nmodes}), got "
                f"{tuple_spins.shape}."
            )

        A_scaled, _ = _scaled_psi_matrix(self.states, tuple_spins)
        eye = jnp.eye(Nstates, dtype=A_scaled.dtype)
        Ainv = jnp.linalg.solve(A_scaled, jnp.broadcast_to(eye, A_scaled.shape))

        jacobians = self.states.jacobians(tuple_spins)
        blocks = []
        for index, jacobian in enumerate(jacobians):
            jacobian = jacobian.reshape(nsamples, Nstates, -1)
            coeff = Ainv[:, index, :] * A_scaled[:, :, index]
            blocks.append(jnp.einsum("ni,nip->np", coeff, jacobian))

        Omat = jnp.concatenate(blocks, axis=-1)
        reweight = self._reweight(samples)
        self._Omean = jnp.mean(Omat * reweight[:, None], axis=0)
        factor = jnp.sqrt(reweight / samples.nsamples)[:, None]
        return (Omat - jnp.mean(Omat, axis=0, keepdims=True)) * factor

    def _non_holomorphic_blocks(self) -> tuple[int, ...]:
        return tuple(2 * count for count in self.states.nparams_per_state)

    def _raw_non_holomorphic_to_step(self, raw_step: jax.Array) -> jax.Array:
        split_points = jnp.cumsum(
            jnp.asarray(self._non_holomorphic_blocks()[:-1])
        ).tolist()
        raw_blocks = jnp.split(raw_step, split_points, axis=-1)
        blocks = []
        for raw_block, nparams in zip(raw_blocks, self.states.nparams_per_state):
            blocks.append(raw_block[:nparams] + 1j * raw_block[nparams:])
        return jnp.concatenate(blocks, axis=-1)

    def _step_to_raw_columns(self, step: jax.Array) -> jax.Array:
        if self.vs_type != VS_TYPE.non_holomorphic:
            return step

        split_points = jnp.cumsum(
            jnp.asarray(self.states.nparams_per_state[:-1])
        ).tolist()
        blocks = jnp.split(step, split_points, axis=-1)
        return jnp.concatenate(
            [jnp.concatenate([block.real, block.imag], axis=-1) for block in blocks],
            axis=-1,
        )

    def _scale_to_raw_columns(self, scale: jax.Array) -> jax.Array:
        if self.vs_type != VS_TYPE.non_holomorphic:
            return scale

        split_points = jnp.cumsum(
            jnp.asarray(self.states.nparams_per_state[:-1])
        ).tolist()
        blocks = jnp.split(scale, split_points, axis=-1)
        return jnp.concatenate(
            [jnp.concatenate([block, block], axis=-1) for block in blocks],
            axis=-1,
        )

    def _sr_solve(self, Obar: jax.Array, Ebar: jax.Array) -> jax.Array:
        if self.vs_type == VS_TYPE.real_or_holomorphic:
            if not self.imag_time:
                Ebar = Ebar * 1j
            step = self._solver(Obar, Ebar)

        else:
            Obar = jnp.concatenate([Obar.real, Obar.imag], axis=0)
            if self.imag_time:
                Ebar = jnp.concatenate([Ebar.real, Ebar.imag])
            else:
                Ebar = jnp.concatenate([-Ebar.imag, Ebar.real])

            step = self._solver(Obar, Ebar)
            if self.vs_type == VS_TYPE.non_holomorphic:
                step = self._raw_non_holomorphic_to_step(step)

        return step.astype(get_default_dtype())

    def solve(self, Obar: jax.Array, Ebar: jax.Array) -> jax.Array:
        r"""Solve the AdamSR-corrected NES stochastic reconfiguration equation."""
        self._t += 1

        g = self._sr_solve(Obar, Ebar)
        self._m = self._mu * self._m + (1 - self._mu) * g
        self._v = self._beta * self._v + (1 - self._beta) * jnp.abs(g) ** 2
        m = self._m / (1 - self._mu**self._t)
        v = self._v / (1 - self._beta**self._t)
        V = v**0.25 + 1e-8

        raw_m = self._step_to_raw_columns(m.astype(Obar.dtype))
        raw_V = self._scale_to_raw_columns(V)
        Ebar = Ebar - Obar @ raw_m
        step = self._sr_solve(Obar / raw_V[None, :], Ebar)
        step = (step / V + m).astype(step.dtype)
        return step

    def get_step(self, samples: Samples) -> jax.Array:
        r"""Return a flat NES update ordered as ``NaturalStateSet.states``."""
        Ebar = self.get_Ebar(samples)
        Obar = self.get_Obar(samples)
        return self.solve(Obar, Ebar)

    def save(self, file: Union[str, Path, BinaryIO]) -> None:
        val = (self._mu, self._beta, self._m, self._v, self._t)
        if jax.process_index() == 0:
            eqx.tree_serialise_leaves(file, val)
