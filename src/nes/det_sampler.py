from __future__ import annotations

from functools import partial
from typing import Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import quantax as qtx
from jaxtyping import Key
from quantax.global_defs import get_subkeys
from quantax.sampler import Samples
from quantax.utils import (
    LogArray,
    PsiArray,
    filter_tree_map,
    rand_states,
    to_distribute_array,
)

from ..quantumhall_samplers import (
    _get_site_neighbors,
    _propose_exchange_dipolecons,
)
from .state_set import NaturalStateSet


ProposalResult = Union[jax.Array, Tuple[jax.Array, jax.Array]]


def _natural_logdet(psi_matrix: PsiArray) -> LogArray:
    psi_matrix = LogArray.from_value(psi_matrix)
    row_shift = jnp.max(psi_matrix.logabs, axis=-1)
    scaled = psi_matrix.sign * jnp.exp(psi_matrix.logabs - row_shift[..., None])
    sign, logabs_scaled = jnp.linalg.slogdet(scaled)
    return LogArray(sign, logabs_scaled + jnp.sum(row_shift, axis=-1))


class NaturalDetAbstractSampler:
    r"""
    Abstract sampler for natural excited state determinants.

    Samples are tuples ``(s_1, ..., s_n)`` with one configuration per member
    state. The sampled amplitude is ``det A`` with
    ``A[i, a] = psi_a(s_i)``, so the stationary probability is proportional to
    ``|det A| ** reweight``.
    """

    def __init__(
        self,
        states: NaturalStateSet,
        nsamples: int,
        reweight: float = 2.0,
    ):
        if not isinstance(states, NaturalStateSet):
            raise TypeError("NaturalDetAbstractSampler expects a NaturalStateSet.")
        if nsamples % jax.device_count() != 0:
            raise ValueError(
                "`nsamples` should be a multiple of the number of devices, but got "
                f"{nsamples} samples and {jax.device_count()} devices."
            )

        self._states = states
        self._nsamples = nsamples
        self._reweight = reweight

    @property
    def states(self) -> NaturalStateSet:
        """The natural state set whose determinant defines the sampled amplitude."""
        return self._states

    @property
    def Nsites(self) -> int:
        return self.states.Nsites

    @property
    def Nmodes(self) -> int:
        return self.states.Nmodes

    @property
    def nsamples(self) -> int:
        return self._nsamples

    @property
    def reweight(self) -> Union[float, jax.Array]:
        return self._reweight

    @property
    def Nstates(self) -> int:
        return self.states.Nstates

    def determinant(self, tuple_spins: jax.Array) -> LogArray:
        return _natural_logdet(self.states.psi_matrix(tuple_spins))

    def sweep(self) -> Samples:
        raise NotImplementedError

    def _get_reweight_factor(self, psi: PsiArray) -> jax.Array:
        reweight_factor = abs(psi) ** (2 - self.reweight)
        return jnp.asarray(reweight_factor / reweight_factor.mean())


class NaturalDetSampler(NaturalDetAbstractSampler):
    r"""
    Metropolis sampler base for natural excited state determinants.
    """

    def __init__(
        self,
        state: NaturalStateSet,
        nsamples: int,
        reweight: float = 2.0,
        thermal_steps: Optional[int] = None,
        sweep_steps: Optional[int] = None,
        initial_spins: Optional[jax.Array] = None,
    ):
        super().__init__(state, nsamples, reweight)

        self._thermal_steps = 20 * self.Nstates * self.Nmodes if thermal_steps is None else thermal_steps
        self._sweep_steps = 2 * self.Nstates * self.Nmodes if sweep_steps is None else sweep_steps

        self.reset(initial_spins)

    def reset(self, initial_spins: jax.Array | None = None) -> None:
        if initial_spins is None:
            self._spins = rand_states(self.nsamples * self.Nstates).reshape(self.nsamples, self.Nstates, self.Nmodes)
        else:
            if initial_spins.ndim == 1:
                initial_spins = jnp.tile(initial_spins, (self.nsamples, self.Nstates, 1))
            else:
                initial_spins = initial_spins.reshape(self.nsamples, self.Nstates, self.Nmodes)
            self._spins = to_distribute_array(initial_spins.astype(jnp.int8))

        if self._thermal_steps > 0:
            self.sweep(self._thermal_steps)

    def propose(self, key: Key, old_spins: jax.Array) -> ProposalResult:
        raise NotImplementedError

    def sweep(self, nsweeps: Optional[int] = None) -> Samples:
        if nsweeps is None:
            nsweeps = self._sweep_steps

        samples = self._partial_sweep(nsweeps, self._spins)
        self._spins = samples.spins
        return samples

    def _partial_sweep(self, nsweeps: int, spins: jax.Array) -> Samples:
        psi = self.determinant(spins)
        samples = Samples(spins, psi)

        keys_propose = get_subkeys(nsweeps)
        keys_update = get_subkeys(nsweeps)
        for keyp, keyu in zip(keys_propose, keys_update):
            samples = self._single_sweep(keyp, keyu, samples)

        return Samples(samples.spins, samples.psi, None, self._get_reweight_factor(samples.psi))

    def _single_sweep(self, keyp: Key, keyu: Key, samples: Samples) -> Samples:
        proposal = self.propose(keyp, samples.spins)
        if isinstance(proposal, tuple):
            new_spins, propose_ratio = proposal
        else:
            new_spins = proposal
            propose_ratio = None

        new_samples = Samples(new_spins, self.determinant(new_spins))
        return self._update(keyu, propose_ratio, samples, new_samples)

    def _update(
        self,
        key: Key,
        propose_ratio: Optional[jax.Array],
        old_samples: Samples,
        new_samples: Samples,
    ) -> Samples:
        old_logabs = LogArray.from_value(old_samples.psi).logabs
        new_logabs = LogArray.from_value(new_samples.psi).logabs
        log_accept = self.reweight * (new_logabs - old_logabs)
        if propose_ratio is not None:
            log_accept += jnp.log(propose_ratio)

        nsamples = old_samples.spins.shape[0]
        log_uniform = jnp.log(jr.uniform(key, (nsamples,)))
        accepted = (log_accept > log_uniform) | (old_logabs == -jnp.inf)

        sites = qtx.get_sites()
        if sites.particle_type == qtx.PARTICLE_TYPE.spinful_fermion and not sites.double_occ:
            nsites = self.Nmodes // 2
            spins = new_samples.spins.reshape(nsamples, self.Nstates, 2, nsites)
            occ_allowed = jnp.all(jnp.any(spins <= 0, axis=2), axis=(1, 2))
        else:
            occ_allowed = True

        updated = jnp.any(old_samples.spins != new_samples.spins, axis=(1, 2))
        cond = accepted & updated & occ_allowed

        def select(new, old):
            cond_expand = cond.reshape([-1] + [1] * (new.ndim - 1))
            return jnp.where(cond_expand, new, old)

        return filter_tree_map(select, new_samples, old_samples)


class NaturalLzDetSampler(NaturalDetSampler):
    r"""
    Dipole-conserving Metropolis sampler for natural determinant tuples.

    Each proposal applies the plain non-squeezed quantum Hall dipole-conserving
    move to every configuration in every determinant tuple. The proposal ratio
    is one, as in ``FermionTwoBodyDipoleCons``.
    """

    def __init__(
        self,
        state: NaturalStateSet,
        nsamples: int,
        reweight: float = 2.0,
        thermal_steps: Optional[int] = None,
        sweep_steps: Optional[int] = None,
        initial_spins: Optional[jax.Array] = None,
        n_neighbor: Union[int, Sequence[int]] = 1,
    ):
        sites = qtx.get_sites()
        if sites.Nparticles is None:
            raise ValueError(
                "The number of fermions should be specified in sites for "
                "`NaturalLzDetSampler`."
            )
        if sites.particle_type not in (
            qtx.PARTICLE_TYPE.spinful_fermion,
            qtx.PARTICLE_TYPE.spinless_fermion,
        ):
            raise ValueError(
                "`NaturalLzDetSampler` supports only spinful or spinless fermions."
            )

        self._hopping_particle = 1 if 2 * sites.Ntotal <= state.Nmodes else -1
        self._spinful = sites.particle_type == qtx.PARTICLE_TYPE.spinful_fermion
        self._neighbors = _get_site_neighbors(n_neighbor)

        super().__init__(
            state, nsamples, reweight, thermal_steps, sweep_steps, initial_spins
        )

    @partial(jax.jit, static_argnums=0)
    def propose(self, key: Key, old_spins: jax.Array) -> jax.Array:
        nsamples, Nstates, Nmodes = old_spins.shape
        flat_spins = old_spins.reshape(nsamples * Nstates, Nmodes)
        new_selected_spins = _propose_exchange_dipolecons(
            key,
            flat_spins,
            self._hopping_particle,
            self._neighbors,
            self._spinful,
        )
        return new_selected_spins.reshape(nsamples, Nstates, Nmodes)
