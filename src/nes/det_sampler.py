from __future__ import annotations

from functools import partial
from typing import Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import quantax as qtx
from jax.scipy.special import logsumexp
from jaxtyping import Key
from quantax.global_defs import get_subkeys
from quantax.sampler import Samples
from quantax.utils import (
    LogArray,
    PsiArray,
    filter_tree_map,
    rand_states,
    to_distributed_array,
)

from ..quantumhall_samplers import (
    _get_site_neighbors,
    _mix_flavour_flips,
    _pair_hop_offsets,
    _pair_hop_weights,
    _propose_exchange_dipolecons,
    _propose_valid_pair_hop,
)
from .state_set import NaturalStateSet


ProposalResult = Union[jax.Array, Tuple[jax.Array, jax.Array]]


def _natural_logdet(psi_matrix: PsiArray) -> LogArray:
    psi_matrix = LogArray.from_value(psi_matrix)
    row_shift = jnp.max(psi_matrix.logabs, axis=-1)
    scaled = psi_matrix.sign * jnp.exp(psi_matrix.logabs - row_shift[..., None])
    sign, logabs_scaled = jnp.linalg.slogdet(scaled)
    return LogArray(sign, logabs_scaled + jnp.sum(row_shift, axis=-1))


@jax.jit
def _log_soft_min(logA: jax.Array, logc: jax.Array, p) -> jax.Array:
    r"""
    :math:`\log \sum_i u_i^{-p}` with :math:`u_i^2 = \sum_a |A_{ia}|^2 / c_a`, from ``log|A|`` of shape
    ``(..., K, K)`` (row i = configuration, column a = member) and ``log c`` of shape ``(K,)``: shape ``(...)``.
    """
    return logsumexp(-0.5 * p * logsumexp(2 * logA - logc, axis=-1), axis=-1)


@partial(jax.jit, static_argnums=3)
def _member_scales(logA: jax.Array, logw: jax.Array, logc: jax.Array, iters: int) -> jax.Array:
    r"""
    The soft-min member scales :math:`\log c_a` (largest 0) from tuples ``log|A|`` ``(n, K, K)`` with log weights
    ``logw`` ``(n,)`` that take them to :math:`|\det A|^2`: the fixed point of
    :math:`c_a = K\, E_{\rm slot}[|\psi_a(s)|^2 / \rho(s)]`, :math:`\rho = \sum_b |\psi_b|^2 / c_b`, iterated
    ``iters`` times from ``logc``. For near-orthogonal members a slot of a :math:`|\det A|^2` tuple is distributed
    as :math:`\rho / K`, so :math:`c_a` is the norm of member a and :math:`u^2 = \rho` the span's density at s. Slots
    whose amplitudes all vanish and tuples of weight zero are left out.
    """
    K = logA.shape[-1]
    lA2 = 2 * logA.reshape(-1, K)
    ok = jnp.repeat(jnp.isfinite(logw), K) & jnp.any(jnp.isfinite(lA2), axis=1) & ~jnp.any(jnp.isnan(lA2), axis=1)
    lw = jnp.where(ok, jnp.repeat(logw, K), -jnp.inf)
    lA2 = jnp.where(ok[:, None], lA2, 0.0)

    def body(_, logc):
        logrho = logsumexp(lA2 - logc, axis=1)
        new = logsumexp(lw[:, None] + lA2 - logrho[:, None], axis=0)
        return new - jnp.max(new)

    return jax.lax.fori_loop(0, iters, body, logc - jnp.max(logc))


class NaturalDetAbstractSampler:
    r"""
    Abstract sampler for natural excited state determinants.

    Samples are tuples ``(s_1, ..., s_n)`` with one configuration per member
    state. The sampled amplitude is ``det A`` with
    ``A[i, a] = psi_a(s_i)``, so the stationary probability is proportional to
    ``|det A| ** reweight``.

    With ``soft_min = p`` the stationary probability is instead the soft-min tempering

    .. math::

        Q(s_1, \dots, s_K) \propto |\det A|^2 \sum_i u_i^{-p}, \qquad u_i^2 = \sum_a |\psi_a(s_i)|^2 / c_a,

    with :math:`c_a` the members' norms (``member_log_scales``), refreshed after every sweep from that sweep's
    tuples. :math:`u_i^2` is the span's density at :math:`s_i`, so the factor raises the tuples holding a
    configuration the span barely covers, the rare tuples whose large local values dominate the noise of
    tr(S^-1 H) and tr(S^-1 L^- L^+) under :math:`|\det A|^2`. The samples carry the true :math:`\det A` and the
    reweight factor :math:`1 / \sum_i u_i^{-p}` (normalised) back to :math:`|\det A|^2`, exact for any
    :math:`c`; the scales only shape the tempering. ``reweight`` must then stay 2. p = 1 had the lowest noise
    of energy and L^- L^+ among the reweight and soft-min settings measured (N=14 K=4, N=20 K=2;
    scripts/NESReweightDiagnostics.py).
    """

    def __init__(
        self,
        states: NaturalStateSet,
        nsamples: int,
        reweight: float = 2.0,
        soft_min: Optional[float] = None,
    ):
        if not isinstance(states, NaturalStateSet):
            raise TypeError("NaturalDetAbstractSampler expects a NaturalStateSet.")
        if nsamples % jax.device_count() != 0:
            raise ValueError(
                "`nsamples` should be a multiple of the number of devices, but got "
                f"{nsamples} samples and {jax.device_count()} devices."
            )
        if soft_min is not None and reweight != 2.0:
            raise ValueError(f"soft_min samples |det A|^2 sum_i u_i^-p: reweight must stay 2, got {reweight}")

        self._states = states
        self._nsamples = nsamples
        self._reweight = reweight
        self._soft_min = None if soft_min is None else float(soft_min)
        self._logc = None

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

    @property
    def soft_min(self) -> Optional[float]:
        """The soft-min exponent p, or None for :math:`|\\det A|^{\\rm reweight}` sampling."""
        return self._soft_min

    @property
    def member_log_scales(self) -> Optional[jax.Array]:
        """Soft-min: the :math:`\\log c_a` of the next sweep (largest 0); None before the first sweep or without
        soft-min."""
        return self._logc

    def determinant(self, tuple_spins: jax.Array) -> LogArray:
        return _natural_logdet(self.states.psi_matrix(tuple_spins))

    def sweep(self) -> Samples:
        raise NotImplementedError

    def _get_reweight_factor(self, psi: PsiArray) -> jax.Array:
        reweight_factor = abs(psi) ** (2 - self.reweight)
        return jnp.asarray(reweight_factor / reweight_factor.mean())

    def _log_target(self, det_logabs: jax.Array, logA: Optional[jax.Array] = None,
                    logc: Optional[jax.Array] = None) -> jax.Array:
        """log of the sampled density up to a constant, -inf where det A vanishes (or is not finite). ``logA``
        (log|A|) and ``logc`` are read only with soft-min; inside jit, ``logc`` must be an argument."""
        finite = jnp.isfinite(det_logabs)
        if self._soft_min is None:
            return jnp.where(finite, self.reweight * det_logabs, -jnp.inf)
        return jnp.where(finite, 2 * det_logabs + _log_soft_min(logA, logc, self._soft_min), -jnp.inf)

    def _sweep_weights(self, psi: PsiArray, log_target: jax.Array) -> jax.Array:
        """Reweight factors to :math:`|\\det A|^2`, normalised to mean 1."""
        if self._soft_min is None:
            return self._get_reweight_factor(psi)
        logw = jnp.where(jnp.isfinite(log_target), 2 * LogArray.from_value(psi).logabs - log_target, -jnp.inf)
        w = jnp.exp(logw - jnp.max(logw))
        return w / w.mean()

    def _init_scales(self, logA: jax.Array) -> None:
        """Soft-min scales before the first sweep, from the starting tuples with equal weights."""
        K = logA.shape[-1]
        lA2 = 2 * logA.reshape(-1, K)
        start = logsumexp(jnp.where(jnp.isfinite(lA2), lA2, -jnp.inf), axis=0)
        self._logc = _member_scales(logA, jnp.zeros(logA.shape[0]), start, 200)

    def _refresh_scales(self, logA: jax.Array, weights: jax.Array) -> None:
        """Soft-min scales for the next sweep, from this sweep's tuples weighted to :math:`|\\det A|^2`."""
        self._logc = _member_scales(logA, jnp.log(weights), self._logc, 200)


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
        soft_min: Optional[float] = None,
    ):
        super().__init__(state, nsamples, reweight, soft_min)

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
            self._spins = to_distributed_array(initial_spins.astype(jnp.int8))

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

    @property
    def acceptance(self) -> dict:
        r"""
        Over the Metropolis steps and chains of the last sweep: ``moved``, the fraction of
        steps that changed the tuple, and ``valid``, the fraction whose proposal was a real
        move (something changed, no doubly occupied orbital). ``moved / valid`` is the
        Metropolis acceptance of the joint move of all ``Nstates`` configurations.
        """
        return {key: float(value) for key, value in self._acceptance.items()}

    def _evaluate(self, spins: jax.Array) -> Tuple[PsiArray, jax.Array]:
        """det A on tuples and the log of the sampled density there (`_log_target`)."""
        if self._soft_min is None:
            psi = self.determinant(spins)
            return psi, self._log_target(LogArray.from_value(psi).logabs)
        A = LogArray.from_value(self.states.psi_matrix(spins))
        det = _natural_logdet(A)
        return det, self._log_target(det.logabs, A.logabs, self._logc)

    def _partial_sweep(self, nsweeps: int, spins: jax.Array) -> Samples:
        if self._soft_min is not None and self._logc is None:
            self._init_scales(LogArray.from_value(self.states.psi_matrix(spins)).logabs)
        psi, log_target = self._evaluate(spins)
        samples = Samples(spins, psi)

        keys_propose = get_subkeys(nsweeps)
        keys_update = get_subkeys(nsweeps)
        moved = valid = 0.0
        for keyp, keyu in zip(keys_propose, keys_update):
            samples, log_target = self._single_sweep(keyp, keyu, samples, log_target)
            # device scalars, summed without a host sync per step
            moved, valid = moved + self._step_moved, valid + self._step_valid
        self._acceptance = dict(moved=moved / max(nsweeps, 1), valid=valid / max(nsweeps, 1))

        weights = self._sweep_weights(samples.psi, log_target)
        if self._soft_min is not None:
            # one more evaluation of A per sweep (a sweep takes 2 K Nmodes of them by default)
            self._refresh_scales(LogArray.from_value(self.states.psi_matrix(samples.spins)).logabs, weights)
        return Samples(samples.spins, samples.psi, None, weights)

    def _single_sweep(self, keyp: Key, keyu: Key, samples: Samples,
                      log_target: jax.Array) -> Tuple[Samples, jax.Array]:
        proposal = self.propose(keyp, samples.spins)
        if isinstance(proposal, tuple):
            new_spins, propose_ratio = proposal
        else:
            new_spins = proposal
            propose_ratio = None

        new_psi, new_log_target = self._evaluate(new_spins)
        return self._update(keyu, propose_ratio, samples, Samples(new_spins, new_psi), log_target, new_log_target)

    def _update(
        self,
        key: Key,
        propose_ratio: Optional[jax.Array],
        old_samples: Samples,
        new_samples: Samples,
        old_log_target: jax.Array,
        new_log_target: jax.Array,
    ) -> Tuple[Samples, jax.Array]:
        log_accept = new_log_target - old_log_target
        if propose_ratio is not None:
            log_accept += jnp.log(propose_ratio)

        nsamples = old_samples.spins.shape[0]
        log_uniform = jnp.log(jr.uniform(key, (nsamples,)))
        accepted = (log_accept > log_uniform) | (old_log_target == -jnp.inf)

        sites = qtx.get_sites()
        if sites.particle_type == qtx.PARTICLE_TYPE.spinful_fermion and not sites.double_occ:
            nsites = self.Nmodes // 2
            spins = new_samples.spins.reshape(nsamples, self.Nstates, 2, nsites)
            occ_allowed = jnp.all(jnp.any(spins <= 0, axis=2), axis=(1, 2))
        else:
            occ_allowed = True

        updated = jnp.any(old_samples.spins != new_samples.spins, axis=(1, 2))
        cond = accepted & updated & occ_allowed
        self._step_moved = jnp.mean(cond)                      # read by `_partial_sweep`
        self._step_valid = jnp.mean(updated & occ_allowed)

        def select(new, old):
            cond_expand = cond.reshape([-1] + [1] * (new.ndim - 1))
            return jnp.where(cond_expand, new, old)

        return filter_tree_map(select, new_samples, old_samples), jnp.where(cond, new_log_target, old_log_target)


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
        soft_min: Optional[float] = None,
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
            state, nsamples, reweight, thermal_steps, sweep_steps, initial_spins, soft_min
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


class NaturalLzDetFlavourFlipSampler(NaturalLzDetSampler):
    r"""
    `NaturalLzDetSampler` mixed with single-electron flavour flips, the NES counterpart of
    `~qhuantax.quantumhall_samplers.FermionDipoleConsFlavourFlip`.

    The pair move never flips one electron alone, so near the polarised vacuum the chains freeze
    on the configurations of the h=0 members. Each configuration of each tuple independently
    takes a flip instead of its pair move with probability ``flip_prob``; both moves are
    symmetric, so the proposal ratio stays one.
    """

    def __init__(self, *args, flip_prob: float = 0.5, **kwargs):
        self.flip_prob = flip_prob
        super().__init__(*args, **kwargs)

    @partial(jax.jit, static_argnums=0)
    def propose(self, key: Key, old_spins: jax.Array) -> jax.Array:
        key_pair, key_flip = jr.split(key)
        pair_spins = super().propose(key_pair, old_spins)
        nsamples, Nstates, Nmodes = old_spins.shape
        mixed = _mix_flavour_flips(key_flip, old_spins.reshape(-1, Nmodes),
                                   pair_spins.reshape(-1, Nmodes), self.flip_prob)
        return mixed.reshape(nsamples, Nstates, Nmodes)


class NaturalLzDetPairHopSampler(NaturalDetSampler):
    r"""
    Metropolis sampler for natural determinant tuples whose only move is the dipole-conserving pair hop, drawn
    among the valid ones; no flavour flips.

    Every configuration of a tuple takes a pair hop chosen uniformly among those that change it (both targets
    empty, in range and distinct): the move set of `NaturalLzDetSampler` without its null proposals, which are
    ~78% of its proposals at h_c (N=14). A pair hop changes the down count by 0, +-1 or +-2. The proposal is not
    symmetric, so the acceptance carries the Hastings factor :math:`\prod_i n(s_i) / n(s'_i)`, n the number of
    valid hops of a configuration (see `~qhuantax.quantumhall_samplers._propose_valid_pair_hop`).

    ``n_neighbor`` holds the hop distances of the first electron, as for `NaturalLzDetSampler`; the orbitals are
    an open chain.
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
        soft_min: Optional[float] = None,
    ):
        sites = qtx.get_sites()
        if sites.Nparticles is None or sites.particle_type != qtx.PARTICLE_TYPE.spinful_fermion:
            raise ValueError("`NaturalLzDetPairHopSampler` needs spinful fermions with a fixed particle number.")
        self._offsets = _pair_hop_offsets(n_neighbor)
        self._nparticles = int(sites.Ntotal)

        super().__init__(
            state, nsamples, reweight, thermal_steps, sweep_steps, initial_spins, soft_min
        )

    @partial(jax.jit, static_argnums=0)
    def propose(self, key: Key, old_spins: jax.Array) -> Tuple[jax.Array, jax.Array]:
        nsamples, Nstates, Nmodes = old_spins.shape
        flat_spins = old_spins.reshape(nsamples * Nstates, Nmodes)
        new_spins, nvalid_old = _propose_valid_pair_hop(key, flat_spins, self._offsets, self._nparticles)
        nvalid_new = jnp.sum(_pair_hop_weights(new_spins == 1, self._offsets, self._nparticles)[3], axis=(1, 2))
        # configurations without valid hops stay put (nvalid_new = nvalid_old = 0): ratio one
        ratio = jnp.where(nvalid_old > 0, nvalid_old / jnp.maximum(nvalid_new, 1), 1.0)
        return (new_spins.reshape(nsamples, Nstates, Nmodes),
                jnp.prod(ratio.reshape(nsamples, Nstates), axis=1))


class NaturalLzDetSingleRowPairHopSampler(NaturalDetSampler):
    r"""
    `NaturalLzDetPairHopSampler` moving one configuration of each tuple per step, as FermiNet's NES moves one
    state's electrons per step.

    Each step picks a row i of every chain's tuple uniformly and proposes a valid pair hop of s_i (no null
    proposals), accepted with :math:`|\det A' / \det A|^{\rm reweight}\, n(s_i) / n(s'_i)` (with ``soft_min``,
    :math:`Q(A') / Q(A)` in place of the determinant ratio, see `NaturalDetAbstractSampler`). Only row i of
    ``A[i, a] = psi_a(s_i)`` changes, so a step evaluates the K members on one configuration per chain, K network
    evaluations instead of the K^2 of a joint move. A is kept between steps and recomputed in full at the start of
    every sweep (the members change between sweeps); its determinant is taken directly, being K x K.

    The defaults are ``sweep_steps = K Nmodes``, about Nmodes moves per configuration per sweep, and
    ``thermal_steps = 20 K Nmodes``.
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
        soft_min: Optional[float] = None,
    ):
        sites = qtx.get_sites()
        if sites.Nparticles is None or sites.particle_type != qtx.PARTICLE_TYPE.spinful_fermion:
            raise ValueError("`NaturalLzDetSingleRowPairHopSampler` needs spinful fermions with a fixed particle "
                             "number.")
        self._offsets = _pair_hop_offsets(n_neighbor)
        self._nparticles = int(sites.Ntotal)
        self._check_occ = not sites.double_occ
        K, Nmodes = state.Nstates, state.Nmodes
        thermal_steps = 20 * K * Nmodes if thermal_steps is None else thermal_steps
        sweep_steps = K * Nmodes if sweep_steps is None else sweep_steps

        super().__init__(
            state, nsamples, reweight, thermal_steps, sweep_steps, initial_spins, soft_min
        )

    def _row_values(self, rows: jax.Array) -> LogArray:
        """The members' amplitudes on configurations ``(m, Nmodes)``: ``(m, Nstates)``."""
        return LogArray.from_value(self.states.amplitudes(rows))

    @partial(jax.jit, static_argnums=0)
    def _propose_row(self, key: Key, spins: jax.Array):
        nsamples, Nstates, Nmodes = spins.shape
        key_row, key_hop = jr.split(key)
        row = jr.randint(key_row, (nsamples,), 0, Nstates)
        old = spins[jnp.arange(nsamples), row]
        new, nvalid_old = _propose_valid_pair_hop(key_hop, old, self._offsets, self._nparticles)
        nvalid_new = jnp.sum(_pair_hop_weights(new == 1, self._offsets, self._nparticles)[3], axis=(1, 2))
        valid = nvalid_old > 0
        if self._check_occ:
            valid &= jnp.all(jnp.any(new.reshape(nsamples, 2, Nmodes // 2) <= 0, axis=1), axis=1)
        ratio = jnp.where(nvalid_old > 0, nvalid_old / jnp.maximum(nvalid_new, 1), 1.0)
        return row, new, ratio, valid

    @partial(jax.jit, static_argnums=0)
    def _update_row(self, key, spins, sign, logabs, det_sign, det_logabs, log_target, logc, row, new, new_sign,
                    new_logabs, ratio, valid):
        nsamples, Nstates, _ = spins.shape
        at_row = jnp.arange(Nstates)[None, :] == row[:, None]                     # (nsamples, Nstates)
        trial_sign = jnp.where(at_row[..., None], new_sign[:, None, :], sign)
        trial_logabs = jnp.where(at_row[..., None], new_logabs[:, None, :], logabs)
        trial = _natural_logdet(LogArray(trial_sign, trial_logabs))
        trial_target = self._log_target(trial.logabs, trial_logabs, logc)
        log_accept = trial_target - log_target + jnp.log(ratio)
        accepted = ((log_accept > jnp.log(jr.uniform(key, (nsamples,)))) | (log_target == -jnp.inf)) & valid
        keep = accepted[:, None, None]
        return (jnp.where(keep & at_row[..., None], new[:, None, :], spins),
                jnp.where(keep, trial_sign, sign), jnp.where(keep, trial_logabs, logabs),
                jnp.where(accepted, trial.sign, det_sign), jnp.where(accepted, trial.logabs, det_logabs),
                jnp.where(accepted, trial_target, log_target), jnp.mean(accepted), jnp.mean(valid))

    def _partial_sweep(self, nsweeps: int, spins: jax.Array) -> Samples:
        nsamples, Nstates, Nmodes = spins.shape
        values = self._row_values(spins.reshape(-1, Nmodes))
        sign = values.sign.reshape(nsamples, Nstates, Nstates)
        logabs = values.logabs.reshape(nsamples, Nstates, Nstates)
        if self._soft_min is not None and self._logc is None:
            self._init_scales(logabs)
        det = _natural_logdet(LogArray(sign, logabs))
        det_sign, det_logabs = det.sign, det.logabs
        logc = self._logc                       # an argument of the jitted step: fixed within the sweep
        log_target = self._log_target(det_logabs, logabs, logc)

        moved = valid = 0.0
        for key in get_subkeys(nsweeps):
            key_propose, key_update = jr.split(key)
            row, new, ratio, step_valid = self._propose_row(key_propose, spins)
            new_values = self._row_values(new)
            spins, sign, logabs, det_sign, det_logabs, log_target, step_moved, step_valid = self._update_row(
                key_update, spins, sign, logabs, det_sign, det_logabs, log_target, logc, row, new, new_values.sign,
                new_values.logabs, ratio, step_valid)
            # device scalars, summed without a host sync per step
            moved, valid = moved + step_moved, valid + step_valid
        self._acceptance = dict(moved=moved / max(nsweeps, 1), valid=valid / max(nsweeps, 1))

        psi = LogArray(det_sign, det_logabs)
        weights = self._sweep_weights(psi, log_target)
        if self._soft_min is not None:
            self._refresh_scales(logabs, weights)
        return Samples(spins, psi, None, weights)
