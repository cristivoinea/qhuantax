r"""
Deflated NES: the trace of :math:`H + \beta \Pi`, with :math:`\Pi` the projector onto states lowered from a higher
:math:`L_z`.

At :math:`L_z = 0` the lowest L = 2 levels lie below the third and fourth L = 0 ones (N = 8-24 at h_c), and the
:math:`L^- L^+` penalty that keeps them out of a K = 4 window prices every L > 0 by :math:`\lambda L(L+1)`, the high-L
junk of training included: its local estimator has variance ~ L(L+1) per unit of junk, 15-25x VarE_H at N = 24.
Deflation lifts only the intruders. Their highest-weight states :math:`\phi_j` are found where they are the lowest
(an NES run at :math:`L_z = L`, or ED at small N) and lowered, :math:`\chi_j = (L^-)^p \phi_j`,
:math:`p = L_z(\phi) - L_z`. A state at :math:`L_z(\phi)` holds no L below :math:`L_z(\phi)` and :math:`L^-` keeps L,
so :math:`\chi_j` has no L = 0 content whatever the errors of :math:`\phi_j`: every L = 0 eigenstate of H is one of
:math:`H' = H + \beta \Pi` with the same energy, and :math:`\beta \Pi` (bounded by :math:`\beta`) sees nothing else.

:math:`\Pi = \sum_{jk} |\chi_j\rangle (G_\chi^{-1})_{jk} \langle\chi_k|` enters the trace like any operator:
:math:`B_{ia} = (\Pi\psi_a)(x_i) = \sum_j \chi_j(x_i) M_{ja}`, :math:`M = G_\chi^{-1}\langle\chi|\psi\rangle`, where
:math:`\chi_j(x) = \sum_y \langle y|(L^+)^p|x\rangle \phi_j(y)` takes :math:`\phi_j` at the configurations p up-hops
from x. M is a property of the members, estimated on the :math:`\phi` side, where it is zero-variance:
:math:`\langle\chi|\psi\rangle = \langle\phi|(L^+)^p\psi\rangle`, and J-tuples Y drawn from :math:`|\det\Phi(Y)|^2`,
:math:`\Phi_{ij} = \phi_j(y_i)`, give :math:`E[\Phi^{-1} R] = G_\phi^{-1}\langle\phi|(L^+)^p\psi\rangle` with
:math:`R_{ia} = ((L^+)^p\psi_a)(y_i)` (the NES identity: Cauchy-Binet and Cramer). :math:`(L^+)^p\psi = 0` for an
L < p member, so every sample gives exactly zero on a clean member, and the noise scales with the contamination,
like :math:`L^- L^+`'s. (Sampling the lowered states instead, :math:`E_{\chi}[X^{-1}\Psi]`, has an absolute noise
floor ~1/sqrt(n) on the overlap, which a reused bank turns into a fixed force on clean members: tested at N = 8,
abandoned.) :math:`G_\chi = c\, G_\phi` with :math:`c = \prod_{m=L-p+1}^{L} [L(L+1) - m(m-1)]` for highest-weight
:math:`\phi` of :math:`L = L_z(\phi)` (24 for L = p = 2); higher-L junk in :math:`\phi` only shifts :math:`\Pi`'s
eigenvalue on it slightly from 1.

:math:`\phi` is fixed, so the tuples are drawn once into a bank (`sample_bank`, the NES sampler on the
:math:`\phi`, plain evaluations) and every sweep needs the members at the configurations p down-hops from the bank's,
~(N/2)^2/2 per configuration for p = 2, as many as :math:`L^- L^+` connects; with the zero-variance property a reused
bank leaves the clean members exactly unpenalised and only scales the penalty on contaminated ones by ~1/sqrt(n).

:math:`L^\pm` hops move one electron between adjacent modes of one flavour (up modes 0..N-1, down N..2N-1), so they
carry no fermion sign; the hop m-1 -> m has the :math:`L^+` element :math:`\sqrt{m(N-m)}`, as in
`qhuantax.nes.lmetric`. States at different :math:`L_z(\phi)` go in separate `DeflationGroup` s (orthogonal for
exact :math:`\phi`), each with its own bank; :math:`\Pi` is their sum.
"""

from __future__ import annotations

from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from quantax.sampler import Samples
from quantax.utils import LogArray

from .det_sampler import NaturalLzDetFlavourFlipSampler
from .optimizer import NaturalTraceEnergyGrad
from .state_set import NaturalStateSet


def _to_bits(spins: np.ndarray) -> np.ndarray:
    """Configurations (n, Nmodes) of +-1 as occupation bit strings (uint64, mode k at bit k)."""
    occ = np.asarray(spins).reshape(len(spins), -1) > 0
    if occ.shape[1] > 64:
        raise ValueError("deflation packs configurations into 64 bits: at most 32 orbitals")
    weights = np.left_shift(np.uint64(1), np.arange(occ.shape[1], dtype=np.uint64))
    return np.sum(occ * weights, axis=1, dtype=np.uint64)


def _from_bits(bits: np.ndarray, nmodes: int) -> np.ndarray:
    occ = (bits[:, None] >> np.arange(nmodes, dtype=np.uint64)[None, :]) & np.uint64(1)
    return np.where(occ == 1, 1, -1).astype(np.int8)


def lplus_power(spins: np.ndarray, power: int, N: int, up: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    For each configuration of ``spins`` (n, 2N), the configurations ``power`` :math:`L^+` hops above it (``up``) or
    below it, with the matrix element of :math:`(L^+)^p` between the two (all positive): ``(index, bits, coeff)``,
    one entry per distinct pair, sorted by the index into ``spins``, the other configuration as a bit string.
    Configurations without a hop have no entry.
    """
    bits = _to_bits(spins)
    index, coeff = np.arange(len(bits)), np.ones(len(bits))
    one = np.uint64(1)
    pos = np.array([f * N + m for f in range(2) for m in range(N - 1)], dtype=np.uint64)   # modes k <-> k + 1
    elem = np.array([np.sqrt((m + 1) * (N - m - 1)) for _ in range(2) for m in range(N - 1)])
    src, dst = (pos, pos + one) if up else (pos + one, pos)
    for _ in range(power):
        allowed = (((bits[:, None] >> src) & one) == 1) & (((bits[:, None] >> dst) & one) == 0)
        r, h = np.nonzero(allowed)
        new = bits[r] ^ (one << pos[h]) ^ (one << (pos[h] + one))
        idx, c = index[r], coeff[r] * elem[h]
        # the same configuration reached by different hop orders: one entry, elements summed
        order = np.lexsort((new, idx))
        idx, new, c = idx[order], new[order], c[order]
        start = np.ones(len(idx), bool)
        start[1:] = (idx[1:] != idx[:-1]) | (new[1:] != new[:-1])
        coeff = np.bincount(np.cumsum(start) - 1, weights=c)
        index, bits = idx[start], new[start]
    return index, bits, coeff


def log_amplitudes(state, spins: np.ndarray, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    """sign and log|.| of ``state`` on ``spins`` (n, Nmodes), in float64, in chunks."""
    signs, logs = [], []
    for start in range(0, len(spins), chunk):
        v = state(jnp.asarray(spins[start:start + chunk]))
        v = v if isinstance(v, LogArray) else LogArray.from_value(jnp.asarray(v))
        signs.append(np.real(np.asarray(v.sign)).astype(np.float64))
        logs.append(np.asarray(v.logabs, dtype=np.float64))
    if not signs:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(signs), np.concatenate(logs)


def _segment_logsum(index, coeff, sign, logabs, n) -> tuple[np.ndarray, np.ndarray]:
    """sum over each entry's segment of coeff * sign * exp(logabs), as (sign, log|.|) per index (0, -inf if none)."""
    out_sign, out_log = np.zeros(n), np.full(n, -np.inf)
    if len(index) == 0:
        return out_sign, out_log
    starts = np.flatnonzero(np.r_[True, index[1:] != index[:-1]])
    lengths = np.diff(np.r_[starts, len(index)])
    finite = np.isfinite(logabs)
    top = np.maximum.reduceat(np.where(finite, logabs, -np.inf), starts)
    shift = np.repeat(np.where(np.isfinite(top), top, 0.0), lengths)
    terms = np.where(finite, coeff * sign * np.exp(np.where(finite, logabs, 0.0) - shift), 0.0)
    total = np.add.reduceat(terms, starts)
    x = index[starts]
    nonzero = (total != 0) & np.isfinite(top)
    out_sign[x] = np.sign(total)
    out_log[x] = np.where(nonzero, top + np.log(np.abs(np.where(nonzero, total, 1.0))), -np.inf)
    return out_sign, out_log


def applied_lplus(states: Sequence, spins, power: int, N: int, up: bool, chunk: int = 65536):
    r"""
    sign and log|.| (two (n, len(states)) arrays) of :math:`(L^-)^p f` (``up``: f at the configurations above) or
    :math:`(L^+)^p f` (f below) on ``spins`` (n, 2N), for each state f of ``states``; one hop expansion for all.
    """
    spins = np.asarray(spins).reshape(-1, 2 * N)
    n = len(spins)
    index, bits, coeff = lplus_power(spins, power, N, up=up)
    other = _from_bits(bits, 2 * N)
    sign, logabs = np.zeros((n, len(states))), np.full((n, len(states)), -np.inf)
    for j, state in enumerate(states):
        s, l = log_amplitudes(state, other, chunk)
        sign[:, j], logabs[:, j] = _segment_logsum(index, coeff, s, l, n)
    return sign, logabs


def lowering_norm(L: int, power: int) -> float:
    """|(L^-)^p |L, L>|^2 = prod over m = L .. L-p+1 of L(L+1) - m(m-1)."""
    return float(np.prod([L * (L + 1) - m * (m - 1) for m in range(L, L - power, -1)]))


class _FrozenSet(NaturalStateSet):
    """A `NaturalStateSet` that also takes a single state; only ever sampled, never trained."""

    def __init__(self, states):
        self._states = tuple(states)
        self._trainable = (True,) * len(self._states)


def sample_bank(
    phi_states: Sequence,
    nbank: int,
    nchains: int,
    initial_spins: np.ndarray,
    flip_prob: float = 0.5,
    thermal_steps: Optional[int] = None,
    sweep_steps: Optional[int] = None,
) -> np.ndarray:
    r"""
    ``nbank`` J-tuples (nbank, J, 2N) from :math:`|\det\Phi|^2`, :math:`\Phi_{ij} = \phi_j(y_i)`, by the NES sampler
    (pair moves and flavour flips) with ``nchains`` chains started from ``initial_spins`` (nchains, J, 2N) in the
    :math:`\phi`'s :math:`L_z` sector.
    """
    sampler = NaturalLzDetFlavourFlipSampler(
        _FrozenSet(phi_states), nchains, n_neighbor=np.arange(1, 3), initial_spins=jnp.asarray(initial_spins),
        reweight=2.0, flip_prob=flip_prob, thermal_steps=thermal_steps, sweep_steps=sweep_steps)
    draws, total = [], 0
    while total < nbank:
        draws.append(np.asarray(sampler.sweep().spins))
        total += len(draws[-1])
    return np.concatenate(draws)[:nbank]


class DeflationGroup:
    r"""
    The states :math:`\phi_j` of one :math:`L_z(\phi) = L_z + p`, highest weight of ``L`` (normally
    :math:`L_z(\phi)`), with their bank of J-tuples. Scales, all in float64: :math:`\Phi_{ij} = \tilde\Phi_{ij} e^{r_i + d_j}` and
    :math:`R_{ia} = \tilde R_{ia} e^{r_i + c_a}` on the bank, :math:`d_j` the largest :math:`\log|\phi_j|` there,
    :math:`r_i` each configuration's largest :math:`\log|\phi_j| - d_j`, :math:`c_a` the largest
    :math:`\log|R_{ia}| - r_i` of the member, so :math:`M_{ja} = e^{c_a - d_j} \hat M_{ja} / c` with
    :math:`\hat M = \mathrm{mean}(\tilde\Phi^{-1}\tilde R)`.
    """

    def __init__(self, phi_states: Sequence, L: int, power: int, N: int, bank_spins: np.ndarray, chunk: int = 65536):
        if power < 1:
            raise ValueError("deflation lowers: the power must be >= 1")
        self.states, self.power, self.N, self.chunk = list(phi_states), int(power), N, chunk
        J = len(self.states)
        self.bank = np.asarray(bank_spins).reshape(-1, J, 2 * N)
        nb = len(self.bank)
        flat = self.bank.reshape(-1, 2 * N)
        sign = np.zeros((nb * J, J))
        logabs = np.zeros((nb * J, J))
        for j, state in enumerate(self.states):
            sign[:, j], logabs[:, j] = log_amplitudes(state, flat, chunk)
        sign, logabs = sign.reshape(nb, J, J), logabs.reshape(nb, J, J)
        self.d = np.array([np.max(logabs[:, :, j][np.isfinite(logabs[:, :, j])]) for j in range(J)])
        rel = logabs - self.d[None, None, :]
        self.r = np.max(rel, axis=2)                                                   # (nb, J)
        ok = np.isfinite(rel)
        Phi = np.where(ok, sign * np.exp(np.where(ok, rel, 0.0) - self.r[:, :, None]), 0.0)
        self.Phi_inv = np.linalg.pinv(Phi)                                             # (nb, J, J)
        self.c_norm = lowering_norm(L, self.power)
        self._M = {}
        self._x_cache = None

    @property
    def J(self) -> int:
        return len(self.states)

    def refresh(self, states: NaturalStateSet) -> None:
        """:math:`\\hat M` and :math:`c_a` of every member, for the members as they stand."""
        nb, J = len(self.bank), self.J
        self._M = {}
        sign, logabs = applied_lplus(states.states, self.bank.reshape(-1, 2 * self.N), self.power, self.N, up=False,
                                     chunk=self.chunk)                                 # (nb J, K)
        for a, state in enumerate(states.states):
            rel = logabs[:, a].reshape(nb, J) - self.r
            ok = np.isfinite(rel)
            if not ok.any():                                                           # no hop anywhere
                self._M[id(state)] = (np.zeros(J), 0.0)
                continue
            c = np.max(rel[ok])
            R = np.where(ok, sign[:, a].reshape(nb, J) * np.exp(np.where(ok, rel, 0.0) - c), 0.0)
            self._M[id(state)] = (np.mean(np.einsum("nji,ni->nj", self.Phi_inv, R), axis=0), c)

    def coefficients(self, state) -> np.ndarray:
        """:math:`M_{ja} = (G_\\chi^{-1}\\langle\\chi|\\psi_a\\rangle)_j` of member ``state`` (last `refresh`)."""
        Mhat, c = self._M[id(state)]
        return Mhat * np.exp(c - self.d) / self.c_norm

    def amplification(self, nsub: int = 512) -> np.ndarray:
        r"""
        Eigenvalues of :math:`T = G_\phi^{-1}\langle\phi|(L^+)^p(L^-)^p|\phi\rangle / c` on the first ``nsub`` tuples
        of the bank: all 1 for exact highest-weight states of L; junk of L' in :math:`\phi` enters the lowered states
        amplified by its own :math:`|(L^-)^p|L',L\rangle|^2 / c` (~1400 for L' = 13, p = 2), so an eigenvalue
        :math:`\tau` means the lowered direction is a share :math:`1 - 1/\tau` junk, through which a member can mix
        with the intruder unpenalised (N = 8: fine up to ~1.5, escapes by ~5). ~(N/2)^4/64 evaluations per
        configuration.
        """
        J, n = self.J, min(nsub, len(self.bank))
        y = self.bank[:n].reshape(-1, 2 * self.N)
        index, bits, coeff = lplus_power(y, self.power, self.N, up=False)
        down = _from_bits(bits, 2 * self.N)
        chi_s, chi_l = (np.concatenate(v) for v in zip(*[
            applied_lplus(self.states, down[c:c + 8192], self.power, self.N, up=True, chunk=self.chunk)
            for c in range(0, len(down), 8192)])) if len(down) else (np.zeros((0, J)), np.zeros((0, J)))
        T = np.zeros((n, J, J))
        rel_all = []
        for j in range(J):
            s, l = _segment_logsum(index, coeff, chi_s[:, j], chi_l[:, j], n * J)
            rel_all.append((s, l))
        for j, (s, l) in enumerate(rel_all):
            rel = l.reshape(n, J) - self.d[j] - self.r[:n]
            ok = np.isfinite(rel)
            T[:, :, j] = np.where(ok, s.reshape(n, J) * np.exp(np.where(ok, rel, 0.0)), 0.0)
        Tm = np.mean(np.einsum("nji,nik->njk", self.Phi_inv[:n], T), axis=0) / self.c_norm
        return np.sort(np.real(np.linalg.eigvals(Tm)))

    def lowered(self, spins) -> tuple[np.ndarray, np.ndarray]:
        """sign and log|chi_j| on ``spins`` (n, 2N); the last call's result is reused for the same configurations."""
        spins = np.asarray(spins).reshape(-1, 2 * self.N)
        cached = self._x_cache
        if cached is not None and cached[0].shape == spins.shape and np.array_equal(cached[0], spins):
            return cached[1], cached[2]
        sign, logabs = applied_lplus(self.states, spins, self.power, self.N, up=True, chunk=self.chunk)
        self._x_cache = (spins.copy(), sign, logabs)
        return sign, logabs

    def ratio(self, state, spins, psi_sign, psi_log) -> np.ndarray:
        """:math:`(\\Pi_{group}\\psi_a)(x)/\\psi_a(x)` on ``spins`` for member ``state``."""
        if id(state) not in self._M:
            raise RuntimeError("DeflationGroup used before refresh() for these members")
        chi_sign, chi_log = self.lowered(spins)
        Mhat, c = self._M[id(state)]
        ok = np.isfinite(chi_log)
        expo = np.where(ok, chi_log, 0.0) - self.d[None, :] + c - psi_log[:, None]
        with np.errstate(over="ignore", invalid="ignore"):
            terms = np.where(ok, chi_sign * np.exp(expo) * Mhat[None, :], 0.0)
            return psi_sign * np.sum(terms, axis=1) / self.c_norm


class DeflationProjector:
    r"""
    :math:`\Pi = \sum_g \Pi_g` over `DeflationGroup` s, as an operator for `NaturalTraceEnergyGrad.local_energies`:
    ``Oloc(state, samples)`` returns :math:`(\Pi\psi_a)(x)/\psi_a(x)` for the member ``state``. `refresh` must be
    called with the members as they stand before the local energies of a sweep (`DeflatedTraceEnergyGrad` does).
    """

    def __init__(self, groups: Sequence[DeflationGroup]):
        self.groups = list(groups)

    @property
    def J(self) -> int:
        return sum(g.J for g in self.groups)

    def refresh(self, states: NaturalStateSet) -> None:
        for g in self.groups:
            g.refresh(states)

    def Oloc(self, state, samples) -> jax.Array:
        spins = np.asarray(samples.spins if isinstance(samples, Samples) else samples)
        spins = spins.reshape(-1, spins.shape[-1])
        psi = samples.psi if isinstance(samples, Samples) and samples.psi is not None else state(jnp.asarray(spins))
        psi = psi if isinstance(psi, LogArray) else LogArray.from_value(jnp.asarray(psi))
        psi_sign = np.real(np.asarray(psi.sign)).astype(np.float64)
        psi_log = np.asarray(psi.logabs, dtype=np.float64)
        return jnp.asarray(sum(g.ratio(state, spins, psi_sign, psi_log) for g in self.groups))


class DeflatedTraceEnergyGrad(NaturalTraceEnergyGrad):
    r"""
    `NaturalTraceEnergyGrad` of :math:`H + \beta\Pi (+ \lambda P)`. :math:`\Pi` is always evaluated as its own
    operator (it is no Quantax operator to merge); `energy_H` and `penalty_value` keep their meaning, and
    `deflation_value` is :math:`\mathrm{tr}(S^{-1}\Pi)`, the summed weight of the lowered states in the span (0 to
    J), with `VarDeflation` its sample variance.
    """

    def __init__(self, hamiltonian, projector: DeflationProjector, beta: float, penalty=None,
                 penalty_coeff: float = 0.0, penalty_every: int = 1):
        super().__init__(hamiltonian, penalty, penalty_coeff, penalty_every)
        self.projector, self.beta = projector, float(beta)
        self._deflation_value = self._VarDeflation = None

    @property
    def deflation_value(self):
        return self._deflation_value

    @property
    def VarDeflation(self):
        return self._VarDeflation

    def local_energies(self, states, samples, resolve: bool = True):
        self.projector.refresh(states)
        return super().local_energies(states, samples, resolve=resolve)

    def _operators(self, hamiltonian, resolve: bool) -> list:
        return super()._operators(hamiltonian, resolve) + [self.projector]

    def _combine(self, values, resolve: bool):
        *values, Eloc_Pi = values
        Eloc, Eloc_H, Eloc_P = super()._combine(values, resolve)
        self._Eloc_Pi = Eloc_Pi
        self._Eloc_total = Eloc + self.beta * Eloc_Pi
        return self._Eloc_total, Eloc_H, Eloc_P

    def ebar(self, states, samples):
        Ebar = super().ebar(states, samples)
        # the same samples and weights as `_record`'s
        rf = jnp.ones(samples.nsamples) if samples.reweight_factor is None else samples.reweight_factor
        finite = jnp.isfinite(self._Eloc_total)
        rf = jnp.where(finite, rf, 0.0)
        Pi = jnp.where(finite, self._Eloc_Pi, 0.0)
        mean = jnp.mean(Pi * rf)
        self._deflation_value = mean.real
        self._VarDeflation = jnp.mean(jnp.abs(Pi - mean) ** 2 * rf).real
        return Ebar
