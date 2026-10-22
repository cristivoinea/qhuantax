r"""Exact :math:`h=0` exciton states of the :math:`\nu=1` fuzzy sphere, as backflow starting points.

At :math:`h=0` only the inter-flavour density interaction acts, so the number :math:`k` of flipped
(down) electrons is conserved and every sector is a few-body problem on the fully polarised
vacuum :math:`|\Uparrow\rangle`. With :math:`b_{mn}=c^\dagger_{\downarrow m}c_{\uparrow n}` and
:math:`B_{\ell\mu}=\sum_{mn}t^{\ell\mu}_{mn}b_{mn}` (the multipole tensor of
``quantumhall_meanfield``):

* :math:`k=0` is :math:`|\Uparrow\rangle` and :math:`k=1` is :math:`B_{\ell\mu}|\Uparrow\rangle`, one
  state per :math:`\ell`. Both are **one determinant** inside the :math:`L_z` sector the sampler
  keeps (``vacuum_orbitals``, ``exciton_orbitals``); see ``docs/exciton_h0.pdf``, section 1.
* :math:`k=2` states are superpositions of two-exciton channels, found here by exact
  diagonalisation in the channel basis (``two_exciton_states``). They are not one determinant, so
  they are reached by supervised pretraining of a ``DetBackflow`` whose correction has rank 2
  (``pretrain_fidelity``). On the :math:`\theta=0` vacuum such an ansatz vanishes identically for
  :math:`k\ge3`, so its support is the handful of :math:`k\le2` configurations returned by
  ``support`` and the fidelity is an exact finite sum -- no sampling.
* Preferred for :math:`k\ge2`: the **block reference** (``block_orbitals``, ``block_backflow``,
  ``pretrain_masked``, ``k2_sector_checks``; ``docs/exciton_h0.pdf``, section 2). Its support is
  exactly sector :math:`k` by construction, so the exact sum needs no :math:`k+1` guard shell and
  no :math:`\delta` block, it trains at production width, and overlap and energy are checked
  exactly without ED at any :math:`N`. ``FuzzySphereExcitonCheck.py`` runs it (default ansatz),
  ``FuzzySphereExcitonLRSweep.py`` sweeps its learning rate.

* Any :math:`k`, and the response to a transverse field: ``KSectors`` works on the explicit
  configuration lists of each :math:`k` sector with quantax's own matrix elements, and solves
  first-order perturbation theory in :math:`h` exactly (``first_order``). A :math:`k_0` state then
  becomes a target on :math:`k_0\pm1` that a rank-:math:`(k_0+1)` backflow is pretrained on, with
  :math:`k_0+2` as the guard shell. Valid for :math:`k_0+1<N/2`, where the two Z2 branches do not
  share configurations.

Orbital matrices use the mode order of the rest of the package: row ``f*nm + m``, up (``f=0``)
first, one column per electron.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from qhuantax.quantumhall_meanfield import multipole_tensor
from qhuantax.quantumhall_operators import get_int_matrix


class H0Excitons:
    r"""Exact :math:`k\le2` exciton states at :math:`h=0`.

    :param nm: number of orbitals, :math:`2s+1`, equal to the particle number at :math:`\nu=1`.
    :param ps_pot: inter-flavour pseudopotentials exactly as passed to ``GetSpinfulDenIntTerms``
        (i.e. already including the factor 2 the scripts apply).
    """

    def __init__(self, nm: int, ps_pot):
        self.nm = nm
        U = get_int_matrix(nm, ps_pot)

        def u(a, b, c):
            return U[a, b, c] if 0 <= a + b - c < nm else 0.0

        # Hartree shift of one down electron in the full up band (uniform in m by rotation
        # invariance), and the particle-hole kernel K[(m', n'), (m, n)] = U(n, m', m) with
        # n' = n + m' - m, so that H_1 = E_H - K on b_{mn}|Up>.
        self.hartree = float(sum(u(k, 0, 0) for k in range(nm)))
        K = np.zeros((nm,) * 4)
        for m in range(nm):
            for n in range(nm):
                for mp in range(nm):
                    q = n + mp - m
                    if 0 <= q < nm:
                        K[mp, q, m, n] = u(n, mp, m)
        self.K = K
        self._t = {}

    # ------------------------------------------------------------------ k = 0, 1

    def t(self, ell: int, mu: int) -> np.ndarray:
        """Multipole tensor, indexed ``[down particle, up hole]``; cached."""
        if (ell, mu) not in self._t:
            self._t[(ell, mu)] = multipole_tensor(self.nm, ell, mu)
        return self._t[(ell, mu)]

    def vacuum_orbitals(self) -> np.ndarray:
        r""":math:`k=0`: :math:`\phi_j=c^\dagger_{\uparrow j}`, i.e. :math:`U=\binom{I}{0}`."""
        return np.vstack([np.eye(self.nm), np.zeros((self.nm, self.nm))])

    def backflow_vacuum(self, delta: float = 1e-2, seed: int = 0) -> np.ndarray:
        r"""The :math:`k=0` orbitals plus a random down block of size ``delta``: the ``U0`` for
        :math:`k=2` pretraining.

        On the exact vacuum every :math:`k\ge3` backflow matrix is singular, so
        :math:`\nabla\log|\det|` is infinite there and a loss that includes those configurations
        (see ``pretrain_fidelity``) returns NaN. The perturbation makes them small, of order
        ``delta``, and differentiable; ``delta = 1e-2`` costs nothing measurable at :math:`N=8`.
        """
        rng = np.random.default_rng(seed)
        return self.vacuum_orbitals() + np.vstack(
            [np.zeros((self.nm, self.nm)), delta * rng.standard_normal((self.nm, self.nm))])

    def exciton_orbitals(self, ell: int, mu: int) -> np.ndarray:
        r""":math:`k=1`: one determinant equal to :math:`B_{\ell\mu}|\Uparrow\rangle` in the
        :math:`L_z=\mu` sector.

        Every orbital gets a multiple of one delocalised mode,
        :math:`\phi_j=c^\dagger_{\uparrow j}+a_j\eta^\dagger` with
        :math:`\eta^\dagger=\sum_m(c^\dagger_{\downarrow m}+\lambda c^\dagger_{\uparrow m})`. As
        :math:`(\eta^\dagger)^2=0` no :math:`k\ge2` term appears, the :math:`L_z` restriction keeps
        only :math:`b_{j+\mu,j}`, and :math:`a_j=t^{\ell\mu}_{j+\mu,j}` makes that exactly
        :math:`B_{\ell\mu}`. The vacuum term :math:`(1+\lambda\sum_ja_j)|\Uparrow\rangle` has
        :math:`L_z=0`: it drops out for :math:`\mu\ne0`, is cancelled by :math:`\lambda` for
        :math:`\ell=\mu=0`, and cannot be cancelled for :math:`\mu=0,\ \ell\ge1`
        (:math:`\sum_ja_j=\mathrm{tr}\,t^{\ell0}=0`) -- use :math:`\mu\ge1` there.
        """
        nm = self.nm
        if not 0 <= mu <= ell:
            raise ValueError(f"need 0 <= mu <= ell, got ell={ell}, mu={mu}")
        if mu == 0 and ell > 0:
            raise ValueError(
                f"B_{{{ell}0}}|Up> is not one determinant at Lz=0: the vacuum term cannot be "
                f"cancelled because tr t^({ell},0) = 0. Use mu >= 1 (e.g. Lz = L)."
            )
        t = self.t(ell, mu)
        a = np.array([t[j + mu, j] if j + mu < nm else 0.0 for j in range(nm)])
        lam = -1.0 / a.sum() if mu == 0 else 0.0
        return np.vstack([np.eye(nm) + lam * np.outer(np.ones(nm), a), np.outer(np.ones(nm), a)])

    def exciton_terms(self, ell: int, mu: int) -> tuple[list[np.ndarray], list[float]]:
        r""":math:`k=1`: :math:`B_{\ell\mu}|\Uparrow\rangle` in the :math:`L_z=\mu` sector as a sum of
        determinants, ``(orbital matrices, coefficients)``: ``exciton_orbitals`` alone where it is
        exact, and at :math:`\mu=0,\ \ell\ge1` the determinant without the :math:`\lambda` term,
        :math:`|\Uparrow\rangle+B_{\ell0}|\Uparrow\rangle`, minus the vacuum."""
        if mu > 0 or ell == 0:
            return [self.exciton_orbitals(ell, mu)], [1.0]
        nm = self.nm
        a = np.diag(self.t(ell, 0)).copy()
        return [np.vstack([np.eye(nm), np.outer(np.ones(nm), a)]), self.vacuum_orbitals()], [1.0, -1.0]

    def exciton_energy(self, ell: int) -> float:
        r""":math:`E_\ell=E_H-\langle t^{\ell0}|K|t^{\ell0}\rangle/\langle t^{\ell0}|t^{\ell0}\rangle`,
        equal to the closed form :math:`\sum_l2V_l(2J+1)[\frac1{2s+1}-\{s\,s\,\ell;s\,s\,J\}]`."""
        v = self.t(ell, 0).ravel()
        return float(self.hartree - v @ self.K.reshape(self.nm ** 2, -1) @ v / (v @ v))

    # ------------------------------------------------------------------ k = 2

    @staticmethod
    def _anti(P):
        """Physical part of a two-exciton tensor: antisymmetric in the particles and in the holes."""
        return (P - P.transpose(2, 1, 0, 3) - P.transpose(0, 3, 2, 1) + P.transpose(2, 3, 0, 1)) / 4

    def _h2(self, P):
        r""":math:`H_2=2E_H-\sum_{i,j}K(\text{particle }i,\text{hole }j)` on ``P[m, n, m', n']``."""
        K = self.K
        o = 2 * self.hartree * P
        o -= np.einsum("abmn,mnpq->abpq", K, P)   # particle 1 - hole 1
        o -= np.einsum("abpq,mnpq->mnab", K, P)   # particle 2 - hole 2
        o -= np.einsum("abmq,mnpq->anpb", K, P)   # particle 1 - hole 2
        o -= np.einsum("abpn,mnpq->mbaq", K, P)   # particle 2 - hole 1
        return self._anti(o)

    def two_exciton_states(self, L: int, nstates: int = 1):
        r"""Lowest :math:`k=2` states of total :math:`L` at :math:`L_z=L`.

        Rayleigh-Ritz in the complete (over-complete) channel basis
        :math:`[B_{\ell_1}\otimes B_{\ell_2}]_{LL}|\Uparrow\rangle`, :math:`\ell_1\le\ell_2`, which is
        exact since those channels span the :math:`k=2`, :math:`L` sector.

        :return: ``(energies, tensors)``; ``tensors[i][m, n, m', n']`` is the normalised amplitude
            of :math:`\sum\Phi\,b_{mn}b_{m'n'}|\Uparrow\rangle` (particles ``m, m'``, holes ``n, n'``).
        """
        from sympy.physics.wigner import clebsch_gordan

        nm = self.nm
        vecs = []
        for l1 in range(nm):
            for l2 in range(l1, nm):
                if not abs(l1 - l2) <= L <= l1 + l2 or (l1 == l2 and L % 2):
                    continue
                P = np.zeros((nm,) * 4)
                for mu in range(-l1, l1 + 1):
                    if abs(L - mu) <= l2:
                        cg = float(clebsch_gordan(l1, l2, L, mu, L - mu, L))
                        if cg:
                            P += cg * np.einsum("mn,pq->mnpq", self.t(l1, mu), self.t(l2, L - mu))
                v = self._anti(P).ravel()
                if np.linalg.norm(v) > 1e-12:
                    vecs.append(v)
        B = np.array(vecs).T
        HB = np.array([self._h2(b.reshape((nm,) * 4)).ravel() for b in B.T]).T
        S = B.T @ B
        Hm = (B.T @ HB + (B.T @ HB).T) / 2
        s_, V = np.linalg.eigh(S)
        keep = s_ > 1e-10 * s_.max()
        X = V[:, keep] / np.sqrt(s_[keep])
        e, C = np.linalg.eigh(X.T @ Hm @ X)
        W = B @ (X @ C[:, :nstates])
        W /= np.linalg.norm(W, axis=0)
        return e[:nstates], [W[:, i].reshape((nm,) * 4) for i in range(nstates)]

    # ------------------------------------------------------------------ configurations

    def support(self, lz: int, kmax: int = 2):
        r"""All configurations with :math:`k\le k_{\max}` flipped electrons in the :math:`L_z` sector.

        :return: ``(spins, k, particles, holes)``: ``spins`` is ``(nconf, 2*nm)`` in quantax's
            :math:`\pm1` format (up modes first); ``particles``/``holes`` are sorted tuples of down
            orbitals and emptied up orbitals, so :math:`L_z=\sum\text{particles}-\sum\text{holes}`.
        """
        nm = self.nm
        spins, ks, parts, holes = [], [], [], []
        for k in range(kmax + 1):
            for P in combinations(range(nm), k):
                for Hh in combinations(range(nm), k):
                    if sum(P) - sum(Hh) != lz:
                        continue
                    s = -np.ones(2 * nm, dtype=np.int8)
                    s[[m for m in range(nm) if m not in Hh]] = 1
                    s[[nm + p for p in P]] = 1
                    spins.append(s); ks.append(k); parts.append(P); holes.append(Hh)
        return np.array(spins), np.array(ks), parts, holes

    def target_amplitudes(self, Phi: np.ndarray, particles, holes) -> np.ndarray:
        r"""Amplitudes of :math:`\sum\Phi\,b\,b|\Uparrow\rangle` on ``support`` configurations, in the
        convention of quantax's determinant models (:math:`\det U[\text{occupied}]`, modes ascending,
        up to a global sign).

        For particles :math:`p_1<p_2` and holes :math:`h_1<h_2` the configuration is
        :math:`\pm b_{p_1h_1}b_{p_2h_2}|\Uparrow\rangle`, and its sign is that of the permutation
        determinant :math:`\det(U_0+Y)[\text{occ}]` with :math:`Y` mapping :math:`p_i\to h_i` -- the
        same determinant quantax would evaluate. Configurations with :math:`k\ne2` get 0.
        """
        nm = self.nm
        A = 2 * self._anti(Phi)
        out = np.zeros(len(particles))
        for i, (P, Hh) in enumerate(zip(particles, holes)):
            if len(P) != 2:
                continue
            U = self.vacuum_orbitals()
            U[nm + P[0], Hh[0]] = 1.0
            U[nm + P[1], Hh[1]] = 1.0
            occ = sorted([m for m in range(nm) if m not in Hh] + [nm + p for p in P])
            sign = np.sign(np.linalg.det(U[occ]))
            out[i] = sign * A[P[0], Hh[0], P[1], Hh[1]]
        return out

    def k2_sign(self, P, Hh) -> float:
        r"""Sign of :math:`b_{p_1h_1}b_{p_2h_2}|\Uparrow\rangle` in the determinant convention,
        the one ``target_amplitudes`` uses; ``k2_sector_checks`` inverts it."""
        nm = self.nm
        U = self.vacuum_orbitals()
        U[nm + P[0], Hh[0]] = 1.0
        U[nm + P[1], Hh[1]] = 1.0
        occ = sorted([m for m in range(nm) if m not in Hh] + [nm + p for p in P])
        return float(np.sign(np.linalg.det(U[occ])))


# ---------------------------------------------------------------------- pretraining


def freeze_rank(model, rank: int = 2):
    r"""Zero all but the first ``rank`` columns of a ``DetBackflow``'s ``W``.

    The backflow correction is :math:`X(s)W^\top`, so its rank is bounded by the number of non-zero
    columns of ``W`` whatever the network width. Zero columns are also an exact way to grow the rank
    later: :math:`\psi` is unchanged, and since the features are :math:`\exp(\cdot)>0` the zero
    columns still receive gradient.
    """
    import equinox as eqx

    return eqx.tree_at(lambda m: m.W, model, model.W.at[:, rank:].set(0.0))


def pretrain_fidelity(model, spins, target, steps: int = 3000, lr: float = 1e-3,
                      rank: int = 2, log_every: int = 250, verbose: bool = True,
                      lr_t0: float | None = None):
    r"""Maximise the fidelity of a rank-``rank`` ``DetBackflow`` with a known target, summed exactly.

    :math:`F=|\sum_s\psi_\theta(s)t(s)|^2/(\sum_s|\psi_\theta(s)|^2\sum_s|t(s)|^2)` over ``spins``.
    Columns of ``W`` beyond ``rank`` are held at zero throughout (their gradient is masked).
    Full-batch Adam.

    ``spins`` should be every :math:`k\le3` configuration of the sector
    (``H0Excitons.support(lz, kmax=3)``, target 0 on :math:`k=3`), with ``U0`` from
    ``H0Excitons.backflow_vacuum``. In exact arithmetic a rank-2 correction on the vacuum vanishes
    for :math:`k\ge3`, but to push :math:`k=0,1` to zero the network drives its features over many
    decades, and the :math:`k=3` determinants -- singular on paper -- then come out at
    :math:`\sim10\%` of the :math:`k=2` ones in floating point. A loss over :math:`k\le2` alone does
    not see this (at :math:`N=8`: loss :math:`1-F=10^{-4}`, true fidelity :math:`0.990`); the
    :math:`k=3` shell makes the norm honest (true fidelity :math:`0.999996`, leftover weight
    :math:`5\cdot10^{-12}` at :math:`k=3` and :math:`10^{-22}` at :math:`k=4`).

    ``lr_t0`` switches on the decay ``lr / (1 + step / lr_t0)`` (the form of
    ``adaptive_learning_rate_inv``); ``None`` keeps ``lr`` constant.

    :return: ``(model, history)`` with ``history`` a list of ``(step, 1 - F)``.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    model = freeze_rank(model, rank)
    spins = jnp.asarray(spins)
    tgt = jnp.asarray(target / np.linalg.norm(target))
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def infidelity(p):
        psi = jax.vmap(eqx.combine(p, static))(spins)
        amp = psi.sign * jnp.exp(psi.logabs - jnp.max(psi.logabs))
        return 1.0 - (amp @ tgt) ** 2 / (amp @ amp)

    value_and_grad = eqx.filter_jit(jax.value_and_grad(infidelity))
    W_mask = jnp.zeros_like(model.W).at[:, :rank].set(1.0)

    def mask(g):
        return eqx.tree_at(lambda m: m.W, g, g.W * W_mask)

    m1 = jax.tree_util.tree_map(jnp.zeros_like, params)
    m2 = jax.tree_util.tree_map(jnp.zeros_like, params)
    b1, b2, eps = 0.9, 0.999, 1e-12
    history = []
    for it in range(1, steps + 1):
        lr_it = lr if lr_t0 is None else lr / (1 + it / lr_t0)
        val, g = value_and_grad(params)
        g = mask(g)
        m1 = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m1, g)
        m2 = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, m2, g)
        params = jax.tree_util.tree_map(
            lambda p, a, b: p - lr_it * (a / (1 - b1 ** it)) / (jnp.sqrt(b / (1 - b2 ** it)) + eps),
            params, m1, m2)
        if it % log_every == 0 or it == 1 or it == steps:
            history.append((it, float(val)))
            if verbose:
                print(f"    step {it:5d}  1-F = {float(val):.3e}", flush=True)
    return freeze_rank(eqx.combine(params, static), rank), history


# ---------------------------------------------------------------------- fixed-k sectors


def spins_key(spins) -> np.ndarray:
    """Integer key of each configuration: bit ``i`` set when mode ``i`` is occupied."""
    spins = np.atleast_2d(np.asarray(spins))
    return (np.where(spins > 0, 1, 0).astype(np.int64)
            << np.arange(spins.shape[1], dtype=np.int64)).sum(1)


class KSectors:
    r"""The configurations with exactly :math:`k` flipped electrons in one :math:`L_z` sector, and
    the blocks of an operator between them.

    The generalisation of the channel algebra of ``H0Excitons`` to any :math:`k`: sectors are plain
    configuration lists and operators act through quantax's own matrix elements
    (``apply_diag``/``apply_off_diag``), so signs are those of the models and of ED, and no
    :math:`k`-specific formula is needed. :math:`H_\text{int}` is block diagonal in :math:`k`, a
    transverse field connects :math:`k` to :math:`k\pm1`, and ``first_order`` solves the resulting
    perturbation theory exactly on the lists. Sector dimensions grow as
    :math:`\binom{n_m}{k}^2/n_{L_z}`, so this is meant for :math:`k\lesssim4`.

    Configurations are unsymmetrised, on the :math:`|\Uparrow\rangle` branch: for :math:`k<N/2` the
    Z2 images (:math:`k\to N-k`) never enter.

    :param nm: number of orbitals, equal to the particle number at :math:`\nu=1`.
    :param lz: the sector, :math:`L_z=\sum\text{particles}-\sum\text{holes}` relative to the vacuum.
    """

    def __init__(self, nm: int, lz: int):
        self.nm = nm
        self.lz = lz
        self._configs = {}
        self._blocks = {}

    def configs(self, k: int) -> np.ndarray:
        """``(nconf, 2*nm)`` spins with exactly ``k`` flips, sorted by ``spins_key``."""
        if k not in self._configs:
            nm = self.nm
            by_sum = {}
            for c in combinations(range(nm), k):
                by_sum.setdefault(sum(c), []).append(c)
            rows = []
            for sp, parts in by_sum.items():
                for Hh in by_sum.get(sp - self.lz, []):
                    for P in parts:
                        s = -np.ones(2 * nm, dtype=np.int8)
                        s[[m for m in range(nm) if m not in Hh]] = 1
                        s[[nm + p for p in P]] = 1
                        rows.append(s)
            spins = np.array(rows, dtype=np.int8).reshape(-1, 2 * nm)
            self._configs[k] = spins[np.argsort(spins_key(spins))]
        return self._configs[k]

    def dim(self, k: int) -> int:
        return len(self.configs(k))

    def block(self, op, k_out: int, k_in: int, chunk: int | None = None):
        r""":math:`\langle s'|O|s\rangle` for ``s'`` in sector ``k_out`` and ``s`` in ``k_in``, as a
        scipy CSR matrix. Connected configurations outside ``k_out`` are dropped.

        ``apply_off_diag`` builds every connected configuration of a batch at once, so memory goes as
        ``chunk`` x (number of terms): 68 GB for 2048 configurations of :math:`H_0` at :math:`N=24`.
        By default ``chunk`` keeps that product below :math:`10^6` (a few GB)."""
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix

        cache_key = (id(op), k_out, k_in)
        if cache_key in self._blocks:
            return self._blocks[cache_key]
        if chunk is None:
            nterms = sum(len(term.strength) for term in op.op_list)
            chunk = int(np.clip(1_000_000 // max(nterms, 1), 16, 2048))

        s_in = self.configs(k_in)
        keys_out = spins_key(self.configs(k_out))
        n_out, n_in = len(keys_out), len(s_in)

        rows, cols, vals = [], [], []
        for start in range(0, n_in, chunk):
            # both take a batch; off-diagonal output is (batch, nterms, Nmodes), with nan
            # strengths for terms that annihilate the configuration
            s = jnp.asarray(s_in[start:start + chunk])
            idx = np.arange(start, start + len(s))
            if k_out == k_in:
                rows.append(idx); cols.append(idx); vals.append(np.real(np.asarray(op.apply_diag(s))))
            for _, s_conn, strength in op.apply_off_diag(s):
                s_conn, strength = np.asarray(s_conn), np.real(np.asarray(strength))
                col = np.broadcast_to(idx[:, None], strength.shape).ravel()
                s_conn = s_conn.reshape(-1, s_conn.shape[-1])
                strength = strength.ravel()
                keep = np.isfinite(strength) & (strength != 0)
                key = spins_key(s_conn[keep])
                pos = np.clip(np.searchsorted(keys_out, key), 0, max(n_out - 1, 0))
                hit = keys_out[pos] == key if n_out else np.zeros(len(key), bool)
                rows.append(pos[hit]); cols.append(col[keep][hit]); vals.append(strength[keep][hit])
        M = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(n_out, n_in))
        M.sum_duplicates()
        self._blocks[cache_key] = M
        return M

    def _shifted(self, H0, k, L2, mu):
        r""":math:`H_0` on sector ``k`` plus :math:`\mu(L^2-L_z(L_z+1))`, which is :math:`\ge0` and
        vanishes exactly on :math:`L=L_z`, so it lifts every other :math:`L` without touching the
        states :math:`L=L_z` that the rest of this class works with."""
        from scipy.sparse import identity

        A = self.block(H0, k, k)
        if L2 is not None and mu:
            A = A + mu * (self.block(L2, k, k) - self.lz * (self.lz + 1) * identity(A.shape[0]))
        return A.tocsr()

    def eigenstates(self, H0, k: int, nstates: int = 1, L2=None, nlevels: int = 20, L: int | None = None):
        r"""Lowest ``nstates`` eigenstates of ``H0`` with exactly ``k`` flips and angular momentum ``L``
        (default :math:`L_z`; when ``L2`` is given, otherwise any :math:`L\ge L_z`). As in
        ``FuzzySphereNESTrain``: the lowest ``nlevels`` levels of the block, every :math:`L`, keeping
        those with that :math:`L` by :math:`\langle L^2\rangle`; ``nlevels`` doubles until ``nstates``
        of them are found.

        :return: ``(energies, vectors)``, ``vectors[:, i]`` on ``configs(k)``.
        """
        from scipy.sparse.linalg import eigsh

        target = self.lz if L is None else L
        A = self.block(H0, k, k)
        L2k = None if L2 is None else self.block(L2, k, k)
        n = A.shape[0]
        nlevels = max(nlevels, nstates) if L2 is not None else nstates
        while True:
            if nlevels >= n - 1:
                e, V = np.linalg.eigh(A.toarray())
            else:
                e, V = eigsh(A, k=nlevels, which="SA", tol=1e-12)
                o = np.argsort(e); e, V = e[o], V[:, o]
            if L2k is None:
                return e[:nstates], V[:, :nstates]
            q = np.einsum("si,si->i", V, L2k @ V)
            Ls = np.round((-1 + np.sqrt(1 + 4 * np.clip(q, 0, None))) / 2)
            keep = np.flatnonzero(Ls == target)[:nstates]
            if len(keep) == nstates or nlevels >= n - 1:
                break
            nlevels = 2 * nlevels
        if len(keep) < nstates:
            raise ValueError(f"k={k}: only {len(keep)} levels with L={target} in the whole sector")
        off = np.abs(q[keep] - target * (target + 1))
        if np.any(off > 1e-6):
            raise ValueError(f"k={k}: <L^2> of the L={target} levels off by {off}: degenerate levels of "
                             "different L mix")
        return e[keep], V[:, keep]

    def first_order(self, H0, V, k0: int, psi0: np.ndarray, E0: float, L2=None, mu: float = 10.0):
        r"""First-order response of an exact ``H0`` eigenstate to :math:`H=H_0-hV`, with :math:`V`
        changing :math:`k` by :math:`\pm1` (e.g. ``GetSpinfulPolTerms`` with :math:`\sigma_x`).

        :math:`\psi(h)=\psi_0+h\psi_1+O(h^2)`, :math:`\psi_1=(H_0-E_0)^{-1}V\psi_0` on the sectors
        :math:`k_0\pm1`, and :math:`E(h)=E_0+h^2E_2+O(h^4)` with
        :math:`E_2=-\langle\psi_0|V|\psi_1\rangle`. :math:`V\psi_0` keeps the :math:`L` of
        :math:`\psi_0`, so with ``L2`` the solve is done on the :math:`L`-shifted blocks
        (``_shifted``), which leaves it unchanged and removes accidental degeneracies with other
        :math:`L`. A same-:math:`L` level of :math:`H_0` at :math:`E_0` in :math:`k_0\pm1` makes the
        problem degenerate, which is reported rather than solved.

        :return: ``(psi1, E2)``, ``psi1`` a dict ``{k: vector on configs(k)}``.
        """
        from scipy.sparse import identity
        from scipy.sparse.linalg import spsolve

        psi1, E2 = {}, 0.0
        for k in (k0 - 1, k0 + 1):
            if not 0 <= k <= self.nm or self.dim(k) == 0:
                continue
            rhs = self.block(V, k, k0) @ psi0
            if not np.any(rhs):
                continue
            A = (self._shifted(H0, k, L2, mu) - E0 * identity(self.dim(k))).tocsc()
            x = spsolve(A, rhs) if A.shape[0] > 1 else rhs / A.toarray()[0, 0]
            x = np.atleast_1d(x)
            resid = np.linalg.norm(A @ x - rhs) / np.linalg.norm(rhs)
            if not np.all(np.isfinite(x)) or resid > 1e-8:
                raise ValueError(f"k={k}: (H0 - E0) is singular on V psi0 (residual {resid:.1e}); "
                                 "a same-L level of the neighbouring sector is degenerate with E0")
            psi1[k] = x
            E2 -= float(rhs @ x)
        return psi1, E2

    def target(self, pieces: dict, kmax: int):
        r"""Stack sector vectors ``{k: vector}`` into ``(spins, amplitudes)`` over every
        :math:`k\le k_{\max}` (zeros where ``pieces`` has none), the input of
        ``pretrain_fidelity``."""
        spins = [self.configs(k) for k in range(kmax + 1)]
        amps = [np.asarray(pieces[k], float) if k in pieces else np.zeros(len(s))
                for k, s in zip(range(kmax + 1), spins)]
        return np.concatenate(spins), np.concatenate(amps)


# ---------------------------------------------------------------------- block reference for k >= 2
#
# docs/exciton_h0.pdf, section 2. On the vacuum, a rank-2 correction gives psi_{k=2} = det X_down *
# det W_holes exactly, but psi_{k<=1} = 0 needs the fine-tuning G(s) = -1 on every k<=1
# configuration, which is what forced the k=3 guard shell and the delta block of
# ``pretrain_fidelity``. The block reference removes both: the support is exactly sector k by
# construction, so the exact sum runs over sector k alone.


def hole_frame(nm: int, k: int) -> np.ndarray:
    r"""An ``(nm, k)`` orthonormal frame whose :math:`k\times k` minors never vanish.

    :math:`k=2`: the span of :math:`\cos,\sin(\pi(m+\frac12)/n_m)`, whose minors are
    :math:`\propto\sin(\pi(m_2-m_1)/n_m)>0` (the frame of every test so far). Otherwise a
    Vandermonde frame :math:`x_m^j`, :math:`x_m=(m+\frac12)/n_m`, which is totally positive;
    orthonormalising multiplies every minor by the same constant.
    """
    if k == 2:
        th = np.pi * (np.arange(nm) + 0.5) / nm
        F = np.stack([np.cos(th), np.sin(th)], 1)
    else:
        F = ((np.arange(nm) + 0.5) / nm)[:, None] ** np.arange(k)[None, :]
    return np.linalg.qr(F)[0]


def block_orbitals(nm: int, k: int, holes: np.ndarray | None = None,
                   particles: np.ndarray | None = None, seed: int = 0):
    r"""The sector-:math:`k` block reference :math:`U_0=\begin{pmatrix}A&0\\0&B\end{pmatrix}` and the
    mask of its trainable entries.

    Columns ``0..nm-k-1`` are *core*, the last ``k`` are *excitation* columns. Up rows reach only
    the core (``A``, an orthonormal complement of ``holes``), down rows only the excitation columns
    (``B = particles``). The static amplitude is :math:`\pm\det(\text{holes}[H])
    \det(\text{particles}[P])` on sector :math:`k` and 0 elsewhere.

    :param holes: ``(nm, k)`` hole frame :math:`a_\perp`; default ``hole_frame(nm, k)``. For
        :math:`k=1`, the hole weights :math:`a_j=t^{\ell\mu}_{j+\mu,j}` with ``particles = 1`` give
        :math:`B_{\ell\mu}|\Uparrow\rangle` exactly for every :math:`\mu` (derived, not yet run).
    :param particles: ``(nm, k)`` particle frame ``B``; default ``hole_frame(nm, k)``.
    :param seed: seed of the random completion that fixes ``A`` within the complement.
    :return: ``(U0, U0_mask)``, both ``(2*nm, nm)``; the mask is 1 on the two diagonal blocks.
    """
    holes = hole_frame(nm, k) if holes is None else np.asarray(holes, float).reshape(nm, k)
    particles = (hole_frame(nm, k) if particles is None
                 else np.asarray(particles, float).reshape(nm, k))
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(np.hstack([holes, rng.standard_normal((nm, nm - k))]))[0]
    U0 = np.zeros((2 * nm, nm))
    U0[:nm, :nm - k] = Q[:, k:]
    U0[nm:, nm - k:] = particles
    mask = np.zeros_like(U0)
    mask[:nm, :nm - k] = 1.0
    mask[nm:, nm - k:] = 1.0
    return U0, mask


def block_backflow(model, k: int, gap: float = 20.0):
    r"""Restrict a ``DetBackflow`` built on ``block_orbitals(..., k)`` to sector :math:`k`.

    * ``W`` is zeroed on the core columns (its first ``N-k`` rows), so the correction reaches only
      the :math:`k` excitation columns. Its rank is then bounded by :math:`k` at any width
      :math:`d`, so every channel is used and no rank freeze is needed.
    * The up-mode features are suppressed. The last FFN's bias ``b2`` gets
      :math:`-g\sqrt{n_b+1}` on the even channels, which ``DetBackflow`` maps to up modes
      (``x.reshape(d/2, 2N)``). After ``final_layer``'s rescaling, :math:`x_\uparrow\sim e^{-g}`.

    More than :math:`k` down electrons then give a rank-deficient matrix (:math:`\psi=0`
    identically), and fewer give :math:`O(e^{-g})`. At :math:`N=12` with :math:`g=20`: weight 0 at
    :math:`k\ge3` and :math:`10^{-17}` at :math:`k=1`. For a later transfer to :math:`h>0` with SR,
    prefer :math:`g\approx8`, since the suppressed features have gradients :math:`\propto e^{-g}`.

    :return: ``(model, masks)``, with ``masks = {"W": ..., "U0": ...}`` for ``pretrain_masked``.
    """
    import equinox as eqx
    import jax.numpy as jnp

    Ntot = model.W.shape[0]
    W_mask = jnp.zeros_like(model.W).at[Ntot - k:, :].set(1.0)
    model = eqx.tree_at(lambda m: m.W, model, model.W * W_mask)
    b2 = model.net.layers[-2].b2.at[0::2].set(-gap * float(np.sqrt(model.net.nblocks + 1)))
    model = eqx.tree_at(lambda m: m.net.layers[-2].b2, model, b2)
    U0_mask = (jnp.zeros_like(model.U0).at[:Ntot, :Ntot - k].set(1.0)
               .at[Ntot:, Ntot - k:].set(1.0))
    return model, {"W": W_mask, "U0": U0_mask}


def release_block(model, masks: dict, delta_up: float = 1e-2, delta_down: float = 1e-3,
                  seed: int = 0):
    r"""Open a pretrained block-reference ``DetBackflow`` to the neighbouring sectors, for training
    at :math:`h>0`.

    The zeros that confine the block member to sector :math:`k` are structural: the off-diagonal
    blocks of :math:`U_0` and the core rows of :math:`W`. Released as they are, the
    :math:`k+1` amplitudes stay exactly 0 and the :math:`k-1` ones :math:`O(e^{-g})`, where no
    sampler -- reweighted or not -- puts samples, while the field couples :math:`k` to exactly
    those sectors at first order. Their gradient does not open them either: on sector :math:`k` the
    matrix is block triangular, so the derivative with respect to those entries is
    :math:`O(e^{-g})`. This fills them with Gaussian noise instead, the block counterpart of the
    random down block (``backflow_vacuum``) the :math:`k\le1` members start from.

    * ``delta_up`` fills up rows :math:`\times` excitation columns of :math:`U_0`: opens
      :math:`k-1`.
    * ``delta_down`` fills down rows :math:`\times` core columns of :math:`U_0` and the core rows of
      :math:`W`: opens :math:`k+1`.

    At :math:`N=8`, :math:`k=2`, ``delta_up = delta_down = 1e-3`` gave weight :math:`5\cdot10^{-7}`
    at :math:`k=1` and :math:`4\cdot10^{-4}` at :math:`k=3`, at no measurable cost in fidelity;
    ``1e-2`` on both already moved 4% of the state to :math:`k=3`. The masks are not needed after
    this: the returned model is trained unconstrained, bias included.
    """
    import equinox as eqx

    rng = np.random.default_rng(seed)
    U0_mask, W_mask = np.asarray(masks["U0"]), np.asarray(masks["W"])
    Ntot = U0_mask.shape[1]
    off = 1.0 - U0_mask
    up = np.zeros_like(off); up[:Ntot] = off[:Ntot]
    dU = (delta_up * up + delta_down * (off - up)) * rng.standard_normal(off.shape)
    dW = delta_down * (1.0 - W_mask) * rng.standard_normal(W_mask.shape)
    model = eqx.tree_at(lambda m: m.U0, model, model.U0 + dU.astype(model.U0.dtype))
    return eqx.tree_at(lambda m: m.W, model, model.W + dW.astype(model.W.dtype))


def pretrain_masked(model, spins, target, masks: dict | None = None, steps: int = 4000,
                    lr: float = 1e-2, lr_t0: float | None = 1000.0, keep_best: bool = True,
                    log_every: int = 250, verbose: bool = True):
    r"""Maximise the exact fidelity of ``model`` with ``target`` over ``spins``. The gradient of
    each leaf named in ``masks`` (``{"W": ..., "U0": ...}``, from ``block_backflow``) is multiplied
    by its mask.

    Full-batch Adam with :math:`\mathrm{lr}/(1+t/t_0)`. With the block reference, ``spins`` is
    sector :math:`k` alone (e.g. ``H0Excitons.support(lz, kmax=2)``, whose :math:`k<2` entries
    have target 0 and amplitude :math:`O(e^{-g})`); there is no guard shell. The defaults are the
    optimum of the sweep in ``docs/exciton_h0.pdf`` (:math:`N=10,12,16`, :math:`L=0`, :math:`d=8`,
    :math:`n_b=4`, script ``FuzzySphereExcitonLRSweep.py``): ``lr = 1e-2`` at every :math:`N`
    (larger is worse) and ``lr_t0 ~ steps/4``. Steps grow with :math:`N`: ~4k at :math:`N=12`,
    ~8k at :math:`N=16`. The loss spikes late in training, so with ``keep_best`` the lowest-loss
    parameters are returned.

    :return: ``(model, history)``, ``history`` a list of ``(step, 1 - F)``; with ``keep_best`` the
        last entry is the best loss.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    masks = masks or {}
    spins = jnp.asarray(spins)
    tgt = jnp.asarray(target / np.linalg.norm(target))
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def infidelity(p):
        psi = jax.vmap(eqx.combine(p, static))(spins)
        amp = psi.sign * jnp.exp(psi.logabs - jnp.max(psi.logabs))
        return 1.0 - (amp @ tgt) ** 2 / (amp @ amp)

    value_and_grad = eqx.filter_jit(jax.value_and_grad(infidelity))

    def mask(g):
        for name, M in masks.items():
            g = eqx.tree_at(lambda m, n=name: getattr(m, n), g, getattr(g, name) * M)
        return g

    m1 = jax.tree_util.tree_map(jnp.zeros_like, params)
    m2 = jax.tree_util.tree_map(jnp.zeros_like, params)
    b1, b2, eps = 0.9, 0.999, 1e-12
    history, best = [], (np.inf, params)
    for it in range(1, steps + 1):
        lr_it = lr if lr_t0 is None else lr / (1 + it / lr_t0)
        val, g = value_and_grad(params)
        val = float(val)
        if keep_best and val < best[0]:
            best = (val, params)        # the loss of `params` before this update
        g = mask(g)
        m1 = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m1, g)
        m2 = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, m2, g)
        params = jax.tree_util.tree_map(
            lambda p, a, b: p - lr_it * (a / (1 - b1 ** it)) / (jnp.sqrt(b / (1 - b2 ** it)) + eps),
            params, m1, m2)
        if it % log_every == 0 or it == 1 or it == steps:
            history.append((it, val))
            if verbose:
                print(f"    step {it:5d}  1-F = {val:.3e}", flush=True)
    if keep_best:
        final = float(value_and_grad(params)[0])
        if final < best[0]:
            best = (final, params)
        params = best[1]
        history.append((steps, best[0]))
        if verbose:
            print(f"    best       1-F = {best[0]:.3e}", flush=True)
    return eqx.combine(params, static), history


def k2_sector_checks(ex: H0Excitons, model, spins, ks, particles, holes, target, E_exact=None):
    r"""Exact overlap and :math:`\langle H\rangle` of ``model`` inside sector :math:`k=2`, without ED.

    :math:`N_\downarrow` is conserved at :math:`h=0`, so the target is an exact eigenstate and
    :math:`H` maps sector 2 to itself. Both numbers are therefore exact whenever the model vanishes
    outside :math:`k\le2`, as the block reference does. :math:`\langle H\rangle` is the Rayleigh
    quotient of the amplitude tensor under ``H0Excitons._h2``, as in ``two_exciton_states``; it
    agrees with ED to all digits at :math:`N=8`. ``model`` is unsymmetrised (its Z2 image lives in
    :math:`k=N-2`). Arguments are those of ``support`` and ``target_amplitudes``.

    :return: dict with ``F`` (overlap squared with ``target``), ``E``, ``w_lower`` (weight on the
        configurations of ``spins`` outside :math:`k=2`) and ``E_exact`` if given.
    """
    import jax
    import jax.numpy as jnp

    nm = ex.nm
    out = jax.vmap(model)(jnp.asarray(spins))
    psi = np.asarray(out.sign * jnp.exp(out.logabs - jnp.max(out.logabs)), dtype=float)
    target = np.asarray(target, float)
    k2 = np.asarray(ks) == 2
    F = float((psi[k2] @ target[k2]) ** 2 / (psi @ psi) / (target @ target))
    A = np.zeros((nm,) * 4)
    for i in np.flatnonzero(k2):
        (p1, p2), (h1, h2) = particles[i], holes[i]
        v = psi[i] * ex.k2_sign(particles[i], holes[i])
        A[p1, h1, p2, h2] = v
        A[p2, h1, p1, h2] = -v
        A[p1, h2, p2, h1] = -v
        A[p2, h2, p1, h1] = v
    res = dict(F=F, E=float(np.sum(A * ex._h2(A)) / np.sum(A * A)),
               w_lower=float(psi[~k2] @ psi[~k2] / (psi @ psi)))
    if E_exact is not None:
        res["E_exact"] = float(E_exact)
    return res


# ---------------------------------------------------------------------- transverse field: rotation
#
# At h > 0 the HF vacuum is |Up> with every flavour spinor rotated to (cos th/2, sin th/2). The
# same one-body rotation R(theta) = exp(theta/2 sum_m (c+_dn,m c_up,m - h.c.)) carries every h=0
# exciton state along: it commutes with Lz and L^2, so a single determinant stays a single
# determinant (rows rotated, ``rotation_orbitals``), while a backflow member has to be refitted to
# R|psi> (``FlavourRotation``, ``pretrain_rotated``).


def rotation_orbitals(nm: int, theta: float) -> np.ndarray:
    r"""The one-body rotation on the rows of an orbital matrix, ``(2*nm, 2*nm)``.

    ``rotation_orbitals(nm, theta) @ U`` is the determinant of :math:`R(\theta)|\Phi(U)\rangle`: an
    up row goes to :math:`(\cos\frac\theta2,\sin\frac\theta2)`, a down row to
    :math:`(-\sin\frac\theta2,\cos\frac\theta2)`, so the vacuum becomes ``chart(hf_vacuum, 0)``.
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    eye = np.eye(nm)
    return np.block([[c * eye, -s * eye], [s * eye, c * eye]])


def rotated_vacuum_configs(nm: int, n: int, theta: float, lz: int = 0, both: bool = False,
                           rng: np.random.Generator | None = None) -> np.ndarray:
    r"""``n`` random configurations distributed like the rotated vacuum :math:`R(\theta)|\Uparrow\rangle`,
    as starting points for Markov chains: one electron per orbital, each down with probability
    :math:`\sin^2(\theta/2)` (with ``both``, each configuration takes :math:`\cos^2(\theta/2)` instead
    with probability 1/2, the Z2 image). For :math:`L_z\ne0` random electrons are then moved up
    (:math:`L_z>0`) or down, each by a random number of orbitals onto an empty mode of the target orbital
    (either flavour), until the total is ``lz``. Quantax layout (up modes, then down modes), :math:`\pm1`."""
    rng = np.random.default_rng() if rng is None else rng
    if abs(lz) > nm * nm // 2:
        raise ValueError(f"|Lz| = {abs(lz)} is out of reach of {nm} electrons in {nm} orbitals")
    p = np.sin(theta / 2) ** 2
    out = -np.ones((n, 2 * nm), np.int8)
    for r in range(n):
        q = 1 - p if both and rng.random() < 0.5 else p
        occ = np.zeros((2, nm), bool)                     # occ[flavour, orbital]
        occ[(rng.random(nm) < q).astype(int), np.arange(nm)] = True
        left, tries = lz, 0
        while left != 0:
            tries += 1
            if tries > 100 * nm * nm:
                raise RuntimeError(f"could not reach Lz = {lz}")
            f, m = rng.choice(np.argwhere(occ))
            step = int(np.sign(left))
            reach = min(abs(left), nm - 1 - m if step > 0 else m)
            if reach < 1:
                continue
            t = m + step * int(rng.integers(1, reach + 1))
            empty = np.flatnonzero(~occ[:, t])
            if len(empty) == 0:
                continue
            occ[f, m], occ[rng.choice(empty), t] = False, True
            left -= t - m
        out[r, occ.ravel()] = 1
    return out


def rotated_vacuum_tuples(nm: int, ntuples: int, K: int, theta: float, lz: int = 0, both: bool = False,
                          rng: np.random.Generator | None = None) -> np.ndarray:
    """``(ntuples, K, 2 nm)`` seeds for the NES sampler from ``rotated_vacuum_configs``, the K of a tuple distinct
    (rotated members are nonzero on all of them, so the tuple's determinant is too)."""
    rng = np.random.default_rng() if rng is None else rng
    out = rotated_vacuum_configs(nm, ntuples * K, theta, lz, both, rng).reshape(ntuples, K, 2 * nm)
    for t in range(ntuples):
        while len({row.tobytes() for row in out[t]}) < K:
            out[t] = rotated_vacuum_configs(nm, K, theta, lz, both, rng)
    return out


def _vacuum_energy(nm: int, theta: float, z2: int, hamiltonian) -> float:
    """``vacuum_energy`` without the dtype switch; the caller holds a complex default dtype."""
    from qhuantax.quantumhall_meanfield import HFVacuum, _ratio, chart, reduced_matrices

    vac = HFVacuum(nm, float(theta))
    U = chart(vac, np.zeros((nm, nm)))
    if np.sin(theta) == 0:
        # |Up> (or |Down>) and its Z2 image are orthogonal and H, at most two-body, does not couple them: the
        # projection leaves the energy, while the projected Wick ratio is 0/0
        return float(_ratio(U[None], np.ones(1), hamiltonian).real)
    Hm, S = reduced_matrices(vac, U[None], 0, z2, 1, hamiltonian)
    return float(Hm[0, 0] / S[0, 0])


def _complex_dtype(fn):
    """Runs ``fn()`` under a complex default dtype: the projected determinants carry complex
    coefficients, which quantax's mean-field expectation accepts only then; restored afterwards."""
    import jax.numpy as jnp
    import quantax as qtx
    from quantax.global_defs import get_default_dtype

    previous = get_default_dtype()
    qtx.set_default_dtype(jnp.complex128)
    try:
        return fn()
    finally:
        qtx.set_default_dtype(previous)


def vacuum_energy(nm: int, theta: float, z2: int, hamiltonian) -> float:
    r"""The exact :math:`Z_2`-projected energy of the vacuum rotated by ``theta``, in sector
    ``z2``: Wick on the projected determinant pair (``quantumhall_meanfield.reduced_matrices``
    with one gauge angle, exact at :math:`L_z=0`)."""
    return _complex_dtype(lambda: _vacuum_energy(nm, theta, z2, hamiltonian))


def vacuum_angle(nm: int, ps_pot, h: float, z2: int, hamiltonian, xatol: float = 1e-6):
    r"""The canting angle that minimises the exact :math:`Z_2`-projected energy of the rotated
    vacuum, and that energy.

    The energy (``vacuum_energy``) is minimised over :math:`\theta\in(0,\pi/2)`, which covers
    every distinct state since :math:`Z_2` identifies :math:`\theta` with :math:`\pi-\theta`. The
    closed-form ``hf_vacuum`` angle is returned alongside for comparison.

    :return: ``(theta, energy, theta_hf)``.
    """
    from scipy.optimize import minimize_scalar

    from qhuantax.quantumhall_meanfield import hf_vacuum

    theta_hf = hf_vacuum(nm, ps_pot, h, z2=z2).theta
    res = _complex_dtype(lambda: minimize_scalar(lambda t: _vacuum_energy(nm, t, z2, hamiltonian),
                                                 bounds=(1e-4, np.pi / 2 - 1e-4), method="bounded",
                                                 options=dict(xatol=xatol)))
    return float(res.x), float(res.fun), float(theta_hf)


class FlavourRotation:
    r""":math:`R(\theta)` applied to a state supported on one exciton sector :math:`k`, evaluated
    exactly on arbitrary configurations without touching the full :math:`L_z` sector.

    :math:`R` only reassigns the flavour of singly occupied orbitals, so

    .. math::

        (R\psi)(s) = \sum_{s'} \langle s|R|s'\rangle\,\psi(s'),

    with :math:`s'` running over the sector-:math:`k` configurations with the same orbital
    occupations as :math:`s` (at most :math:`\binom{N}{k}` of them). Each matrix element is a
    product over singly occupied orbitals of :math:`\cos\frac\theta2` (flavour kept),
    :math:`+\sin\frac\theta2` (up in :math:`s'`, down in :math:`s`) or :math:`-\sin\frac\theta2`
    (down in :math:`s'`, up in :math:`s`), times the parity of the reordering; doubly occupied
    orbitals are invariant. Checked against :math:`\exp(\theta G/2)` on the full sector at
    :math:`N=8` for :math:`k=0,1,2` (agreement :math:`10^{-15}`).

    ``sample`` draws i.i.d. configurations from the mixture
    :math:`q(s)=\sum_{s'}|\psi(s')|^2\sum_a w_a\,p_a^{n_{\rm flip}}(1-p_a)^{n_{\rm keep}}`: pick
    :math:`s'` from :math:`|\psi|^2`, then flip each singly occupied orbital with probability
    :math:`p_a`. ``amplitude`` returns :math:`(R\psi)(s)` and :math:`q(s)` exactly, so a fidelity
    can be importance-sampled with weights :math:`|R\psi|^2/q` bounded by the number of terms.

    :param nm: number of orbitals (= particles).
    :param spins: ``(nk, 2*nm)`` the sector-:math:`k` configurations, quantax format.
    :param psi: amplitudes of the state on ``spins`` (any normalisation).
    :param theta: rotation angle.
    """

    def __init__(self, nm: int, spins, psi, theta: float):
        self.nm = nm
        self.c, self.s = np.cos(theta / 2), np.sin(theta / 2)
        spins = np.asarray(spins)
        self.psi = np.asarray(psi, float)
        self.w = self.psi ** 2 / np.sum(self.psi ** 2)
        up, dn = spins[:, :nm] > 0, spins[:, nm:] > 0
        self.n = up.astype(int) + dn.astype(int)
        self.fl = dn & ~up                                  # flavour of singles, True = down
        self.occ = np.array([np.flatnonzero(x > 0) for x in spins])
        groups = {}
        for j, n in enumerate(self.n):
            groups.setdefault(n.tobytes(), []).append(j)
        self.groups = {key: np.array(v) for key, v in groups.items()}

    def amplitude(self, spins, flip_probs=None, mix=None):
        """``(R psi)(s)`` on ``spins``, and ``q(s)`` when ``flip_probs`` and ``mix`` are given."""
        spins = np.asarray(spins)
        nm = self.nm
        up, dn = spins[:, :nm] > 0, spins[:, nm:] > 0
        n = up.astype(int) + dn.astype(int)
        fl_s = dn & ~up
        rows, cand = [], []
        for i, ni in enumerate(n):
            g = self.groups.get(ni.tobytes())
            if g is not None:
                rows.append(np.full(len(g), i))
                cand.append(g)
        T, q = np.zeros(len(spins)), np.zeros(len(spins))
        if not rows:
            return T, q
        rows, cand = np.concatenate(rows), np.concatenate(cand)
        single = n[rows] == 1
        fs, fp = fl_s[rows], self.fl[cand]
        keep = (single & (fs == fp)).sum(1)
        flip = (single & (fs != fp)).sum(1)
        to_up = single & ~fs & fp                           # down in s', up in s: -sin
        amp = self.c ** keep * self.s ** flip * (-1.0) ** to_up.sum(1)
        occ2 = self.occ[cand]
        m = occ2 % nm
        img = np.where(np.take_along_axis(n[rows], m, 1) == 1,
                       np.take_along_axis(fl_s[rows], m, 1) * nm + m, occ2)
        inv = np.sum(np.triu(img[:, :, None] > img[:, None, :], 1), axis=(1, 2))
        T = np.bincount(rows, amp * (-1.0) ** inv * self.psi[cand], len(spins))
        if flip_probs is not None:
            dens = sum(a * p ** flip * (1 - p) ** keep for a, p in zip(mix, flip_probs))
            q = np.bincount(rows, self.w[cand] * dens, len(spins))
        return T, q

    def sample(self, nsamples: int, rng, flip_probs, mix):
        """i.i.d. draws from ``q`` (see the class docstring), ``(nsamples, 2*nm)`` int8."""
        nm = self.nm
        j = rng.choice(len(self.psi), nsamples, p=self.w)
        p = np.asarray(flip_probs)[rng.choice(len(mix), nsamples, p=mix)][:, None]
        n, fl = self.n[j], self.fl[j]
        fl = fl ^ ((n == 1) & (rng.random((nsamples, nm)) < p))
        up = (n == 2) | ((n == 1) & ~fl)
        dn = (n == 2) | ((n == 1) & fl)
        return np.where(np.concatenate([up, dn], 1), 1, -1).astype(np.int8)


def pretrain_rotated(model, rotation: FlavourRotation, theta: float, steps: int = 3000,
                     nsamples: int = 1024, lr: float = 1e-2, lr_t0: float | None = None,
                     uniform_mix: float = 0.1, seed: int = 0, log_every: int = 500,
                     verbose: bool = True):
    r"""Fit a ``DetBackflow`` to :math:`R(\theta)|\psi\rangle` by an importance-sampled fidelity.

    Every step draws a fresh i.i.d. batch from ``rotation.sample`` -- a mixture of flip
    probability :math:`\sin^2\frac\theta2` (weight ``1 - uniform_mix``) and :math:`\frac12`, the
    latter so that the batch also sees configurations the rotated state weights little -- and
    estimates :math:`1-|\langle\phi|R\psi\rangle|^2/\langle\phi|\phi\rangle\langle R\psi|R\psi\rangle`
    with weights :math:`1/q`. Targets and densities are exact (``FlavourRotation.amplitude``), so
    the only noise is the finite batch. The fit is unsymmetrised; a :math:`Z_2`-symmetrised state
    built on the result projects it. Adam on the sampled loss, :math:`\mathrm{lr}/(1+t/t_0)`,
    keeping the parameters with the lowest running-mean loss.

    At :math:`N=8` (:math:`d=8`) this reached fidelity :math:`0.9999` to the exact rotation in
    3000 steps. At :math:`N=12` it is slower: :math:`1-F\approx3\cdot10^{-3}` at :math:`d=16`
    after 3000 steps, through a plateau near :math:`1.5\cdot10^{-2}` that breaks as the rate decays.

    :param model: the starting ``DetBackflow``, e.g. the :math:`h=0` member with
        ``rotation_orbitals @ U0`` and its up-feature gap lowered.
    :return: ``(model, history)``, ``history`` the sampled loss per step.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    lr_t0 = steps / 4 if lr_t0 is None else lr_t0
    flips, mix = [np.sin(theta / 2) ** 2, 0.5], [1 - uniform_mix, uniform_mix]
    rng = np.random.default_rng(seed)
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def loss(p, s, T, wt):
        out = jax.vmap(eqx.combine(p, static))(s)
        phi = out.sign * jnp.exp(out.logabs - jax.lax.stop_gradient(jnp.max(out.logabs)))
        return 1 - jnp.mean(phi * T * wt) ** 2 / (jnp.mean(phi ** 2 * wt) * jnp.mean(T ** 2 * wt))

    @eqx.filter_jit
    def step(p, m1, m2, it, s, T, wt):
        val, g = jax.value_and_grad(loss)(p, s, T, wt)
        a = lr / (1 + it / lr_t0)
        m1 = jax.tree_util.tree_map(lambda x, y: 0.9 * x + 0.1 * y, m1, g)
        m2 = jax.tree_util.tree_map(lambda x, y: 0.999 * x + 0.001 * y * y, m2, g)
        # cast back: the float64 step counter would otherwise promote float32 network weights, and
        # the saved model would no longer load into a freshly built (float32) skeleton
        p = jax.tree_util.tree_map(
            lambda q, x, y: (q - a * (x / (1 - 0.9 ** it)) / (jnp.sqrt(y / (1 - 0.999 ** it)) + 1e-12)).astype(q.dtype),
            p, m1, m2)
        return p, m1, m2, val

    m1 = jax.tree_util.tree_map(jnp.zeros_like, params)
    m2 = jax.tree_util.tree_map(jnp.zeros_like, params)
    history, best, window = [], (np.inf, params), 50
    for it in range(1, steps + 1):
        s = rotation.sample(nsamples, rng, flips, mix)
        T, q = rotation.amplitude(s, flips, mix)
        T = T / np.sqrt(np.mean(T ** 2 / q))
        # the step counter must be an array: a Python scalar is static under filter_jit and
        # recompiles every step
        new, m1, m2, val = step(params, m1, m2, jnp.asarray(it, dtype=jnp.float64),
                                jnp.asarray(s), jnp.asarray(T), jnp.asarray(1 / q))
        history.append(float(val))
        if it >= window and np.mean(history[-window:]) < best[0]:
            best = (float(np.mean(history[-window:])), params)
        params = new
        if verbose and (it % log_every == 0 or it == steps):
            print(f"    step {it:5d}  sampled 1-F (mean of last {window}) "
                  f"{np.mean(history[-window:]):.2e}", flush=True)
    return eqx.combine(best[1], static), history
