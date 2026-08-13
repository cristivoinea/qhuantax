r"""Hartree-Fock reference states (``U0``) for the :math:`\nu=1` fuzzy sphere.

Builds mean-field orbitals with exact :math:`L_z` and :math:`Z_2` and (near-)exact :math:`L`, to
serve as the ``U0`` of a ``MultiDetBackflow`` ansatz. Derivation and numerical validation live in
``docs/exciton_references.pdf`` and ``docs/mf_vs_ed.pdf``; the essentials are:

* The HF vacuum puts one flavour spinor in every orbital, so it is manifestly :math:`L=L_z=0` and
  its canting angle is closed-form -- no optimisation loop.
* Excitations use the pure creator :math:`\mathcal{N}_{\ell\mu}=|+\rangle\langle-|\otimes t^{\ell\mu}`,
  which moves one particle to the empty band and shifts :math:`L_z` by exactly :math:`\mu`. Because
  :math:`\mathcal{N}_i\mathcal{N}_j=0` as matrices,
  :math:`\exp(\sum\lambda_i\mathcal{N}_i)=I+\sum\lambda_i\mathcal{N}_i` exactly.
* One determinant is a *generating function*: restricted to the :math:`(L_z, Z_2)` sector the
  sampler enforces, its amplitudes are those of a definite-:math:`L` exciton state. On a canted
  vacuum both :math:`Z_2` branches are non-zero, so the same ``U0`` serves either sector.

``solve_modes`` is the entry point, driven by ``scripts/FuzzySphereMeanField.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy.physics.wigner as spw
from sympy import Rational

from qhuantax.quantumhall_operators import get_int_matrix


DEFAULT_WEIGHT = 0.3


@dataclass(frozen=True)
class HFVacuum:
    r"""Uniform canted HF vacuum: everything follows from the canting angle :math:`\theta`."""

    nm: int
    theta: float

    @property
    def v_minus(self) -> np.ndarray:
        """Filled-band flavour spinor."""
        return np.array([np.cos(self.theta / 2), np.sin(self.theta / 2)])

    @property
    def v_plus(self) -> np.ndarray:
        """Empty-band flavour spinor, orthogonal to ``v_minus``."""
        return np.array([-np.sin(self.theta / 2), np.cos(self.theta / 2)])


def projected_hf_energy(x, N: int, A: float, h: float, z2: int):
    r"""HF energy of the :math:`Z_2`-projected vacuum."""
    return (A * x**2 - h * N * x) * (1 + z2 * x ** (N - 2)) / (1 + z2 * x**N)


def hf_vacuum(nm: int, ps_pot, h: float, z2: int | None = None) -> HFVacuum:
    r"""The :math:`\nu=1` canted ferromagnet state, in closed form.

    :param h:
        The transverse field strength.

    :param z2:
        If given, the `Z_2`-symmetry sector onto which the state is projected.
    """
    U = get_int_matrix(nm, ps_pot)
    m1, m2 = np.meshgrid(np.arange(nm), np.arange(nm), indexing="ij")
    A = 0.25 * (U[m1, m2, m2].sum() - U[m1, m2, m1].sum())

    x = np.clip(abs(h) * nm / (2 * A), 0.0, 1.0)
    if z2 is not None:
        # The stationarity condition is a polynomial of degree 2N-2, so bracket it numerically:
        # nested grids on (0, 1), which converge to ~1e-12 in x for a few thousand evaluations.
        lo, hi = 1e-9, 1.0 - 1e-9
        for _ in range(5):
            xs = np.linspace(lo, hi, 2001)
            k = int(np.argmin(projected_hf_energy(xs, nm, A, abs(h), z2)))
            lo, hi = xs[max(k - 1, 0)], xs[min(k + 1, xs.size - 1)]
        x = 0.5 * (lo + hi)
    return HFVacuum(nm, float(np.copysign(np.arcsin(x), h)))


def multipole_tensor(nm: int, ell: int, mu: int) -> np.ndarray:
    r"""Orbital part :math:`t^{\ell\mu}_{m_1m_2}` of the rank-:math:`\ell` multipole.

    The Wigner-3j selection rule makes it non-zero only on :math:`m_1=m_2+\mu`, which is why the
    multipole shifts :math:`L_z` by exactly :math:`\mu`. Note :math:`t^{00}=I/\sqrt{nm}`.
    """
    s = Rational(nm - 1, 2)
    pref = float(np.sqrt(2 * ell + 1))
    t = np.zeros((nm, nm))
    for m2 in range(nm):
        m1 = m2 + mu
        if 0 <= m1 < nm:
            sign = (-1) ** int(s - (m1 - s))
            t[m1, m2] = pref * sign * float(spw.wigner_3j(s, ell, s, -(m1 - s), mu, m2 - s))
    return t


def chart(vac: HFVacuum, M) -> np.ndarray:
    r"""Orbitals of :math:`\exp(\hat{\mathcal{N}}[M])|\Phi_{HF}\rangle`, shape ``(2*nm, nm)``.

    :math:`U=|-\rangle\otimes I+|+\rangle\otimes M` on modes ``(f, m) -> f*nm + m``: every
    reference in this module is one point of this chart, and the mode spec only says which ``M``.
    """
    vm, vp = vac.v_minus, vac.v_plus
    eye = np.eye(vac.nm)
    return np.vstack([vm[0] * eye + vp[0] * M, vm[1] * eye + vp[1] * M])


def wigner_dy(nm: int, beta: float) -> np.ndarray:
    r"""The spin-:math:`s` rotation :math:`D(R_y(\beta))` on the orbitals, :math:`s=(nm-1)/2`.

    Real, because :math:`\ell_y` is purely imaginary in this basis -- which is what keeps ``U0``
    real, as ``DetBackflow`` wants.
    """
    from scipy.linalg import expm

    s = (nm - 1) / 2
    m = np.arange(nm) - s
    lp = np.zeros((nm, nm))
    for i in range(nm - 1):
        lp[i + 1, i] = np.sqrt(s * (s + 1) - m[i] * (m[i] + 1))
    D = expm(-1j * beta * (lp - lp.T) / (2j))
    assert np.abs(D.imag).max() < 1e-10, "Wigner d-matrix must be real"
    return D.real


def beta_quadrature(kbeta: int, L: int):
    r"""Nodes and weights of :math:`\tfrac12\int\sin\beta\,d\beta\;d^L_{LL}(\beta)\,R_y(\beta)`.

    Gauss-Legendre in :math:`\cos\beta` with :math:`d^L_{LL}=\cos^{2L}(\beta/2)`. After the free
    outer :math:`L_z` projection the integrand is exactly :math:`\sum_L c_L P_L(\cos\beta)`, so
    ``kbeta`` nodes annihilate :math:`P_1\dots P_{2k_\beta-1}` and the residual is the first
    surviving coefficient. Those fall about two orders of magnitude per two units of :math:`L`, so
    ``kbeta = 2`` already leaves :math:`|\langle L^2\rangle-L(L{+}1)|\sim0.05`, independently of
    :math:`N`.
    """
    x, w = np.polynomial.legendre.leggauss(kbeta)
    betas = np.arccos(x)
    return betas, 0.5 * w * np.cos(betas / 2) ** (2 * L)


def parse_modes(spec: str):
    """``'(0,0),(0,0),(1,0,2)'`` -> ``[(0, 0, 1), (0, 0, 1), (1, 0, 2)]``, ``kbeta`` defaulting to 1."""
    import re

    groups = re.findall(r"\(([^()]*)\)", spec)
    if not groups:
        raise ValueError(f"no (ell,mu[,kbeta]) entries found in mode spec {spec!r}")
    entries = []
    for g in groups:
        try:
            parts = [int(v) for v in g.replace(",", " ").split()]
        except ValueError:
            raise ValueError(f"entry ({g}) in mode spec {spec!r} is not integer") from None
        if len(parts) == 2:
            parts = parts + [1]
        if len(parts) != 3:
            raise ValueError(f"entry ({g}) must be (ell,mu) or (ell,mu,kbeta)")
        if parts[2] < 1:
            raise ValueError(f"entry ({g}) needs kbeta >= 1")
        entries.append(tuple(parts))
    return entries


def mode_references(vac: HFVacuum, spec: str, L: int):
    r"""Reference descriptors of a mode spec, in Rayleigh-Ritz basis order.

    :return:
        ``(refs, Mbase, ndets, nfree)``: one dict per basis vector, the stretched base matrix, the
        determinants the stack will hold, and the free parameters to optimise. Each ref carries
        ``mmax``, the largest :math:`|\mu|` its own determinants contain, which is all that sets its
        gauge count -- a plain reference is the unrotated base, while a beta rotation spreads every
        generator over all :math:`|\mu|\le\ell`.
    """
    nm = vac.nm
    entries = parse_modes(spec)

    stretched, i = [], 0
    while i < len(entries) and entries[i][1] != 0:
        ell, mu, kb = entries[i]
        if kb != 1:
            raise ValueError(f"stretched entry ({ell},{mu},{kb}) cannot carry a projection count")
        stretched.append((ell, mu))
        i += 1
    if any(mu != 0 for _, mu, _ in entries[i:]):
        raise ValueError("stretched (mu != 0) entries must come first in the mode spec")
    if sum(mu for _, mu in stretched) != L:
        raise ValueError(
            f"stretched entries {stretched} carry Lz={sum(mu for _, mu in stretched)}, "
            f"but the requested sector is L={L}; at L=0 give no mu!=0 entries"
        )
    Mbase = sum((multipole_tensor(nm, ell, mu) for ell, mu in stretched), np.zeros((nm, nm)))
    mumax = max((abs(mu) for _, mu in stretched), default=0)
    lmax = max((ell for ell, _ in stretched), default=0)

    refs, ndets, nfree = [], 0, 0
    if stretched:
        refs.append(dict(kind="plain", ell=0, kbeta=1, free=0, mmax=mumax))
        ndets += 1
    for ell, _, kb in entries[i:]:
        if ell == 0:
            if kb != 1:
                raise ValueError(f"(0,0,{kb}) makes no sense: N00 is a scalar, already exact-L")
            first = not refs
            refs.append(dict(kind="plain", ell=0, kbeta=1, free=0 if first else 1, mmax=mumax))
            nfree += 0 if first else 1
            ndets += 1
        else:
            refs.append(dict(kind="proj", ell=ell, kbeta=kb, free=2, mmax=max(lmax, ell)))
            nfree += 2
            ndets += kb
    if not refs:
        raise ValueError("mode spec produced no references")
    return refs, Mbase, ndets, nfree


def mode_stack(vac: HFVacuum, refs, Mbase, params, L: int):
    r"""Determinant stack and grouping matrix of a mode spec.

    :return:
        ``(U0, W)`` of shapes ``(ndets, 2*nm, nm)`` and ``(ndets, nrefs)``, with ``W`` holding the
        fixed quadrature weights so that reference ``r`` is ``sum_d W[d, r] |Phi(U0[d])>``. A plain
        reference occupies one row of ``W``, a projected one its ``kbeta`` rows.
    """
    nm = vac.nm
    Us, cols, p = [], [], 0
    for r in refs:
        if r["kind"] == "plain":
            lam = 0.0
            if r["free"]:
                lam, p = params[p], p + 1
            Us.append(chart(vac, Mbase + lam * np.eye(nm)))
            cols.append(([len(Us) - 1], [1.0]))
        else:
            c, lam = params[p], params[p + 1]
            p += 2
            T = multipole_tensor(nm, r["ell"], 0)
            idx, wts = [], []
            for beta, w in zip(*beta_quadrature(r["kbeta"], L)):
                D = wigner_dy(nm, beta)
                Us.append(chart(vac, D @ (Mbase + c * T) @ D.T + lam * np.eye(nm)))
                idx.append(len(Us) - 1)
                wts.append(w)
            cols.append((idx, wts))
    W = np.zeros((len(Us), len(refs)), dtype=complex)
    for r, (idx, wts) in enumerate(cols):
        W[idx, r] = wts
    return np.stack(Us), W


def mode_kphi_list(vac: HFVacuum, refs, lz: int):
    r"""Gauge angles **per determinant**, in stack order.

    The Fourier projector is exact once ``K`` exceeds :math:`|L_z-l_z|` over the :math:`L_z` values
    that determinant actually holds -- a per-determinant property. A uniform-spinor determinant
    carries :math:`L_z=0` alone, so ``K=1`` already projects it exactly, whereas a beta-rotated one
    spreads over :math:`|L_z|\le nm\,\ell`. The :math:`|l_z|` floor matters because ``K=1`` keeps
    *everything*, which is only right when that single :math:`L_z` equals ``lz``.
    """
    ks = []
    for r in refs:
        ks += [max(vac.nm * r["mmax"], abs(lz)) + 1] * r["kbeta"]
    return ks


# ------------------------------------------------------------------ projected energy


def projected_dets(vac: HFVacuum, U0, coeffs, lz: int, z2: int, kphi: int):
    r"""Expand :math:`P_{\rm sector}|\Psi\rangle` as an explicit determinant list.

    ``P_sector`` applied to a determinant is a finite sum of determinants,
    which is what makes the projected energy exactly computable. Both projectors act one-body:
    :math:`Z_2` is the flavour swap, a mode permutation; and by Thouless
    :math:`e^{i\phi\hat L_z}|\Phi(U)\rangle=|\Phi(e^{i\phi\Lambda}U)\rangle` with
    :math:`\Lambda={\rm diag}(m)`, hence the offset :math:`\mu'=nm(nm-1)/2+l_z` at which
    :math:`U_{HF}` sits. Orbitals come back orthonormal with :math:`\det R` folded into the
    coefficients, ready for ``MultiDet``; returns ``2*kphi*len(U0)`` determinants.
    """
    nm = vac.nm
    orb = np.tile(np.arange(nm), 2)  # Lambda on modes (f, m) -> m
    mu_p = nm * (nm - 1) // 2 + lz
    swap = np.roll(np.eye(2 * nm), nm, axis=0)  # (f, m) -> (1 - f, m)

    dets, cs = [], []
    for k in range(kphi):
        phi = 2 * np.pi * k / kphi
        gauge = np.exp(1j * phi * orb)[:, None]
        weight = np.exp(-1j * phi * mu_p) / kphi
        for U, c in zip(U0, coeffs):
            rotated = gauge * U
            for V, sign in ((rotated, 0.5), (swap @ rotated, 0.5 * z2)):
                Q, R = np.linalg.qr(V)
                dets.append(Q)
                cs.append(weight * sign * c * np.linalg.det(R))
    return np.array(dets), np.array(cs)


def _overlap(Da, ca, Db, cb) -> complex:
    r""":math:`\langle\Psi_a|\Psi_b\rangle` for two multi-determinant states."""
    gram = np.einsum("aim,bin->abmn", np.conj(Da), Db)
    return complex(np.einsum("a,b,ab->", np.conj(ca), cb, np.linalg.det(gram)))


def _ratio(dets, coeffs, hamiltonian) -> complex:
    r"""quantax's exact :math:`\langle\Psi|H|\Psi\rangle/\langle\Psi|\Psi\rangle` by Wick.

    Needs ``quantax.set_default_dtype(jnp.complex128)`` whenever ``kphi > 1``, since the gauge
    rotation makes the orbitals complex.
    """
    import jax.numpy as jnp
    import quantax as qtx

    model = qtx.model.MultiDet(
        ndets=len(dets), U=jnp.asarray(dets), coeffs=jnp.asarray(coeffs)
    )
    return complex(qtx.state.MultiDetState(model).mf_expectation(hamiltonian))


def reduced_matrices(vac: HFVacuum, U0, lz: int, z2: int, kphi, hamiltonian):
    r"""Reduced :math:`(H, S)` of the projected references, for the Rayleigh-Ritz problem.

    ``S`` is a determinant overlap and needs no operator. For ``H`` we reuse quantax's exact
    multi-determinant expectation, which returns the *ratio*
    :math:`c^\dagger Hc/c^\dagger Sc`; since ``S`` is known, the choices :math:`c=e_i` and
    :math:`e_i+e_j` give every entry, so no Wick code is duplicated here. Both matrices are real --
    the orbitals and the Hamiltonian are, and the projectors are real in the occupation basis
    despite their complex gauge factors -- so one contraction per pair suffices.

    Cost is set by ``kphi``: determinant ``i`` expands into ``2*kphi[i]`` determinants, so an element
    costs :math:`4(k_i+k_j)^2` Wick pairs. Pass one count per determinant (see ``mode_kphi_list``);
    a scalar applies to all of them.
    """
    ks = [int(kphi)] * len(U0) if np.ndim(kphi) == 0 else [int(k) for k in kphi]
    if len(ks) != len(U0):
        raise ValueError(f"kphi has {len(ks)} entries for {len(U0)} determinants")
    blocks = [projected_dets(vac, U[None], np.ones(1), lz, z2, k) for U, k in zip(U0, ks)]
    n = len(blocks)
    S = np.array([[_overlap(*blocks[i], *blocks[j]) for j in range(n)] for i in range(n)])

    def hc(idx, weights):
        d = np.concatenate([blocks[i][0] for i in idx])
        c = np.concatenate([w * blocks[i][1] for i, w in zip(idx, weights)])
        return _ratio(d, c, hamiltonian) * _overlap(d, c, d, c)

    H = np.zeros((n, n))
    for i in range(n):
        H[i, i] = hc([i], [1.0]).real
    for i in range(n):
        for j in range(i + 1, n):
            H[i, j] = H[j, i] = 0.5 * (hc([i, j], [1.0, 1.0]).real - H[i, i] - H[j, j])
    return H, S.real


# ------------------------------------------------------------------ driver interface


def solve_modes(
    vac: HFVacuum,
    spec: str,
    L: int,
    lz: int,
    z2: int,
    hamiltonian,
    nstates: int = 1,
    l2_terms=None,
    tol: float = 1e-2,
    cond_max: float = 1e8,
):
    r"""Optimise a mode spec and diagonalise it in the :math:`(L_z, z_2)` sector.

    The free weights are tuned on the state-averaged trace of the lowest ``nstates`` roots;
    minimising the leading root alone would trade the excited references away. Rayleigh-Ritz is
    solved in the *reference* basis, with the quadrature weights held fixed, which is what keeps
    each root an (almost) definite-:math:`L` state. ``cond(S)`` is returned because it, not physics,
    limits how many references can be stacked, and parameters exceeding ``cond_max`` are rejected --
    beyond it the objective is round-off, which a variational search will happily exploit.

    :return:
        A dict with ``U0`` ``(ndets, 2*nm, nm)``, ``coeffs`` ``(nstates, ndets)`` -- one
        determinant-level vector per state -- plus ``energies``, ``residual``
        (:math:`|\langle L^2\rangle-L(L{+}1)|` per state, if ``l2_terms`` is given), ``cond``,
        ``params``, ``ndets``, ``nrefs`` and ``kphi``.
    """
    from scipy.optimize import minimize

    refs, Mbase, ndets, nfree = mode_references(vac, spec, L)
    nrefs = len(refs)
    if nstates > nrefs:
        raise ValueError(
            f"mode spec gives {nrefs} references, so at most {nrefs} states; asked for {nstates}. "
            "Add another (0,0) or (ell,0,kbeta) entry."
        )
    kphi = mode_kphi_list(vac, refs, lz)

    def reduced(params, terms):
        U0, W = mode_stack(vac, refs, Mbase, params, L)
        A, S = reduced_matrices(vac, U0, lz, z2, kphi, terms)
        return W.conj().T @ A @ W, W.conj().T @ S @ W, U0, W

    def roots(params, terms=hamiltonian):
        A, S, U0, W = reduced(params, terms)
        d, V = np.linalg.eigh(S)
        keep = d > 1e-11 * d.max()
        X = V[:, keep] / np.sqrt(d[keep])
        e, C = np.linalg.eigh(X.conj().T @ A @ X)
        return e.real, X @ C, float(d.max() / d[keep].min()), U0, W

    def objective(params):
        try:
            e, _, cond, _, _ = roots(params)
        except np.linalg.LinAlgError:
            return np.inf
        if cond > cond_max or len(e) < nstates:
            return np.inf
        return float(np.sum(e[:nstates]))

    best = (np.inf, np.zeros(nfree))
    if nfree:
        # Seed each slot by what it is, staggered within its kind: two references of the same kind
        # differ only by these weights, so equal seeds would start at a rank-deficient point.
        seed0, kp, kj = [], 0, 0
        for r in refs:
            if r["free"] == 1:
                seed0.append(-DEFAULT_WEIGHT * (1.0 + 0.8 * kp))
                kp += 1
            elif r["free"] == 2:
                seed0 += [0.9 * (1.0 + 0.5 * kj), DEFAULT_WEIGHT * (1.0 + 0.8 * kj)]
                kj += 1
        # The seed and its negation: the two signs cant the vacuum up and down from theta*, which
        # are physically distinct references, so this is genuine exploration of the other basin.
        for x0 in (np.array(seed0), -np.array(seed0)):
            r = minimize(objective, x0, method="Nelder-Mead", options=dict(xatol=tol, fatol=tol))
            if r.fun < best[0]:
                best = (float(r.fun), np.asarray(r.x, dtype=float))
        if not np.isfinite(best[0]):
            raise RuntimeError("no usable parameters found: every start hit the cond guard")

    params = best[1]
    energies, C, cond, U0, W = roots(params)
    coeffs = np.array([(W @ C[:, k]).real for k in range(nstates)])

    residual = None
    if l2_terms is not None:
        A2, S2, _, _ = reduced(params, l2_terms)
        residual = [
            abs(float(((C[:, k].conj() @ A2 @ C[:, k]) / (C[:, k].conj() @ S2 @ C[:, k])).real)
                - L * (L + 1))
            for k in range(nstates)
        ]

    return dict(U0=U0, coeffs=coeffs, energies=energies[:nstates], cond=cond,
                params=[float(v) for v in params], ndets=ndets, nrefs=nrefs,
                kphi=kphi, residual=residual)


def backflow_coeffs(U0, coeffs):
    r"""Rescale ``coeffs`` to survive ``DetBackflow``'s per-determinant normalisation.

    ``MultiDetBackflow`` builds one ``DetBackflow`` per determinant, each dividing *its own* ``U0``
    by *its own* standard deviation. That scales determinant ``i`` by :math:`\sigma_i^{-N}` and would
    otherwise destroy the relative weights, so undo it here. Only ratios matter, so the result is
    rescaled to ``max |c| = 1``.
    """
    sigma = U0.reshape(len(U0), -1).std(axis=1)
    compensated = coeffs * sigma ** U0.shape[-1]
    return compensated / np.abs(compensated).max()
