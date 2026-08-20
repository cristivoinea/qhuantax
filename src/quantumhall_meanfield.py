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


# A (0,0) mode is a *rigid rotation* of every flavour spinor: with M = lam*I the chart gives every
# column the spinor v_minus + lam*v_plus = sqrt(1+lam^2) * (cos(theta'/2), sin(theta'/2)) with
# theta' = theta + 2*arctan(lam), since (v_minus, v_plus) is an orthonormal frame. Such a reference
# is therefore just the canted vacuum at another angle, and lam is only the stereographic
# coordinate on that circle -- which is why these slots are optimised in the angle instead.
ANGLE_BOUND = np.pi - 1e-3
r"""Search bound on an angle slot. ``lam = tan(delta/2)`` is a bijection from
:math:`(-\pi,\pi)` onto :math:`\mathbb{R}`, so this excludes nothing the unbounded chart
coordinate reached -- it only removes the coordinate's divergence."""

ANGLE_SCAN = np.pi / 2
r"""Grid half-width, used only when every reference is a bare canted vacuum (``Mbase = 0``), where
it is complete: :math:`Z_2` is the flavour flip, so it identifies :math:`\theta` with
:math:`\pi-\theta`, the distinct states are :math:`\theta\in[0,\pi/2]`, and
:math:`[\theta_0-\pi/2,\theta_0+\pi/2]` contains that for any :math:`\theta_0\in(0,\pi/2)`. On a
stretched base the same slot is not a canting angle, so no such argument applies and the grid is
skipped."""

ANGLE_GRID = 9        # coarse scan per angle; only picks the basin, Nelder-Mead then refines to tol
EXCITON_SEED = 0.9
EXCITON_MAX = 10.0


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


def _parse_generator(tok: str, spec: str):
    r"""``'102'`` -> ``(1, 0, 2)``: one digit each to :math:`\ell`, :math:`\mu`, :math:`k_\beta`."""
    tok = tok.strip()
    if not tok.isdigit() or not 1 <= len(tok) <= 3:
        raise ValueError(
            f"generator {tok!r} in mode spec {spec!r} must be 1-3 digits, one each for ell, mu "
            "and kbeta (mu and kbeta default to 0 and 1)"
        )
    ell, mu, kb = int(tok[0]), int(tok[1]) if len(tok) > 1 else 0, int(tok[2]) if len(tok) > 2 else 1
    if kb < 1:
        raise ValueError(f"generator {tok!r} needs kbeta >= 1")
    if mu > ell:
        raise ValueError(f"generator {tok!r} has mu > ell, which is not a multipole component")
    return ell, mu, kb


def parse_modes(spec: str):
    r"""Mode spec -> one list of ``(ell, mu, kbeta)`` generators **per reference**.

    One comma-separated entry is one reference, hence one state; ``*`` joins generators into a
    single reference, whose one determinant is built from the sum of their tensors. A generator is
    bare digits filling :math:`\ell`, :math:`\mu`, :math:`k_\beta` in that order::

        ""         -> []                                   every reference implicit
        "0,0"      -> [[(0,0,1)], [(0,0,1)]]                two N00 references
        "0,0,102"  -> [[(0,0,1)], [(0,0,1)], [(1,0,2)]]     plus an L-projected N10 at kbeta=2
        "11,11"    -> [[(1,1,1)], [(1,1,1)]]                two N11 references
        "11,22"    -> [[(1,1,1)], [(2,2,1)]]                one per stretched partition of 2
        "11*22"    -> [[(1,1,1), (2,2,1)]]                  one reference, two generators
    """
    spec = spec.strip()
    if not spec:
        return []
    out = []
    for group in spec.split(","):
        group = group.strip()
        if not group:
            raise ValueError(f"empty reference in mode spec {spec!r}")
        out.append([_parse_generator(t, spec) for t in group.split("*")])
    return out


def _degrees_reachable(mus, L: int) -> bool:
    r"""Is :math:`\sum_i a_i\mu_i=L` solvable in integers :math:`a_i\ge1`?

    The exponents are not ours to choose: the :math:`L_z` projection keeps exactly the monomials of
    total shift ``L``, so a generator set is usable in that sector iff at least one such monomial
    exists. With one generator this is just :math:`\mu\mid L`.
    """
    base = sum(mus)
    if L < base:
        return False
    rem = L - base
    ok = [False] * (rem + 1)
    ok[0] = True
    for r in range(1, rem + 1):
        ok[r] = any(r >= m and ok[r - m] for m in mus)
    return ok[rem]


def mode_references(vac: HFVacuum, spec: str, L: int, nstates: int | None = None):
    r"""Reference descriptors of a mode spec, in Rayleigh-Ritz basis order.
    :return:
        ``(refs, ndets, nfree)``: one dict per basis vector, the determinants the stack will hold,
        and the free parameters to optimise. Each ref carries its ``Mbase``, and ``mmax``, the
        largest :math:`|\mu|` its own determinants contain, which is all that sets its gauge count
        -- a plain reference is the unrotated base, while a beta rotation spreads every generator
        over all :math:`|\mu|\le\ell`.
    """
    nm = vac.nm
    groups = parse_modes(spec)
    if nstates is not None:
        if nstates < 1:
            raise ValueError(f"nstates must be >= 1, got {nstates}")
        if len(groups) > nstates:
            raise ValueError(
                f"mode spec {spec!r} gives {len(groups)} references but only {nstates} states were "
                "asked for; drop an entry or raise nstates"
            )
        groups = [[(0, 0, 1)]] * (nstates - len(groups)) + groups
    if not groups:
        raise ValueError("mode spec produced no references; give entries or pass nstates")

    refs, ndets = [], 0
    for gens in groups:
        tag = "*".join(f"{e}{m}{k}" for e, m, k in gens)
        stretch = [(e, m) for e, m, _ in gens if m > 0]
        proj = [(e, k) for e, m, k in gens if m == 0 and e > 0]
        zeros = [g for g in gens if g[0] == 0]
        if zeros and len(gens) > 1:
            raise ValueError(
                f"reference {tag!r}: a 0 generator inside a * group is redundant, every reference "
                "already carries an N00 angle"
            )
        if len(proj) > 1:
            raise ValueError(f"reference {tag!r} has two mu=0 projected generators; keep one")
        if any(k != 1 for _, m, k in gens if m > 0):
            raise ValueError(
                f"reference {tag!r}: a stretched (mu > 0) generator needs no kbeta -- L_z alone "
                "makes it exact-L"
            )
        if stretch and not _degrees_reachable([m for _, m in stretch], L):
            raise ValueError(
                f"reference {tag!r} carries mu = {[m for _, m in stretch]}, which cannot sum to "
                f"L={L} with every exponent >= 1"
            )
        # `parts[0]` is the base; a `mix` amplitude scales each later generator against it. The
        # overall tensor scale is inert (the L_z projection makes it a pure factor) but the ratio
        # between two generators is not, so only the extras carry a parameter.
        parts = [multipole_tensor(nm, e, m) for e, m in stretch]
        Mbase = parts[0] if parts else np.zeros((nm, nm))
        mix = ("mix",) * max(len(parts) - 1, 0)
        if proj:
            ell, kb = proj[0]
            refs.append(dict(kind="proj", ell=ell, kbeta=kb, Mbase=Mbase, parts=parts, tag=tag,
                             slots=mix + ("exciton", "angle"),
                             mmax=max([ell] + [e for e, _ in stretch])))
            ndets += kb
        else:
            refs.append(dict(kind="plain", ell=0, kbeta=1, Mbase=Mbase, parts=parts, tag=tag,
                             slots=mix + ("angle",),
                             mmax=max((m for _, m in stretch), default=0)))
            ndets += 1
    if len(refs) == 1 and refs[0]["kind"] == "plain" and not refs[0]["Mbase"].any():
        # One *bare* canted vacuum, and `hf_vacuum` already returns the angle that extremises its
        # (projected) energy in closed form, so there is nothing left to search. This does not
        # extend to a lone stretched reference, whose best angle is not the bare vacuum's.
        refs[0]["slots"] = ()
    for r in refs:
        r["free"] = len(r["slots"])
    return refs, ndets, sum(r["free"] for r in refs)


def mode_stack(vac: HFVacuum, refs, params, L: int):
    r"""Determinant stack and grouping matrix of a mode spec.

    :return:
        ``(U0, W)`` of shapes ``(ndets, 2*nm, nm)`` and ``(ndets, nrefs)``, with ``W`` holding the
        fixed quadrature weights so that reference ``r`` is ``sum_d W[d, r] |Phi(U0[d])>``. A plain
        reference occupies one row of ``W``, a projected one its ``kbeta`` rows.

    Each reference is built on its own ``Mbase``. ``"angle"`` slots hold a canting-angle offset in
    radians, converted here to the chart coordinate by ``lam = tan(delta/2)``; ``"exciton"`` slots
    hold the multipole amplitude directly, and ``"mix"`` slots rescale the second and later
    stretched generators relative to the first -- the overall scale is inert under the :math:`L_z`
    projection, the ratio is not.
    """
    nm = vac.nm
    Us, cols, p = [], [], 0
    for r in refs:
        M = r["Mbase"]
        for j in range(r["slots"].count("mix")):
            M = M + params[p] * r["parts"][j + 1]
            p += 1
        if r["kind"] == "plain":
            lam = 0.0
            if "angle" in r["slots"]:
                lam, p = np.tan(params[p] / 2), p + 1
            Us.append(chart(vac, M + lam * np.eye(nm)))
            cols.append(([len(Us) - 1], [1.0]))
        else:
            c, lam = params[p], np.tan(params[p + 1] / 2)
            p += 2
            T = multipole_tensor(nm, r["ell"], 0)
            idx, wts = [], []
            for beta, w in zip(*beta_quadrature(r["kbeta"], L)):
                D = wigner_dy(nm, beta)
                Us.append(chart(vac, D @ (M + c * T) @ D.T + lam * np.eye(nm)))
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

    The free parameters are tuned on the state-averaged trace of the lowest ``nstates`` roots;
    minimising the leading root alone would trade the excited references away. Rayleigh-Ritz is
    solved in the *reference* basis, with the quadrature weights held fixed, which is what keeps
    each root an (almost) definite-:math:`L` state. ``cond(S)`` is returned because it, not physics,
    limits how many references can be stacked, and parameters exceeding ``cond_max`` are rejected --
    beyond it the objective is round-off, which a variational search will happily exploit.

    Every :math:`\mathcal{N}_{00}` reference is a canted vacuum at its own angle (see ``ANGLE_MAX``),
    so an all-``(0,0)`` spec is a set of angles on a bounded interval: those are scanned on a coarse
    grid and the best basins refined, rather than searched in the unbounded chart coordinate. Note
    ``cond(S)`` is *not* scale-invariant -- it includes the ratio of the references' norms, so two
    parameter sets describing the same states through different representatives can report very
    different values. For sensitivity, use :math:`\|c\|/\sqrt{c^\dagger Sc}` on the returned vectors.

    :return:
        A dict with ``U0`` ``(ndets, 2*nm, nm)``, ``coeffs`` ``(nstates, ndets)`` -- one
        determinant-level vector per state -- plus ``weights`` ``(ndets, nrefs)``, the fixed
        quadrature weights, whose column ``r`` is reference ``r`` on its own determinants and zero
        elsewhere. Both bases span the same space and give the same roots: ``coeffs`` makes the
        states mutually orthogonal, ``weights`` keeps each state on its own reference's
        determinants. Also ``energies``, ``residual``
        (:math:`|\langle L^2\rangle-L(L{+}1)|` per state, if ``l2_terms`` is given), ``cond``,
        ``cond_rel``, ``params``, ``slots``, ``ndets``, ``nrefs`` and ``kphi``.
    """
    from itertools import combinations

    from scipy.optimize import minimize

    refs, ndets, nfree = mode_references(vac, spec, L, nstates=nstates)
    nrefs = len(refs)
    kphi = mode_kphi_list(vac, refs, lz)

    def reduced(params, terms):
        U0, W = mode_stack(vac, refs, params, L)
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

    slot_kinds = [s for r in refs for s in r["slots"]]
    bounds = [(-ANGLE_BOUND, ANGLE_BOUND) if s == "angle" else (-EXCITON_MAX, EXCITON_MAX)
              for s in slot_kinds]

    best = (np.inf, np.zeros(nfree))
    if nfree:
        starts = []
        pure_angles = (all(s == "angle" for s in slot_kinds)
                       and not any(r["Mbase"].any() for r in refs))
        if pure_angles and nfree <= 3:
            # Every reference is a canted vacuum, so this is a set of angles on an interval that
            # ANGLE_SCAN covers completely -- scan it. Only strictly increasing tuples: the plain
            # references are an unordered set, so permutations are the same reference, and increasing
            # tuples also skip the coincident angles where S is singular.
            grid = np.linspace(-ANGLE_SCAN, ANGLE_SCAN, ANGLE_GRID)
            scored = sorted((objective(np.array(c)), c)
                            for c in combinations(grid, nfree))
            starts = [np.array(c) for f, c in scored[:2] if np.isfinite(f)]
        if not starts:
            # Fallback for specs carrying exciton amplitudes, where a grid is not affordable. Seed
            # each slot by what it is and spread it *within its own kind*: two references of the same
            # kind differ only by these parameters, so seeds that are close start near a
            # rank-deficient point and the search stays there. The projected references keep the
            # weights that worked before the reparameterisation, written as the equivalent angles.
            nplain = sum(len(r["slots"]) for r in refs if r["kind"] == "plain")
            plain = np.linspace(-0.35, 0.25, max(nplain, 1))
            seed0, ka, kj = [], 0, 0
            for r in refs:
                for s in r["slots"]:
                    if r["kind"] == "plain":
                        seed0.append(plain[ka]); ka += 1
                    elif s == "exciton":
                        seed0.append(EXCITON_SEED * (1.0 + 0.5 * kj))
                    else:
                        seed0.append(2 * np.arctan(0.3 * (1.0 + 0.8 * kj))); kj += 1
            starts = [np.array(seed0), -np.array(seed0)]

        for x0 in starts:
            r = minimize(objective, x0, method="Nelder-Mead", bounds=bounds,
                         options=dict(xatol=tol, fatol=tol))
            if r.fun < best[0]:
                best = (float(r.fun), np.asarray(r.x, dtype=float))
        if not np.isfinite(best[0]):
            raise RuntimeError("no usable parameters found: every start hit the cond guard")

    params = best[1]
    energies, C, cond, U0, W = roots(params)
    coeffs = np.array([(W @ C[:, k]).real for k in range(nstates)])

    # Scale-invariant companion to `cond`: the guard above needs the raw spectrum of S (it is a
    # numerical-rank test), but that number also contains the ratio of the references' norms, so it
    # says little about how ill-conditioned the *states* are. Normalising the diagonal away does.
    _, S_ref, _, _ = reduced(params, hamiltonian)
    dS = np.sqrt(np.abs(np.diag(S_ref)))
    cond_rel = float(np.linalg.cond(S_ref / np.outer(dS, dS)))

    residual = None
    if l2_terms is not None:
        A2, S2, _, _ = reduced(params, l2_terms)
        residual = [
            abs(float(((C[:, k].conj() @ A2 @ C[:, k]) / (C[:, k].conj() @ S2 @ C[:, k])).real)
                - L * (L + 1))
            for k in range(nstates)
        ]

    return dict(U0=U0, coeffs=coeffs, weights=W.real, energies=energies[:nstates], cond=cond,
                cond_rel=cond_rel, params=[float(v) for v in params], slots=list(slot_kinds),
                ndets=ndets, nrefs=nrefs, kphi=kphi, residual=residual)


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
