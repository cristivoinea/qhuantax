r"""Fuzzy-sphere 3D Ising ED levels (V0 = 4.75, V1 = 1, h = 3.16; FuzzySphere/3d_ising/iqh_gaps.ipynb) and the
smallest :math:`L^- L^+` penalty that makes a given set of them the lowest states of :math:`H + \lambda L^- L^+`.

``ED_benchmark/3d_ising_spectrum_ne_{N}.txt`` holds one level per row, lowest first, columns ``E L Z P``: Z the flavour
flip (``--z2-sect``) and P the FuzzifiED particle hole ``GetParityQNOffd(nm, 2, [2, 1], [-1, 1])``, so ``--ry-ph-sect``
is P (-1)^L. The ED kept the 20 lowest states of each (P, Z, Ry) sector at Lz = 0 and of these the ``N_EIG`` lowest of
each (L, Z, P) with L <= ``L_MAX``; whatever is missing lies above them.
"""

from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "ED_benchmark"
H_FIELD = 3.16
PS_POT = (4.75, 1.0)
N_EIG = 4
L_MAX = 4


def sizes() -> list[int]:
    return sorted(int(f.stem.rsplit("_", 1)[1]) for f in DATA.glob("3d_ising_spectrum_ne_*.txt"))


def load_spectrum(N: int) -> np.ndarray:
    """``(n, 4)`` rows ``E L Z P`` at size N, lowest first."""
    return np.loadtxt(DATA / f"3d_ising_spectrum_ne_{N}.txt", ndmin=2)


def levels(N: int, L: int, Z=None, P=None) -> np.ndarray:
    """Energies of the (L, Z, P) levels at size N, lowest first; None leaves that label unresolved."""
    s = load_spectrum(N)
    keep = s[:, 1] == L
    if Z is not None:
        keep &= s[:, 2] == Z
    if P is not None:
        keep &= s[:, 3] == P
    return s[keep, 0]


def gap_fit(L: int, Z, k: int, P=None):
    """Port of ``get_fit``: the gap of the k-th (L, Z, P) level above the ground state at every size, NaN where it is
    absent, and its least-squares fit ``a + b / sqrt(N)``.

    :return: ``(Ns, gaps, (a, b))``, ``(a, b)`` NaN when fewer than two sizes have the level.
    """
    Ns = np.array(sizes())
    gaps = np.full(len(Ns), np.nan)
    for i, N in enumerate(Ns):
        lv = levels(N, L, Z, P)
        if k <= len(lv):
            gaps[i] = lv[k - 1] - load_spectrum(N)[0, 0]
    ok = ~np.isnan(gaps)
    if ok.sum() < 2:
        return Ns, gaps, (np.nan, np.nan)
    A = np.stack([np.ones(ok.sum()), 1 / np.sqrt(Ns[ok])], axis=1)
    return Ns, gaps, tuple(np.linalg.lstsq(A, gaps[ok], rcond=None)[0])


def min_lambda(N: int, L: int, n_states: int, Z=None, ry_ph=None, ry=None, margin: float = 0.0,
               verbose: bool = True) -> float:
    r"""Smallest :math:`\lambda` for which the ``n_states`` lowest (L, Z) levels are the ``n_states`` lowest states
    of :math:`H + \lambda L^- L^+` at Lz = L, every other level at least ``margin`` above the last of them. A level of
    L' > L moves up by :math:`\lambda [L'(L'+1) - L(L+1)]`, so the lowest level of each L' up to ``L_MAX`` binds
    (L' > ``L_MAX`` is not in the data). With ``ry_ph`` (PH x Ry, eigenvalue P (-1)^L) the targets have
    P = ry_ph (-1)^L and only the L' levels with P = ry_ph (-1)^L' compete; ``ry`` (Lz = 0 only, eigenvalue (-1)^L)
    leaves the even L'; Z = None leaves the Z2 sector unresolved.

    At a size in the data the ED gaps are used, at any other N the ``gap_fit`` extrapolations.
    """
    if n_states > N_EIG:
        # with Z or P unresolved more levels are on file, but beyond N_EIG some may be missing
        raise ValueError(f"the ED files keep only the {N_EIG} lowest levels of each (L, Z, P)")
    if ry is not None and (L != 0 or ry != 1):
        raise ValueError("Ry exists at Lz = 0 only, where the L = 0 targets have Ry = +1")

    def P(l):
        return None if ry_ph is None else ry_ph * (-1) ** l

    def label(l):
        return f"L={l}" + ("" if Z is None else f" Z={Z}") + ("" if ry_ph is None else f" P={P(l)}")

    exact = N in sizes()
    if verbose and not exact:
        print(f"N={N} not in the ED sizes {sizes()}: gaps from the a + b/sqrt(N) fits")

    def gap(l, k):
        if exact:
            lv = levels(N, l, Z, P(l))
            return lv[k - 1] - load_spectrum(N)[0, 0] if k <= len(lv) else np.nan
        a, b = gap_fit(l, Z, k, P(l))[2]
        return a + b / np.sqrt(N)

    gap_target = gap(L, n_states)
    if np.isnan(gap_target):
        raise ValueError(f"level {n_states} of {label(L)} is not among the computed levels at N={N}")
    if verbose:
        print(f"{label(L)} level {n_states}: gap {gap_target}" + (f", margin {margin}" if margin else ""))
    lam = 0.0
    for l in range(L + 1, L_MAX + 1):
        if ry is not None and (-1) ** l != ry:
            continue
        gap_next = gap(l, 1)
        if np.isnan(gap_next):
            if verbose:
                print(f"lowest {label(l)}: not among the computed levels, skipped")
            continue
        need = (gap_target + margin - gap_next) / (l * (l + 1) - L * (L + 1))
        if verbose:
            print(f"lowest {label(l)}: gap {gap_next}, needs lambda > {need}")
        lam = max(lam, need)
    return lam


def lz_dims(N: int) -> dict:
    """Number of configurations of N electrons in 2 flavours x N orbitals at each 2 Lz."""
    dp = [dict() for _ in range(N + 1)]
    dp[0][0] = 1
    for m2 in range(-(N - 1), N, 2):
        for _ in range(2):
            for n in range(N, 0, -1):
                for M2, c in dp[n - 1].items():
                    dp[n][M2 + m2] = dp[n].get(M2 + m2, 0) + c
    return dp[N]


def random_state_LL(N: int, lz: int) -> float:
    """Mean L(L+1) - Lz(Lz+1), the L^- L^+ penalty per unit weight, of a random state of the Lz sector: every multiplet
    of L >= Lz, D(L) - D(L+1) of them, has one state there. Grows ~N^3 (241 at N=14, 1189 at N=24, Lz=0)."""
    D = lz_dims(N)
    Ls = np.arange(lz, max(D) // 2 + 1)
    n = np.array([D.get(2 * L, 0) - D.get(2 * L + 2, 0) for L in Ls], dtype=float)
    return float(np.sum(n * (Ls * (Ls + 1) - lz * (lz + 1))) / n.sum())
