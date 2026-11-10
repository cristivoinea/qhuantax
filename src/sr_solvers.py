r"""
Solvers for the SR equation :math:`\bar O \, \delta\theta = \bar\epsilon`: the ``solver`` argument of quantax's QNGD
optimizers (`NaturalExcitedSR`, `PenalizedAdamSR`, the supervised fits, ...). `make_solver` builds one by name;
`SOLVER_KINDS` lists the names. All work in float64 (complex128 for complex inputs) and return the input dtype.

- ``shift_eig`` (the default): quantax's shifted solve, MinSR
  :math:`x = A^\dagger (A A^\dagger + \epsilon I)^{-1} b` with fewer samples than parameters and
  :math:`x = (A^\dagger A + \epsilon I)^{-1} A^\dagger b` otherwise,
  :math:`\epsilon = \mathrm{rshift}\,\mathrm{Tr}(M)/\sqrt{\dim M} + \mathrm{ashift}` (rshift 1e-12, ashift 1e-6), with the
  shifted matrix inverted through its eigendecomposition.
- ``shift_cholesky``: quantax's `~quantax.optimizer.auto_shift_eig` itself, the same shifted solve by Cholesky.
- ``pinv_eig``: quantax's `~quantax.optimizer.auto_pinv_eig`, the eigendecomposition with the smooth cutoff
  :math:`1 / (w (1 + (\mathrm{rtol}\, w_\max / w)^6))` (rtol 1e-12) in place of the shift, and optionally quantax's
  signal-to-noise truncation ``tol_snr`` (Schmitt and Heyl, PRL 125, 100503; 0 disables).

Why ``shift_eig`` is the default: Cholesky needs every pivot of :math:`M + \epsilon I` positive. For Adam's second solve
(on :math:`\bar O / V`) at N=24, K=4, 4096 samples, about 3970 of the 4096 eigenvalues of M sit at rounding level and
:math:`\epsilon` is only ~1.5x the rounding of the factorisation, so Cholesky returned NaN on 4 of 6 steps after a
restart (and once in 3242 sweeps of a long run); the GPU eigendecomposition never failed and gives the same step up to
rounding, at ~0.14 s per solve instead of ~0.002 s (cluster test of 2026-11-10,
data/rot_test/diag/adam_nan_cases.py).
"""
from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import quantax as qtx
from jax.scipy.linalg import eigh

try:        # quantax's helpers for spreading the MinSR matrix over devices; a no-op on one device
    from quantax.utils import array_extend, get_distributed_sharding
except ImportError:     # pragma: no cover
    array_extend = get_distributed_sharding = None

SOLVER_KINDS = ("shift_eig", "shift_cholesky", "pinv_eig")


def _wide(x: jax.Array) -> jax.Array:
    return x.astype(jnp.complex128 if jnp.iscomplexobj(x) else jnp.float64)


def _shifted_inverse(M: jax.Array, rhs: jax.Array, rshift: Optional[float], ashift: float) -> jax.Array:
    """(M + eps I)^-1 rhs through the eigendecomposition of M + eps I, eps as quantax's ``_diag_shift``; a computed
    eigenvalue at or below zero (a numerically null direction) is left out."""
    rshift = 1e-12 if rshift is None else rshift
    M = M + (rshift * jnp.trace(M).real / jnp.sqrt(M.shape[0]) + ashift) * jnp.identity(M.shape[0], M.dtype)
    w, U = eigh(M)
    inv = jnp.where(w > 0, 1 / jnp.where(w > 0, w, 1.0), 0.0)
    return jnp.matmul(U, inv * jnp.matmul(U.conj().T, rhs, precision="highest"), precision="highest")


def shift_eig(rshift: Optional[float] = None, ashift: float = 1e-6) -> Callable:
    """quantax's auto_shift_eig with the shifted matrix inverted by eigendecomposition instead of Cholesky."""

    @jax.jit
    def solution(A: jax.Array, b: jax.Array, **kwargs) -> jax.Array:
        input_dtype = A.dtype
        n, m = A.shape
        if n < m:       # MinSR, as quantax's minnorm_shift_eig
            Adag = A.conj().T
            if array_extend is not None:
                Adag = array_extend(Adag, jax.device_count())
                Adag = jax.lax.with_sharding_constraint(Adag, get_distributed_sharding())
            with jax.enable_x64(True):
                Adag = _wide(Adag)
                T = jnp.matmul(Adag.conj().T, Adag, precision="highest")
                y = _shifted_inverse(T, _wide(b), rshift, ashift)
                x = jnp.matmul(Adag, y, precision="highest").astype(input_dtype)
            return x[:m]
        with jax.enable_x64(True):      # SR, as quantax's lstsq_shift_eig
            A = _wide(A)
            S = jnp.matmul(A.conj().T, A, precision="highest")
            F = jnp.matmul(A.conj().T, _wide(b), precision="highest")
            return _shifted_inverse(S, F, rshift, ashift).astype(input_dtype)

    return solution


def make_solver(kind: str = "shift_eig", rshift: Optional[float] = None, ashift: float = 1e-6,
                rtol: Optional[float] = None, tol_snr: float = 0.0) -> Callable:
    """The solver called ``kind`` (one of `SOLVER_KINDS`): rshift and ashift set the shift of the two shift kinds,
    rtol and tol_snr the cutoff and the SNR truncation of ``pinv_eig``; the others' parameters are ignored."""
    if kind == "shift_eig":
        return shift_eig(rshift, ashift)
    if kind == "shift_cholesky":
        return qtx.optimizer.auto_shift_eig(rshift=rshift, ashift=ashift)
    if kind == "pinv_eig":
        return qtx.optimizer.auto_pinv_eig(rtol=rtol, tol_snr=tol_snr)
    raise ValueError(f"unknown solver {kind!r}; choose one of {SOLVER_KINDS}")


def default_solver() -> Callable:
    """The solver the qhuantax optimizers use when none is given: ``shift_eig`` with quantax's tolerances."""
    return make_solver("shift_eig")


def add_solver_arguments(parser) -> None:
    """--solver, --rshift, --pinv-rtol and --tol-snr on an argparse parser, for `solver_from_args`."""
    parser.add_argument("--solver", choices=SOLVER_KINDS, default="shift_eig",
                        help="SR solver (qhuantax.sr_solvers): shift_eig = quantax's shifted solve by "
                             "eigendecomposition (the default); shift_cholesky = quantax's own, by Cholesky (NaN when "
                             "a pivot goes negative); pinv_eig = quantax's eigendecomposition pseudo-inverse, smooth "
                             "cutoff at --pinv-rtol of the largest eigenvalue")
    parser.add_argument("--rshift", type=float, default=None,
                        help="shift_eig and shift_cholesky: relative diagonal shift, eps = rshift tr(M)/sqrt(dim M) + "
                             "1e-6; defaults to quantax's 1e-12")
    parser.add_argument("--pinv-rtol", type=float, default=None,
                        help="pinv_eig: cutoff relative to the largest eigenvalue; defaults to quantax's 1e-12")
    parser.add_argument("--tol-snr", type=float, default=0.0,
                        help="pinv_eig: drop eigen-directions whose force has a signal-to-noise ratio below this "
                             "(Schmitt and Heyl, PRL 125, 100503); 0 (the default) disables")


def solver_from_args(args: dict) -> Callable:
    """The solver chosen by `add_solver_arguments`'s options (``args`` as vars(parser.parse_args()))."""
    return make_solver(args["solver"], rshift=args["rshift"], rtol=args["pinv_rtol"], tol_snr=args["tol_snr"])
