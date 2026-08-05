from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve as _dense_solve


def _default_rshift(dtype) -> float:
    """Relative diagonal shift matched to the precision of ``Obar`` itself.

    Quantax picks the shift from the *solver's* working dtype, which is float64 by
    default, so a float32 ``Obar`` gets a 1e-12 shift -- five orders of magnitude
    below its own noise floor.
    """
    real_dtype = jnp.finfo(dtype).dtype
    if real_dtype == jnp.float64:
        return 1e-12
    if real_dtype == jnp.float32:
        return 1e-6
    return 1e-3


def minnorm_chunked(
    rshift: Optional[float] = None,
    ashift: float = 1e-6,
    chunk: int = 1 << 16,
) -> Callable[..., jax.Array]:
    r"""
    MinSR solver :math:`x = A^\dagger (A A^\dagger + \epsilon I)^{-1} b` for the
    strongly underdetermined case, without any :math:`O(N_s N_p)` temporary.

    `~quantax.optimizer.minnorm_shift_eig` materialises ``A.conj().T`` and upcasts
    it to float64, costing 3x the size of ``A`` in scratch space; combined with the
    two solves per `~quantax.optimizer.Adam` step and the ``Obar / diag_preconditioner``
    fallback in `~quantax.optimizer.QNGD.solve_equation`, that is what pushes the
    ``jit_solve`` temp arena to ~3x ``Obar``. Here the Gram matrix is accumulated
    over column blocks of ``A``, so scratch is ``O(N_s * chunk)`` regardless of the
    number of parameters, and ``A`` is never transposed or duplicated.

    Precision is spent where it matters instead of uniformly: the Gram matrix is
    formed in the input dtype with ``precision="highest"`` (no TF32), and only the
    small :math:`N_s \times N_s` shifted solve is done in float64, which is the
    ill-conditioned part.

    ``diag_preconditioner`` is declared explicitly so that `~quantax.optimizer.Adam`
    passes it through rather than falling back to materialising ``Obar / V``.

    :param rshift:
        Relative diagonal shift, entering as
        :math:`\epsilon = \mathrm{Tr}(A A^\dagger) \times \mathrm{rshift} / \sqrt{N_s} + \mathrm{ashift}`.
        Defaults to 1e-6 for float32 input and 1e-12 for float64.

    :param ashift:
        Absolute diagonal shift, default 1e-6.

    :param chunk:
        Number of parameters contracted per block. Peak scratch is about
        ``2 * N_s * chunk`` elements, so this trades memory against kernel-launch
        overhead only.

    :return:
        A solver function ``(A, b) -> x`` suitable for the ``solver`` argument of
        a `~quantax.optimizer.QNGD` optimizer.
    """
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}.")

    @jax.jit
    def solution(
        A: jax.Array, b: jax.Array, *, diag_preconditioner=None, **kwargs
    ) -> jax.Array:
        nsamples, nparams = A.shape
        dtype = A.dtype

        # MinSR only. `auto_shift_eig` would switch to the SR branch here; this
        # solver would instead return a minimum-norm solution of a rank-deficient
        # system, which is silently wrong rather than merely inaccurate.
        if nsamples > nparams:
            raise ValueError(
                f"minnorm_chunked is a MinSR solver and requires an underdetermined "
                f"system, but got nsamples={nsamples} > nparams={nparams}. Use "
                f"quantax.optimizer.auto_shift_eig for this shape."
            )

        if diag_preconditioner is None:
            weight = None
        else:
            # Right preconditioning by D = diag(diag_preconditioner) means solving
            # with A D^-1 and rescaling the result by D^-1, so the Gram matrix picks
            # up D^-2 and the final vector picks up the same factor.
            weight = (1 / diag_preconditioner**2).astype(dtype)

        def accumulate(T, block, scaled):
            return T + jnp.matmul(scaled, block.conj().T, precision="highest")

        # A rolled `fori_loop` rather than a Python loop over static slices: an
        # unrolled loop lets XLA hoist every slice before the first matmul, which
        # keeps a full copy of A alive and defeats the point of chunking.
        def body(i, T):
            start = i * chunk
            block = jax.lax.dynamic_slice(A, (0, start), (nsamples, chunk))
            if weight is None:
                scaled = block
            else:
                scaled = block * jax.lax.dynamic_slice(weight, (start,), (chunk,))
            return accumulate(T, block, scaled)

        nfull, tail = divmod(nparams, chunk)
        if nfull:
            T = jax.lax.fori_loop(
                0, nfull, body, jnp.zeros((nsamples, nsamples), dtype)
            )
        else:
            T = jnp.zeros((nsamples, nsamples), dtype)
        if tail:
            block = A[:, nfull * chunk :]
            scaled = block if weight is None else block * weight[nfull * chunk :]
            T = accumulate(T, block, scaled)

        shift_rel = _default_rshift(dtype) if rshift is None else rshift
        with jax.enable_x64():
            wide = jnp.complex128 if jnp.iscomplexobj(A) else jnp.float64
            T = T.astype(wide)
            shift = shift_rel * jnp.trace(T).real / jnp.sqrt(nsamples) + ashift
            T = T + shift * jnp.eye(nsamples, dtype=wide)
            y = _dense_solve(T, b.astype(wide), assume_a="pos")
        y = y.astype(dtype)

        # Contracting the sample axis directly; A is read, never transposed.
        x = jnp.einsum("sk,s->k", A.conj(), y, precision="highest")
        if weight is not None:
            x = x * weight
        return x

    return solution
