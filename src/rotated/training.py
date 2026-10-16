r"""Exact-sum supervised fits for the block (sector-:math:`k`) rotated models, with restarts.

The multi-determinant charge-only block fit at :math:`h=0`, :math:`k=2` either converges (below
:math:`2\cdot10^{-4}` by step 500 at :math:`N=8,10`, :math:`4` dets) or sticks on a plateau that is
flat from step 500 (:math:`10^{-3}`--:math:`10^{-2}`), about half of the initialisations each. A
stuck run is recognisable early, so ``pretrain_exact`` can abort it and ``fit_block`` re-initialise.
"""

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from quantax.utils import to_distributed_array


def pretrain_exact(model, spins, target, masks: dict | None = None, steps: int = 4000,
                   lr: float = 1e-2, lr_t0: float | None = None, abort_at: int | None = 1000,
                   abort_above: float = 1e-3, log_every: int = 500, log_prefix: str | None = None,
                   projection: tuple | None = None):
    r"""``quantumhall_excitons.pretrain_masked`` (full-batch Adam on the exact infidelity over
    ``spins``, gradients multiplied by ``masks``, best parameters kept) with an early abort: at step
    ``abort_at`` the run stops if the best :math:`1-F` so far is above ``abort_above`` and has not
    halved since step ``abort_at/2``.

    A non-finite gradient also stops the run (``aborted``), with a message naming the parameters it hit.
    The configurations are split over all JAX devices (e.g. ``JAX_NUM_CPU_DEVICES``), padded with copies of
    the first one at weight 0. With ``log_prefix``, every ``log_every`` steps print a line starting with it.

    ``projection = (index, sign, eigval)`` fits the symmetrized model instead, the one a projected variational
    state uses: :math:`\psi_P(s)=\frac12[\psi(s)+\lambda\,\sigma_s\psi(s')]`, with :math:`s'=` ``spins[index[s]]`` the
    image of :math:`s` under an involutive symmetry that maps ``spins`` onto itself and :math:`\sigma_s` = ``sign[s]``
    (``<s|U|psi> = sign psi(s')``). The images are among ``spins``, so this costs nothing extra.

    :return: ``(model, history, aborted)``; ``history`` lists ``(step, 1-F)`` every ``log_every``
        steps and ends with the best value.
    """
    masks = masks or {}
    lr_t0 = steps / 4 if lr_t0 is None else lr_t0
    n, pad = len(spins), -len(spins) % jax.device_count()
    data = (to_distributed_array(np.concatenate([np.asarray(spins), np.repeat(np.asarray(spins)[:1], pad, 0)])),
            to_distributed_array(np.r_[np.ones(n), np.zeros(pad)]),
            to_distributed_array(np.r_[target / np.linalg.norm(target), np.zeros(pad)]))
    params, static = eqx.partition(model, eqx.is_inexact_array)
    start_time = time.time()

    if projection is not None:
        img_index, img_sign, eigval = jnp.asarray(projection[0]), jnp.asarray(projection[1], float), float(projection[2])

    def infidelity(p, data):
        spins, weight, tgt = data
        psi = jax.vmap(eqx.combine(p, static))(spins)
        amp = weight * psi.sign * jnp.exp(psi.logabs - jnp.max(psi.logabs))
        if projection is not None:
            amp, tgt = amp[:n], tgt[:n]
            amp = 0.5 * (amp + eigval * img_sign * amp[img_index])
        return 1.0 - (amp @ tgt) ** 2 / (amp @ amp)

    def mask(g):
        for name, M in masks.items():
            g = eqx.tree_at(lambda m, n=name: getattr(m, n), g, getattr(g, name) * M)
        return g

    @eqx.filter_jit
    def step(p, m1, m2, it, data):
        val, g = jax.value_and_grad(infidelity)(p, data)
        g = mask(g)
        finite = jax.tree_util.tree_map(lambda x: jnp.all(jnp.isfinite(x)), g)
        a = lr / (1 + it / lr_t0)
        m1 = jax.tree_util.tree_map(lambda x, y: 0.9 * x + 0.1 * y, m1, g)
        m2 = jax.tree_util.tree_map(lambda x, y: 0.999 * x + 0.001 * y * y, m2, g)
        # cast back: the float64 step counter would otherwise promote float32 network weights, and
        # the saved model would no longer load into a freshly built (float32) skeleton
        p = jax.tree_util.tree_map(
            lambda q, x, y: (q - a * (x / (1 - 0.9 ** it)) / (jnp.sqrt(y / (1 - 0.999 ** it)) + 1e-12)).astype(q.dtype),
            p, m1, m2)
        return p, m1, m2, val, finite, jnp.all(jnp.stack(jax.tree_util.tree_leaves(finite)))

    m1 = jax.tree_util.tree_map(jnp.zeros_like, params)
    m2 = jax.tree_util.tree_map(jnp.zeros_like, params)
    history, best, at_half, aborted = [], (np.inf, params), None, False
    for it in range(1, steps + 1):
        new, m1, m2, val, finite, all_finite = step(params, m1, m2, jnp.asarray(it, dtype=jnp.float64), data)
        val = float(val)
        if not bool(all_finite):
            # Adam turns one non-finite gradient into non-finite parameters for good: stop, keep the best
            bad = [jax.tree_util.keystr(path) for path, ok in jax.tree_util.tree_flatten_with_path(finite)[0]
                   if not bool(ok)]
            print(f"{log_prefix or '     '} step {it}: non-finite gradient (1-F {val:.3e} here) in {', '.join(bad)}; "
                  f"fit stopped, best 1-F so far {best[0]:.3e} kept", flush=True)
            history.append((it, val))
            aborted = True
            break
        if val < best[0]:
            best = (val, params)            # the loss of `params`, before this update
        params = new
        if it % log_every == 0 or it == steps:
            history.append((it, val))
            if log_prefix is not None:
                print(f"{log_prefix} step {it}: 1-F {val:.3e} (best {best[0]:.3e}) [{time.time() - start_time:.0f}s]",
                      flush=True)
        # judged on the best value so far: the loss spikes, and a spike at abort_at is not a plateau
        if abort_at and it == abort_at // 2:
            at_half = best[0]
        if abort_at and it == abort_at and best[0] > abort_above and best[0] > 0.5 * at_half:
            aborted = True
            break
    final = float(infidelity(params, data))
    if final < best[0]:
        best = (final, params)
    history.append((it, best[0]))
    return eqx.combine(best[1], static), history, aborted


def fit_block(build, spins, target, steps: int = 4000, attempts: int = 3, success: float = 1e-4,
              verbose: bool = True, **kwargs):
    r"""Fit a block model, re-initialising until :math:`1-F<` ``success`` or ``attempts`` run out.

    :param build: ``build(attempt) -> (model, masks)``, a fresh initialisation per call (e.g.
        ``lambda a: multidet_block_reference(new_net(), d, nm, 2, ndets, seed=a)``).
    :param kwargs: passed to ``pretrain_exact``.
    :return: ``(model, best 1-F, log)``, the best model over the attempts; ``log`` has one entry
        ``(attempt, 1-F, aborted, seconds, history)`` per attempt.
    """
    best, log = (np.inf, None), []
    for a in range(attempts):
        model, masks = build(a)
        t = time.time()
        model, hist, aborted = pretrain_exact(model, spins, target, masks, steps=steps, **kwargs)
        dt = time.time() - t
        log.append((a, hist[-1][1], aborted, dt, hist))
        if verbose:
            traj = " ".join(f"{s}:{v:.1e}" for s, v in hist[:-1])
            print(f"      attempt {a}: 1-F {hist[-1][1]:.3e}{' (aborted)' if aborted else ''}, "
                  f"{dt / 60:.1f} min | {traj}", flush=True)
        if hist[-1][1] < best[0]:
            best = (hist[-1][1], model)
        if best[0] < success:
            break
    return best[1], best[0], log
