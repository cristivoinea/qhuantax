r"""References for the rotated backflow models: the sector-:math:`k` block determinant and the sum of
several of them, each supported on sector :math:`k` alone."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from qhuantax.quantumhall_excitons import block_backflow, block_orbitals

from .models import RotatedDetBackflow


def det_block_reference(net, d: int, nm: int, k: int, theta: float = 0.0,
                        charge_only: bool = True, gap: float = 20.0, holes=None, particles=None,
                        seed: int = 0):
    r"""``RotatedDetBackflow`` on ``block_orbitals(nm, k)``, restricted to sector :math:`k` by
    ``block_backflow`` (``W`` on the excitation columns only, up features suppressed by
    :math:`e^{-g}`). The restriction acts on the unrotated matrix, so the state is exactly
    :math:`R(\theta)` of a sector-:math:`k` state when ``charge_only``.

    :return: ``(model, masks)``; ``masks`` also freezes ``theta``.
    """
    U0, _ = block_orbitals(nm, k, holes, particles, seed)
    model = RotatedDetBackflow(net, jnp.asarray(U0), d=d, theta=theta, charge_only=charge_only)
    model, masks = block_backflow(model, k, gap)
    masks["theta"] = jnp.zeros(())
    return model, masks


def multidet_block_reference(net, d: int, nm: int, k: int, ndets: int, theta: float = 0.0,
                             charge_only: bool = True, gap: float = 20.0, frame_noise: float = 0.1,
                             seed: int = 0):
    r"""``RotatedMultiDetBackflow`` of ``ndets`` sector-:math:`k` block terms (``block_orbitals``),
    restricted as ``block_backflow`` restricts one: each ``W`` on its excitation columns only, the
    shared up features suppressed by :math:`e^{-g}`. The sum is supported on sector :math:`k` alone.
    Term 0 has the default hole and particle frames; the others perturb both by ``frame_noise`` so
    that the terms start apart.

    :return: ``(model, masks)`` for ``pretrain_masked``; ``masks`` also freezes ``theta``.
    """
    from qhuantax.quantumhall_excitons import hole_frame

    from .models import RotatedMultiDetBackflow

    rng = np.random.default_rng(seed)
    U0s, U0_masks = [], []
    for a in range(ndets):
        eps = 0.0 if a == 0 else frame_noise
        holes = hole_frame(nm, k) + eps * rng.standard_normal((nm, k))
        particles = hole_frame(nm, k) + eps * rng.standard_normal((nm, k))
        U0, mask = block_orbitals(nm, k, holes, particles, seed=seed + a)
        U0s.append(U0); U0_masks.append(mask)
    model = RotatedMultiDetBackflow(net, jnp.asarray(np.stack(U0s)), d=d, theta=theta,
                                    charge_only=charge_only)
    Nt = model.W.shape[1]
    W_mask = jnp.zeros_like(model.W).at[:, Nt - k:, :].set(1.0)
    model = set_gap(eqx.tree_at(lambda m: m.W, model, model.W * W_mask), gap)
    return model, {"W": W_mask, "U0": jnp.asarray(np.stack(U0_masks)), "theta": jnp.zeros(())}


def set_gap(model, gap: float):
    r"""Reset the up-feature suppression of the last FFN bias to :math:`e^{-g}` (``block_backflow``
    puts :math:`g=20`); a smaller :math:`g` lets the up features, and their gradients, back in."""
    b2 = model.net.layers[-2].b2.at[0::2].set(-gap * float(np.sqrt(model.net.nblocks + 1)))
    return eqx.tree_at(lambda m: m.net.layers[-2].b2, model, b2)
