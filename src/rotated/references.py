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


def set_gap(model, gap: float, flavour: str = "up"):
    r"""Reset the suppression of one flavour's features, the last FFN bias of its channels (even: up modes, odd: down
    modes), to :math:`e^{-g}`. The block members (:math:`k\ge1`, ``block_backflow``) suppress the up features, the live
    vacuum (``live_vacuum``) the down ones, both at :math:`g=20`; a smaller :math:`g` lets those features, and their
    gradients, back in."""
    if flavour not in ("up", "down"):
        raise ValueError(f"flavour 'up' or 'down', got {flavour!r}")
    channels = slice(0, None, 2) if flavour == "up" else slice(1, None, 2)
    b2 = model.net.layers[-2].b2.at[channels].set(-gap * float(np.sqrt(model.net.nblocks + 1)))
    return eqx.tree_at(lambda m: m.net.layers[-2].b2, model, b2)


def live_vacuum(net, d: int, nm: int, gap: float = 20.0):
    r"""The :math:`k=0` member with a live backflow, the mirror image of the block members: the vacuum orbitals
    :math:`U_0=\binom{I}{0}`, a random ``W`` on every column and the *down* features suppressed by :math:`e^{-g}`.
    The down rows of the orbital matrix stay :math:`O(e^{-g})`, so every configuration with a down electron has a
    (near-)zero row, and on the only other one, :math:`|\Uparrow\rangle`, the up-row backflow is a number: the state is
    :math:`|\Uparrow\rangle` up to :math:`O(e^{-g})` for any ``W``, with the up features live from the start."""
    from .models import RotatedDetBackflow

    U0 = np.vstack([np.eye(nm), np.zeros((nm, nm))])
    return set_gap(RotatedDetBackflow(net, jnp.asarray(U0), d=d, charge_only=True), gap, flavour="down")


def k1_block_reference(net, d: int, nm: int, ell: int, mu: int, gap: float = 20.0, seed: int = 0):
    r"""The :math:`k=1` member :math:`B_{\ell\mu}|\Uparrow\rangle` in block form (``det_block_reference`` with
    :math:`k=1`): hole frame :math:`a_j=t^{\ell\mu}_{j+\mu,j}` (normalised), particle frame all ones. Exact at
    ``W = 0`` at every :math:`L_z=\mu`, one determinant (N=8: 1-F ~1e-7, vacuum amplitude 0, also at :math:`\mu=0`,
    :math:`\ell\ge1`, where the plain construction needs two). ``W`` starts random on the excitation column; fit it
    like the :math:`k=2` members.

    :return: ``(model, masks)`` for ``pretrain_exact``.
    """
    from qhuantax.quantumhall_meanfield import multipole_tensor

    t = multipole_tensor(nm, ell, mu)
    a = np.array([t[j + mu, j] if j + mu < nm else 0.0 for j in range(nm)])
    return det_block_reference(net, d, nm, 1, gap=gap, holes=a / np.linalg.norm(a), particles=np.ones(nm), seed=seed)
