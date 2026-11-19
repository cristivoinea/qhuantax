r"""References for the rotated backflow models: the sector-:math:`k` block determinant and the sum of
several of them. Each lives on the sectors :math:`k'\le k` (``KSectors.space(k)``): exactly zero above
:math:`k` by construction, and below :math:`k` of order (up features)\ :sup:`k-k'`, which the fit on
that space sets (target 0 for a sector-:math:`k` state, or any :math:`k'\le k` superposition)."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from qhuantax.quantumhall_excitons import block_backflow, block_orbitals

from .models import RotatedDetBackflow


def det_block_reference(net, d: int, nm: int, k: int, theta: float = 0.0,
                        charge_only: bool = True, holes=None, particles=None, seed: int = 0):
    r"""``RotatedDetBackflow`` on ``block_orbitals(nm, k)``, restricted by ``block_backflow`` (``W`` on
    the excitation columns only): zero above sector :math:`k`, the up features free. With
    ``charge_only`` the state is exactly :math:`R(\theta)` of the :math:`\theta=0` one.

    :return: ``(model, masks)``; ``masks`` also freezes ``theta``.
    """
    U0, _ = block_orbitals(nm, k, holes, particles, seed)
    model = RotatedDetBackflow(net, jnp.asarray(U0), d=d, theta=theta, charge_only=charge_only)
    model, masks = block_backflow(model, k)
    masks["theta"] = jnp.zeros(())
    return model, masks


def multidet_block_reference(net, d: int, nm: int, k: int, ndets: int, theta: float = 0.0,
                             charge_only: bool = True, frame_noise: float = 0.1, seed: int = 0):
    r"""``RotatedMultiDetBackflow`` of ``ndets`` sector-:math:`k` block terms (``block_orbitals``) on one
    shared network, each ``W`` on its excitation columns only (as ``block_backflow`` restricts one): the
    sum is zero above sector :math:`k`. Term 0 has the default hole and particle frames; the others
    perturb both by ``frame_noise`` so that the terms start apart.

    :return: ``(model, masks)`` for ``pretrain_exact``; ``masks`` also freezes ``theta``.
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
    model = eqx.tree_at(lambda m: m.W, model, model.W * W_mask)
    return model, {"W": W_mask, "U0": jnp.asarray(np.stack(U0_masks)), "theta": jnp.zeros(())}


def set_gap(model, gap: float, flavour: str = "up"):
    r"""Legacy: members of ansatz runs before 2026-11-18 were built with one flavour's features suppressed
    by a bias, the last FFN's ``b2`` on its channels (even: up modes, odd: down modes) at
    :math:`-g\sqrt{n_b+1}`, :math:`e^{-g}` with :math:`g=20`: the up ones for the block members
    (:math:`k\ge1`), the down ones for the live vacuum (:math:`k=0`, a random ``W`` on every column). This
    resets that bias to :math:`e^{-g}`; the release for :math:`h>0` lowered it to :math:`g=8` so that the
    features, and their gradients, come back in. Members built since carry no such bias."""
    if flavour not in ("up", "down"):
        raise ValueError(f"flavour 'up' or 'down', got {flavour!r}")
    channels = slice(0, None, 2) if flavour == "up" else slice(1, None, 2)
    b2 = model.net.layers[-2].b2.at[channels].set(-gap * float(np.sqrt(model.net.nblocks + 1)))
    return eqx.tree_at(lambda m: m.net.layers[-2].b2, model, b2)


def k1_block_reference(net, d: int, nm: int, ell: int, mu: int, seed: int = 0):
    r"""The :math:`k=1` member :math:`B_{\ell\mu}|\Uparrow\rangle` in block form (``det_block_reference`` with
    :math:`k=1`): hole frame :math:`a_j=t^{\ell\mu}_{j+\mu,j}` (normalised), particle frame all ones. On
    sector 1 it is exact at ``W = 0`` at every :math:`L_z=\mu`, one determinant (N=8: 1-F ~1e-7, also at
    :math:`\mu=0`, :math:`\ell\ge1`, where the plain construction needs two); the vacuum amplitude (sector 0,
    :math:`L_z=0` only) is linear in the up features. ``W`` starts random on the excitation column; fit it
    on ``KSectors.space(1)`` like the :math:`k\ge2` members.

    :return: ``(model, masks)`` for ``pretrain_exact``.
    """
    from qhuantax.quantumhall_meanfield import multipole_tensor

    t = multipole_tensor(nm, ell, mu)
    a = np.array([t[j + mu, j] if j + mu < nm else 0.0 for j in range(nm)])
    return det_block_reference(net, d, nm, 1, holes=a / np.linalg.norm(a), particles=np.ones(nm), seed=seed)
