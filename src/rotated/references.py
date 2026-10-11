r"""References for the rotated backflow models: the sector-:math:`k` block determinant, and its
embedding into a Pfaffian that is then released ("free Pfaffian").

Why the Pfaffian is released rather than given a block structure of its own: with charge-only
features the configurations without charge fluctuations all see one fixed pairing matrix, and every
Pfaffian with exact :math:`k=2` support (up-up rank :math:`N-2`, up-down rank 2 inside its range,
down-down 0 or rank 2) gives amplitudes of rank :math:`\le4` there, like the block determinant.
A Pfaffian whose :math:`U_0` has only :math:`N_\text{total}` nonzero columns is exactly a
determinant times :math:`\mathrm{pf}(J)`. The Pfaffian only gains when its :math:`k` support is
learned rather than built in.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from qhuantax.quantumhall_excitons import block_backflow, block_orbitals

from .models import RotatedDetBackflow, RotatedPfBackflow


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


def pf_from_det(model: RotatedDetBackflow, noise: float = 0.0, seed: int = 0) -> RotatedPfBackflow:
    r"""The ``RotatedPfBackflow`` with the same amplitudes as ``model`` (up to a sign and
    :math:`\mathrm{pf}(J)=1`), then optionally opened.

    The determinant's :math:`N` columns become the first :math:`N` of :math:`2n_m`, paired by
    ``skew_eye`` among themselves; the other columns of :math:`U_0` and rows of :math:`W` are 0.
    ``noise > 0`` fills the extra columns of :math:`U_0` and every entry of :math:`J` that is 0 with
    Gaussian noise of that size (relative to the unit-std :math:`U_0`), so that the extra columns
    enter and the training can move off the block determinant.
    """
    import lrux

    Nf, Nt = model.U0.shape
    ncols = Nf
    rng = np.random.default_rng(seed)
    U0 = np.zeros((Nf, ncols)); U0[:, :Nt] = np.asarray(model.U0)
    W = np.zeros((ncols, model.W.shape[1])); W[:Nt] = np.asarray(model.W)
    J = np.zeros((ncols, ncols)); J[:Nt, :Nt] = np.asarray(lrux.skew_eye(Nt // 2))
    if noise > 0:
        U0[:, Nt:] = noise * rng.standard_normal((Nf, ncols - Nt))
        R = noise * rng.standard_normal((ncols, ncols))
        J = J + np.where(J == 0, R - R.T, 0.0)
        J = np.triu(J, 1) - np.triu(J, 1).T
    return RotatedPfBackflow(model.net, jnp.asarray(U0), W=jnp.asarray(W), J0=jnp.asarray(J),
                             theta=model.theta, charge_only=model.charge_only,
                             dtype=model.dtype, rescale=False)


def set_gap(model, gap: float):
    r"""Reset the up-feature suppression of the last FFN bias to :math:`e^{-g}` (``block_backflow``
    puts :math:`g=20`); a smaller :math:`g` lets the up features, and their gradients, back in."""
    b2 = model.net.layers[-2].b2.at[0::2].set(-gap * np.sqrt(model.net.nblocks + 1))
    return eqx.tree_at(lambda m: m.net.layers[-2].b2, model, b2)
