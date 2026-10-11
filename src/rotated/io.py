r"""Saving and rebuilding rotated members. ``eqx.tree_deserialise_leaves`` needs a model with the same
tree (and the same static fields, ``charge_only`` included), so every member is saved together with a
small json spec from which that tree is rebuilt:

    {"kind": "det" | "multidet", "ndets": int (multidet only), "d": int, "nb": int, "nh": int,
     "charge_only": bool}
"""

import equinox as eqx
import jax.numpy as jnp

from qhuantax.quantumhall_transformer import Transformer

from .models import RotatedDetBackflow, RotatedMultiDetBackflow


def member_spec(model, d: int, nb: int, nh: int) -> dict:
    """The json spec of a ``RotatedDetBackflow`` or ``RotatedMultiDetBackflow``."""
    spec = dict(d=d, nb=nb, nh=nh, charge_only=model.charge_only)
    if isinstance(model, RotatedMultiDetBackflow):
        return dict(kind="multidet", ndets=model.ndets, **spec)
    return dict(kind="det", **spec)


def member_skeleton(spec: dict, nm: int):
    """A model with the tree of ``spec`` (values arbitrary), to deserialise into."""
    net = Transformer(nblocks=spec["nb"], d=spec["d"], heads=spec["nh"], final_sum=False)
    if spec["kind"] == "multidet":
        return RotatedMultiDetBackflow(net, jnp.zeros((spec["ndets"], 2 * nm, nm)), d=spec["d"],
                                       charge_only=spec["charge_only"], rescale=False)
    return RotatedDetBackflow(net, jnp.zeros((2 * nm, nm)), d=spec["d"],
                              charge_only=spec["charge_only"], rescale=False)


def load_member(spec: dict, nm: int, file):
    return eqx.tree_deserialise_leaves(file, member_skeleton(spec, nm))
