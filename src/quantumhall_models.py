from typing import Sequence, Tuple, Optional, Union
import copy
from jaxtyping import Key
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import quantax as qtx
import equinox as eqx


class MultiDetBackflow(eqx.Module):
    models: Tuple[eqx.Module, ...]
    coeffs: jax.Array
    dtype: jnp.dtype

    def __init__(
        self,
        nets: Sequence[eqx.Module],
        d: int,
        U0: jax.Array,
        coeffs: Optional[jax.Array] = None,
        dtype: jnp.dtype = jnp.float64,
    ):
        r"""
        Sum of independent determinant-backflow wavefunctions.

        Each term has its own backflow network and mean-field orbital matrix,
        and all terms are optimized together as one Equinox pytree.
        """
        self.dtype = dtype
        nets = tuple(nets)
        if len(nets) == 0:
            raise ValueError("nets must contain at least one backflow network.")

        sites = qtx.global_defs.get_sites()
        expected_shape = (sites.Nfmodes, sites.Ntotal)
        if U0.ndim == 2:
            if U0.shape != expected_shape:
                raise ValueError(f"U0 must have shape {expected_shape}, got {U0.shape}")
            U0 = U0[None, ...]
        elif U0.ndim == 3:
            if U0.shape[1:] != expected_shape:
                raise ValueError(
                    f"U0 must have shape (ndets, {expected_shape[0]}, {expected_shape[1]}),"
                    f" got {U0.shape}"
                )
        else:
            raise ValueError("U0 must be a rank-2 or rank-3 array.")

        ndets = U0.shape[0]
        if len(nets) != ndets:
            raise ValueError(f"nets must contain {ndets} networks, got {len(nets)}")

        if coeffs is None:
            coeffs = jnp.ones((ndets,), dtype=dtype) / ndets
        elif coeffs.shape != (ndets,):
            raise ValueError(f"coeffs must have shape {(ndets,)}, got {coeffs.shape}")
        self.coeffs = coeffs.astype(dtype)
        self.models = tuple(
            qtx.model.DetBackflow(net, U0=U0_det, d=d, dtype=dtype)
            for net, U0_det in zip(nets, U0)
        )

    @property
    def ndets(self) -> int:
        return len(self.models)

    def __call__(self, s: jax.Array) -> jax.Array:
        psi = self.coeffs[0] * self.models[0](s)
        for coeff, model in zip(self.coeffs[1:], self.models[1:]):
            psi = psi + coeff * model(s)
        return psi


class MultiPfBackflow(eqx.Module):
    models: Tuple[eqx.Module, ...]
    coeffs: jax.Array
    dtype: jnp.dtype

    def __init__(
        self,
        nets: Sequence[eqx.Module],
        d: int,
        U0: jax.Array,
        coeffs: Optional[jax.Array] = None,
        dtype: jnp.dtype = jnp.float64,
    ):
        r"""
        Sum of independent pfaffian-backflow wavefunctions.

        Each term has its own backflow network and mean-field pairing matrix,
        and all terms are optimized together as one Equinox pytree.
        """
        self.dtype = dtype
        nets = tuple(nets)
        if len(nets) == 0:
            raise ValueError("nets must contain at least one backflow network.")

        sites = qtx.global_defs.get_sites()
        expected_shape = (sites.Nfmodes, sites.Nfmodes)
        if U0.ndim == 2:
            if U0.shape != expected_shape:
                raise ValueError(f"U0 must have shape {expected_shape}, got {U0.shape}")
            U0 = U0[None, ...]
        elif U0.ndim == 3:
            if U0.shape[1:] != expected_shape:
                raise ValueError(
                    f"U0 must have shape (npf, {expected_shape[0]}, {expected_shape[1]}),"
                    f" got {U0.shape}"
                )
        else:
            raise ValueError("U0 must be a rank-2 or rank-3 array.")

        npf = U0.shape[0]
        if len(nets) != npf:
            raise ValueError(f"nets must contain {npf} networks, got {len(nets)}")

        if coeffs is None:
            coeffs = jnp.ones((npf,), dtype=dtype) / npf
        elif coeffs.shape != (npf,):
            raise ValueError(f"coeffs must have shape {(npf,)}, got {coeffs.shape}")
        self.coeffs = coeffs.astype(dtype)
        self.models = tuple(
            qtx.model.PfBackflow(net, U0=U0_pf, d=d, dtype=dtype)
            for net, U0_pf in zip(nets, U0)
        )

    @property
    def npf(self) -> int:
        return len(self.models)

    def __call__(self, s: jax.Array) -> jax.Array:
        psi = self.coeffs[0] * self.models[0](s)
        for coeff, model in zip(self.coeffs[1:], self.models[1:]):
            psi = psi + coeff * model(s)
        return psi