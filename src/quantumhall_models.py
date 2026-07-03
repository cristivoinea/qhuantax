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
from quspin.basis import spinless_fermion_basis_1d, spinful_fermion_basis_1d


class Jacknet(qtx.model.DetBackflow):
    exact_state: jnp.array
    basis: spinless_fermion_basis_1d

    def __init__(
        self,
        net: eqx.Module,
        d: int,
        exact_state: jnp.array,
        basis: spinless_fermion_basis_1d,
        U0: Optional[jax.Array] = None,
        dtype: jnp.dtype = jnp.float64,
    ):
        r"""
        Initialize the determinant backflow model.

        :param net:
            The backflow network that outputs the correction to the mean-field orbitals.

        :param d:
            Channel (hidden) dimension of the network output.

        :param jack:
            Vector of exact (Jack) amplitudes of the Laughlin configuration.

        :param U0:
            The mean-field orbitals. If None, it's initialized close to
            non-interacting fermions.

        :param dtype:
            Data type of the parameters.
        """
        self.exact_state = exact_state
        self.basis = basis
        
        super().__init__(net, d, U0, dtype)

    def get_basis_int(self, s, Nsites):
        powers = 1 << jnp.arange(Nsites)
        return jnp.sum(s.astype(jnp.int64) * powers)

    def get_jack_coeff(self, s, Nsites):
        bits = (s+1)//2
        # convert rows to strings like original function expects
        k = self.get_basis_int(bits, Nsites)
        #print(keys)
        # lookup indices
        idx = self.basis.Ns - 1 - jnp.searchsorted(self.basis.states[::-1], k)

        return self.exact_state[idx]

    def __call__(self, s: jax.Array) -> jax.Array:
        x = self.net(s)

        idx = qtx.nn.fermion_idx(s)
        x = x.reshape(-1, qtx.global_defs.get_sites().Nfmodes).astype(self.dtype)
        x = x.T[idx]

        Nsites = qtx.get_sites().Nsites
        sign0, _ = jnp.linalg.slogdet(self.U0[idx,:])
        jacksign = jnp.sign(self.get_jack_coeff(s, Nsites))
        sign_correction = jnp.where(sign0 == jacksign, 1, -1)

        U = self.U0[idx, :] + x @ self.W.T
        sign, logabs = jnp.linalg.slogdet(U)
        psi = qtx.utils.LogArray(sign * sign_correction, logabs)

        #jack_coeff = self.jack[spinless_fermion_basis_1d(self.L, self.N).get_index(str((s+1)//2))]

        return psi * qtx.nn.fermion_inverse_sign(s)


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