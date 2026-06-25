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
    net: eqx.Module
    U0: jax.Array
    W: jax.Array
    coeffs: jax.Array
    dtype: jnp.dtype

    def __init__(
        self,
        net: eqx.Module,
        d: int,
        U0: jax.Array,
        coeffs: Optional[jax.Array] = None,
        dtype: jnp.dtype = jnp.float64,
    ):
        r"""
        Shared-backflow multi-determinant wavefunction.

        This is useful when an operator acts on a Slater determinant and generates a
        linear combination of determinants rather than another single determinant.
        """
        self.net = net
        self.dtype = dtype

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

        U0 = U0 / jnp.std(U0)
        self.U0 = U0.astype(dtype)

        ndets = U0.shape[0]
        if coeffs is None:
            coeffs = jnp.ones((ndets,), dtype=dtype) / ndets
        elif coeffs.shape != (ndets,):
            raise ValueError(f"coeffs must have shape {(ndets,)}, got {coeffs.shape}")
        self.coeffs = coeffs.astype(dtype)

        if sites.is_spinful:
            d //= 2
        self.W = qtx.nn.lecun_normal(
            qtx.global_defs.get_subkeys(), (sites.Ntotal, d), dtype=dtype
        ) / 10

    @property
    def ndets(self) -> int:
        return self.U0.shape[0]

    def __call__(self, s: jax.Array) -> jax.Array:
        x = self.net(s)

        idx = qtx.nn.fermion_idx(s)
        x = x.reshape(-1, qtx.global_defs.get_sites().Nfmodes).astype(self.dtype)
        x = x.T[idx]
        delta = x @ self.W.T

        U = self.U0[:, idx, :] + delta[None, :, :]
        sign, logabs = jnp.linalg.slogdet(U)
        psi = qtx.utils.LogArray(sign, logabs)
        return (psi * self.coeffs).sum() * qtx.nn.fermion_inverse_sign(s)


class OperatedModel(eqx.Module):
    base_model: eqx.Module
    operator: qtx.operator.Operator
    mode: str

    def __init__(self, base_model: eqx.Module, operator: qtx.operator.Operator):
        self.base_model = base_model
        self.operator = copy.deepcopy(operator)
        # Materialize the operator's JAX cache outside of traced code. If the first
        # `jax_op_list` access happens inside a sampler/model JIT, Quantax stores
        # traced arrays on the Python object and JAX raises an UnexpectedTracerError.
        _ = self.operator.jax_op_list

        has_diag = False
        has_off_diag = False
        for opstr, _ in self.operator.op_list:
            nflips = sum(1 for op in opstr if op not in ("I", "n", "z"))
            has_diag |= nflips == 0
            has_off_diag |= nflips > 0

        if has_diag and has_off_diag:
            raise ValueError(
                "OperatedModel currently supports operators that are either entirely "
                "diagonal or entirely off-diagonal."
            )
        self.mode = "diagonal" if has_diag else "off_diagonal"

    def __call__(self, s: jax.Array) -> jax.Array:
        s_batch = s[None, :]
        if self.mode == "diagonal":
            return self.operator.apply_diag(s_batch)[0] * self.base_model(s)

        out = jnp.array(0.0, dtype=qtx.get_default_dtype())
        off_diag = self.operator.apply_off_diag(s_batch)
        for _, (s_conn, H_conn) in off_diag.items():
            s_conn = s_conn[0]
            coeff = jnp.nan_to_num(H_conn[0], nan=0.0)
            valid = ~jnp.isclose(coeff, 0)
            safe_s_conn = jnp.where(valid[:, None], s_conn, s[None, :])
            psi_conn = jax.vmap(self.base_model)(safe_s_conn)
            out = out + jnp.sum(coeff * psi_conn)
        return out


def _init_params():
    keys = qtx.get_subkeys(3)

    
    Nsites = qtx.get_sites().Nsites
    v = jnp.ones((Nsites, Nsites))
    return v


class JastrowJacks(eqx.Module):
    v: jax.Array
    exact_state: jnp.array
    basis: spinless_fermion_basis_1d

    def __init__(self, exact_state: jnp.array, basis: spinless_fermion_basis_1d):
        self.exact_state = exact_state
        self.basis = basis
        self.v = _init_params()

        
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
        Nsites = qtx.get_sites().Nsites
        jacks = self.get_jack_coeff(s, Nsites)
        jack_sign = jnp.sign(jacks)
        jack_log = jnp.log(jnp.abs(jacks))
        
        psi = qtx.utils.LogArray(jack_sign, jack_log)
        jastrow = qtx.nn.exp_by_log(-0.5 * s @ self.v @ s)
        return jastrow * psi
