from __future__ import annotations
from typing import Optional, Tuple, Union, BinaryIO
from jaxtyping import PyTree
from pathlib import Path

from warnings import warn
from functools import partial
from enum import Enum
import copy
import numpy as np
import jax
import jax.numpy as jnp
import jax.flatten_util as jfu
import equinox as eqx

from quantax.state import State, Variational, VS_TYPE, DenseState
from quantax.symmetry import Symmetry, Identity
from quantax.nn import RefModel, fermion_idx, fermion_inverse_sign, exp_by_log
from quantax.utils import (
    chunk_map,
    shard_vmap,
    chunk_shard_vmap,
    to_distribute_array,
    filter_replicate,
    filter_tree_map,
    array_extend,
    tree_fully_flatten,
    tree_split_cpl,
    tree_combine_cpl,
    apply_updates,
    LogArray,
    ScaleArray,
    PsiArray,
)
from quantax.global_defs import get_default_dtype, is_default_cpl, get_sites, get_subkeys
from quantax.operator.operator import _get_conn_size, _get_conn, Operator
from quantax.utils import chunk_map, to_distribute_array, array_extend
from quspin.basis import spinless_fermion_basis_1d, spinful_fermion_basis_1d

from .quantumhall_symmetries import FuzzySphereSymmetry


class LaughlinExact(State):
    dense_state: jnp.array
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

        idx = fermion_idx(s)
        x = x.reshape(-1, get_sites().Nfmodes).astype(self.dtype)
        x = x.T[idx]

        Nsites = get_sites().Nsites
        sign0, _ = jnp.linalg.slogdet(self.U0[idx,:])
        jacksign = jnp.sign(self.get_jack_coeff(s, Nsites))
        sign_correction = jnp.where(sign0 == jacksign, 1, -1)

        U = self.U0[idx, :] + x @ self.W.T
        sign, logabs = jnp.linalg.slogdet(U)
        psi = LogArray(sign * sign_correction, logabs)

        #jack_coeff = self.jack[spinless_fermion_basis_1d(self.L, self.N).get_index(str((s+1)//2))]

        return psi * fermion_inverse_sign(s)


def _init_params():
    keys = get_subkeys(3)

    
    Nsites = get_sites().Nsites
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
        Nsites = get_sites().Nsites
        jacks = self.get_jack_coeff(s, Nsites)
        jack_sign = jnp.sign(jacks)
        jack_log = jnp.log(jnp.abs(jacks))
        
        psi = LogArray(jack_sign, jack_log)
        jastrow = exp_by_log(-0.5 * s @ self.v @ s)
        return jastrow * psi


class OperatedState(Variational):
    _base_symm: FuzzySphereSymmetry
    _operator: Operator
    _operator_dag: Operator
    _mode: str

    def __init__(
        self,
        model: eqx.Module,
        param_file: Optional[Union[str, Path, BinaryIO]],
        operator: Operator,
        base_symm: Optional[FuzzySphereSymmetry] = None,
        max_parallel: Union[None, int, Tuple[int, int], Tuple[int, int, int]] = None,
        use_ref: bool = True,
    ):

        self._base_symm = base_symm
        self._operator = copy.deepcopy(operator)
        self._operator_dag = self._operator.H
        # Materialize Quantax's internal operator cache outside JIT. Otherwise
        # `operator.apply_diag/off_diag` tries to build `jax_op_list` lazily inside
        # traced forward passes and leaks tracers.
        _ = self._operator.jax_op_list
        _ = self._operator_dag.jax_op_list

        has_diag = False
        has_off_diag = False
        for opstr, _ in self._operator.op_list:
            nflips = sum(1 for op in opstr if op not in ("I", "n", "z"))
            has_diag |= nflips == 0
            has_off_diag |= nflips > 0
        if has_diag and has_off_diag:
            raise ValueError(
                "OperatedState currently supports operators that are either entirely "
                "diagonal or entirely off-diagonal."
            )
        self._mode = "diagonal" if has_diag else "off_diagonal"
        if self._mode != "diagonal":
            raise NotImplementedError(
                "OperatedState currently implements only purely diagonal operators."
            )

        # the symm passed to the parent class init() is none so any conversion to DenseStates will fail.
        super().__init__(model, param_file, Identity(), max_parallel, use_ref)


    def _init_forward(self) -> None:
        def base_forward(model: eqx.Module, s: jax.Array) -> PsiArray:
            s_symm = self._base_symm.get_symm_spins(s)
            psi = jax.vmap(model)(s_symm)
            psi = self._base_symm.symmetrize(psi, s)
            return psi

        def single_forward(model: eqx.Module, s: jax.Array) -> PsiArray:
            psi = base_forward(model, s)
            action = self._operator.apply_diag(s[None, :])[0]
            return (action * psi).astype(get_default_dtype())

        self._single_forward = single_forward
        self._batch_forward = shard_vmap(single_forward, in_axes=(None, 0), out_axes=0)
        self._direct_forward = chunk_map(
            self._batch_forward, in_axes=(None, 0), chunk_size=self.forward_chunk
        )
        self._fulljit_forward = chunk_shard_vmap(
            single_forward, in_axes=(None, 0), out_axes=0, chunk_size=self.forward_chunk
        )

        def init_internal(model, s):
            s_symm = self._base_symm.get_symm_spins(s)
            return jax.vmap(model.init_internal)(s_symm)

        init_internal = chunk_shard_vmap(
            init_internal, in_axes=(None, 0), out_axes=0, chunk_size=self.ref_chunk
        )
        self._init_internal = eqx.filter_jit(init_internal)

        def ref_forward_with_updates(model, s, s_old, nflips, internal):
            s_symm = self._base_symm.get_symm_spins(s)
            s_old_symm = self._base_symm.get_symm_spins(s_old)
            forward = partial(model.ref_forward, return_update=True)
            forward = eqx.filter_vmap(forward, in_axes=(0, 0, None, 0))
            psi, internal = forward(s_symm, s_old_symm, nflips, internal)

            psi = self._base_symm.symmetrize(psi, s)
            action = self._operator.apply_diag(s[None, :])[0]
            return (action * psi).astype(get_default_dtype()), internal

        self._ref_forward_with_updates = chunk_shard_vmap(
            ref_forward_with_updates,
            in_axes=(None, 0, 0, None, 0),
            out_axes=(0, 0),
            chunk_size=self.ref_chunk,
        )

        def ref_forward(model, s, s_old, nflips, idx_segment, internal):
            s_symm = self._base_symm.get_symm_spins(s)
            s_old = s_old[idx_segment]
            s_old_symm = self._base_symm.get_symm_spins(s_old)
            internal = filter_tree_map(lambda x: x[idx_segment], internal)

            forward = partial(model.ref_forward, return_update=False)
            forward = eqx.filter_vmap(forward, in_axes=(0, 0, None, 0))
            psi = forward(s_symm, s_old_symm, nflips, internal)

            psi = self._base_symm.symmetrize(psi, s)
            action = self._operator.apply_diag(s[None, :])[0]
            return (action * psi).astype(get_default_dtype())

        self._batch_ref_forward = shard_vmap(
            ref_forward,
            in_axes=(None, 0, None, None, 0, None),
            out_axes=0,
            shard_axes=(None, 0, 0, None, 0, 0),
        )
        self._ref_forward = chunk_map(
            self._batch_ref_forward,
            in_axes=(None, 0, None, None, 0, None),
            chunk_size=self.forward_chunk,
        )

    def _init_backward(self) -> None:
        def grad_fn(model: eqx.Module, s: jax.Array) -> jax.Array:
            def forward(model, x):
                psi = self._single_forward(model, x)
                if self.vs_type == VS_TYPE.real_or_holomorphic:
                    psi = psi.astype(get_default_dtype())
                elif jnp.iscomplexobj(psi):
                    psi = (psi.real, psi.imag)
                return psi

            def output_fn(psi):
                if isinstance(psi, tuple):
                    psi = psi[0] + 1j * psi[1]

                if isinstance(psi, LogArray):
                    sign = psi.sign
                    logabs = psi.logabs
                    out = sign / jax.lax.stop_gradient(sign) + logabs
                elif isinstance(psi, ScaleArray):
                    significand = psi.significand
                    exponent = psi.exponent
                    out = significand / jax.lax.stop_gradient(significand) + exponent
                else:
                    psi = jnp.asarray(psi)
                    out = psi / jax.lax.stop_gradient(psi)
                return out

            if self.vs_type == VS_TYPE.real_or_holomorphic:
                delta = jax.grad(output_fn, holomorphic=self.holomorphic)(forward(model, s))
            else:
                psi = forward(model, s)
                output_real = lambda outputs: output_fn(outputs).real
                output_imag = lambda outputs: output_fn(outputs).imag
                delta_real = jax.grad(output_real)(psi)
                delta_imag = jax.grad(output_imag)(psi)

            if self.vs_type == VS_TYPE.non_holomorphic:
                model = tree_split_cpl(model)
                fn = lambda net, x: forward(tree_combine_cpl(net[0], net[1]), x)
            else:
                fn = forward

            def backward(net, s, delta):
                f_vjp = eqx.filter_vjp(fn, net, s)[1]
                vjp_vals, _ = f_vjp(delta)
                return tree_fully_flatten(vjp_vals)

            if self.vs_type == VS_TYPE.real_or_holomorphic:
                grad = backward(model, s, delta)
            else:
                grad_real_out = backward(model, s, delta_real)
                grad_imag_out = backward(model, s, delta_imag)
                grad = jax.lax.complex(grad_real_out, grad_imag_out)

            if self.vs_type == VS_TYPE.non_holomorphic:
                grad_real_param = grad[: grad.shape[0] // 2]
                grad_imag_param = grad[grad.shape[0] // 2 :]
                grad = jnp.concatenate([grad_real_param, grad_imag_param], axis=0)

            return grad.astype(get_default_dtype())

        self._grad_vmap = chunk_shard_vmap(
            grad_fn, in_axes=(None, 0), out_axes=0, chunk_size=self.backward_chunk
        )


class OffDiagonalOperatedState(Variational):
    _base_symm: FuzzySphereSymmetry
    _new_symm: Symmetry
    _operator: Operator
    _operator_dag: Operator

    def __init__(
        self,
        model: eqx.Module,
        param_file: Optional[Union[str, Path, BinaryIO]],
        operator: Operator,
        base_symm: Optional[FuzzySphereSymmetry] = None,
        new_symm: Optional[Symmetry] = None,
        max_parallel: Union[None, int, Tuple[int, int], Tuple[int, int, int]] = None,
        use_ref: bool = False,
    ):
        self._base_symm = base_symm
        self._new_symm = new_symm if new_symm is not None else Identity()
        self._operator = copy.deepcopy(operator)
        self._operator_dag = self._operator.H
        _ = self._operator.jax_op_list
        _ = self._operator_dag.jax_op_list

        has_diag = False
        has_off_diag = False
        for opstr, _ in self._operator.op_list:
            nflips = sum(1 for op in opstr if op not in ("I", "n", "z"))
            has_diag |= nflips == 0
            has_off_diag |= nflips > 0
        if has_diag or (not has_off_diag):
            raise ValueError(
                "OffDiagonalOperatedState requires an operator with only off-diagonal terms."
            )

        if use_ref and jax.process_index() == 0:
            warn(
                "OffDiagonalOperatedState currently disables reference updates and uses "
                "direct forward passes."
            )
        super().__init__(model, param_file, self._new_symm, max_parallel, use_ref=False)

    def todense(self, symm: Optional[Symmetry] = None) -> DenseState:
        if symm is None:
            symm = self._new_symm

        symm.basis_make()
        basis = symm.basis
        basis_ints = basis.states
        psi = jnp.asarray(self[basis_ints])
        symm_norm = basis.get_amp(basis_ints)
        if np.isrealobj(psi):
            symm_norm = symm_norm.real
        return DenseState(psi / symm_norm, symm)

    def _init_forward(self) -> None:
        def base_forward(model: eqx.Module, s: jax.Array) -> PsiArray:
            s_symm = self._base_symm.get_symm_spins(s)
            psi = jax.vmap(model)(s_symm)
            psi = self._base_symm.symmetrize(psi, s)
            return psi

        def single_forward(model: eqx.Module, s: jax.Array) -> PsiArray:
            out = None
            off_diags = self._operator_dag.apply_off_diag(s[None, :])

            for _, (s_conn, H_conn) in off_diags.items():
                coeff = jnp.conj(H_conn[0])
                coeff = jnp.nan_to_num(coeff, nan=0.0)
                valid = ~(jnp.isnan(coeff) | jnp.isclose(coeff, 0))
                safe_conn = jnp.where(
                    valid[:, None], s_conn[0], jnp.broadcast_to(s, s_conn[0].shape)
                )
                psi_conn = jax.vmap(lambda x: base_forward(model, x))(safe_conn)
                term = (psi_conn * coeff).sum()
                out = term if out is None else out + term

            return out.astype(get_default_dtype())

        self._single_forward = single_forward
        self._batch_forward = shard_vmap(single_forward, in_axes=(None, 0), out_axes=0)
        self._direct_forward = chunk_map(
            self._batch_forward, in_axes=(None, 0), chunk_size=self.forward_chunk
        )
        self._fulljit_forward = chunk_shard_vmap(
            single_forward, in_axes=(None, 0), out_axes=0, chunk_size=self.forward_chunk
        )

    def _init_backward(self) -> None:
        def grad_fn(model: eqx.Module, s: jax.Array) -> jax.Array:
            def forward(model, x):
                psi = self._single_forward(model, x)
                if self.vs_type == VS_TYPE.real_or_holomorphic:
                    psi = psi.astype(get_default_dtype())
                elif jnp.iscomplexobj(psi):
                    psi = (psi.real, psi.imag)
                return psi

            def output_fn(psi):
                if isinstance(psi, tuple):
                    psi = psi[0] + 1j * psi[1]

                if isinstance(psi, LogArray):
                    sign = psi.sign
                    logabs = psi.logabs
                    out = sign / jax.lax.stop_gradient(sign) + logabs
                elif isinstance(psi, ScaleArray):
                    significand = psi.significand
                    exponent = psi.exponent
                    out = significand / jax.lax.stop_gradient(significand) + exponent
                else:
                    psi = jnp.asarray(psi)
                    out = psi / jax.lax.stop_gradient(psi)
                return out

            if self.vs_type == VS_TYPE.real_or_holomorphic:
                delta = jax.grad(output_fn, holomorphic=self.holomorphic)(forward(model, s))
            else:
                psi = forward(model, s)
                output_real = lambda outputs: output_fn(outputs).real
                output_imag = lambda outputs: output_fn(outputs).imag
                delta_real = jax.grad(output_real)(psi)
                delta_imag = jax.grad(output_imag)(psi)

            if self.vs_type == VS_TYPE.non_holomorphic:
                model = tree_split_cpl(model)
                fn = lambda net, x: forward(tree_combine_cpl(net[0], net[1]), x)
            else:
                fn = forward

            def backward(net, s, delta):
                f_vjp = eqx.filter_vjp(fn, net, s)[1]
                vjp_vals, _ = f_vjp(delta)
                return tree_fully_flatten(vjp_vals)

            if self.vs_type == VS_TYPE.real_or_holomorphic:
                grad = backward(model, s, delta)
            else:
                grad_real_out = backward(model, s, delta_real)
                grad_imag_out = backward(model, s, delta_imag)
                grad = jax.lax.complex(grad_real_out, grad_imag_out)

            if self.vs_type == VS_TYPE.non_holomorphic:
                grad_real_param = grad[: grad.shape[0] // 2]
                grad_imag_param = grad[grad.shape[0] // 2 :]
                grad = jnp.concatenate([grad_real_param, grad_imag_param], axis=0)

            return grad.astype(get_default_dtype())

        self._grad_vmap = chunk_shard_vmap(
            grad_fn, in_axes=(None, 0), out_axes=0, chunk_size=self.backward_chunk
        )
