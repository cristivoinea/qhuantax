from __future__ import annotations

from typing import Optional

import numpy as np
from numba import carray, cfunc, int32, uint32, uint64
from quspin.basis.user import (
    count_particles_sig_32,
    count_particles_sig_64,
    map_sig_32,
    map_sig_64,
    next_state_sig_32,
    next_state_sig_64,
    op_sig_32,
    op_sig_64,
    user_basis,
)
from quantax.global_defs import PARTICLE_TYPE, get_sites

from .quantumhall_symmetries import FuzzySphereSymmetry


def _total_particles(Nparticles: int | tuple[int, ...] | None) -> int:
    if Nparticles is None:
        raise ValueError("Lz user bases require a fixed particle number.")
    if isinstance(Nparticles, tuple):
        return int(sum(Nparticles))
    return int(Nparticles)


def _basis_dtype(Nmodes: int) -> np.dtype:
    if Nmodes <= 32:
        return np.uint32
    if Nmodes <= 64:
        return np.uint64
    raise ValueError("QuSpin user_basis integer states support at most 64 modes.")


def _mode_lz_weight(mode: int, L: int) -> int:
    # Store doubled Lz weights as integers. Orbital m has
    # Lz = m - (L - 1) / 2, so 2*Lz = 2*m - (L - 1).
    return 2 * (mode % L) - (L - 1)


def _generate_lz_basis_states(
    L: int,
    Nparticles: int,
    lz: int,
    nflav: int,
    dtype: np.dtype,
) -> np.ndarray:
    """Generate exactly the integer states in the requested ``(N, Lz)`` sector.

    Internally this uses doubled angular momentum, so the orbital labels are
    ``2*m - (L - 1)`` for ``m=0, ..., L-1``. This corresponds to physical
    single-particle values ``-(L-1)/2, ..., (L-1)/2``.
    """
    Nmodes = nflav * L
    weights = [_mode_lz_weight(mode, L) for mode in range(nflav * L)]
    target = 2 * lz

    reachable = [[set() for _ in range(Nparticles + 1)] for _ in range(Nmodes + 1)]
    reachable[Nmodes][0].add(0)
    for mode in range(Nmodes - 1, -1, -1):
        weight = weights[mode]
        for n in range(Nparticles + 1):
            reachable[mode][n].update(reachable[mode + 1][n])
            if n > 0:
                reachable[mode][n].update(q + weight for q in reachable[mode + 1][n - 1])

    if target not in reachable[0][Nparticles]:
        return np.array([], dtype=dtype)

    states: list[int] = []

    def emit(mode: int, remaining: int, needed_lz2: int, state: int) -> None:
        if mode == Nmodes:
            if remaining == 0 and needed_lz2 == 0:
                states.append(state)
            return

        if needed_lz2 in reachable[mode + 1][remaining]:
            emit(mode + 1, remaining, needed_lz2, state)

        if remaining > 0:
            weight = weights[mode]
            if needed_lz2 - weight in reachable[mode + 1][remaining - 1]:
                bit_pos = Nmodes - mode - 1
                emit(
                    mode + 1,
                    remaining - 1,
                    needed_lz2 - weight,
                    state | (1 << bit_pos),
                )

    emit(0, Nparticles, target, 0)
    states = np.array(sorted(states), dtype=dtype)
    return np.ascontiguousarray(states)


@cfunc(op_sig_32, locals=dict(bit=uint32, occ=uint32, lower=uint32, parity=int32))
def _fermion_op_32(op_struct_ptr, op_str, site_ind, Nmodes, args):
    op_struct = carray(op_struct_ptr, 1)[0]
    site_ind = Nmodes - site_ind - 1
    bit = uint32(1) << site_ind
    occ = (op_struct.state >> site_ind) & uint32(1)
    parity = 0
    lower = op_struct.state & (bit - uint32(1))
    while lower:
        parity ^= int32(lower & uint32(1))
        lower >>= uint32(1)

    if op_str == 110:  # n
        op_struct.matrix_ele *= occ
    elif op_str == 43:  # +
        if occ:
            op_struct.matrix_ele = 0.0
        else:
            if parity:
                op_struct.matrix_ele *= -1.0
            op_struct.state ^= bit
    elif op_str == 45:  # -
        if not occ:
            op_struct.matrix_ele = 0.0
        else:
            if parity:
                op_struct.matrix_ele *= -1.0
            op_struct.state ^= bit
    elif op_str == 73:  # I
        pass
    else:
        op_struct.matrix_ele = 0.0
        return -1
    return 0


@cfunc(op_sig_64, locals=dict(bit=uint64, occ=uint64, lower=uint64, parity=int32))
def _fermion_op_64(op_struct_ptr, op_str, site_ind, Nmodes, args):
    op_struct = carray(op_struct_ptr, 1)[0]
    site_ind = Nmodes - site_ind - 1
    bit = uint64(1) << site_ind
    occ = (op_struct.state >> site_ind) & uint64(1)
    parity = 0
    lower = op_struct.state & (bit - uint64(1))
    while lower:
        parity ^= int32(lower & uint64(1))
        lower >>= uint64(1)

    if op_str == 110:
        op_struct.matrix_ele *= occ
    elif op_str == 43:
        if occ:
            op_struct.matrix_ele = 0.0
        else:
            if parity:
                op_struct.matrix_ele *= -1.0
            op_struct.state ^= bit
    elif op_str == 45:
        if not occ:
            op_struct.matrix_ele = 0.0
        else:
            if parity:
                op_struct.matrix_ele *= -1.0
            op_struct.state ^= bit
    elif op_str == 73:
        pass
    else:
        op_struct.matrix_ele = 0.0
        return -1
    return 0


@cfunc(next_state_sig_32)
def _next_precomputed_32(s, counter, Nmodes, args):
    return args[counter + uint32(1)]


@cfunc(next_state_sig_64)
def _next_precomputed_64(s, counter, Nmodes, args):
    return args[counter + uint64(1)]


@cfunc(count_particles_sig_32, locals=dict(x=uint32))
def _count_particles_32(s, p_number_ptr, args):
    x = s
    while x:
        p_number_ptr[0] += int32(x & uint32(1))
        x >>= uint32(1)


@cfunc(count_particles_sig_64, locals=dict(x=uint64))
def _count_particles_64(s, p_number_ptr, args):
    x = s
    while x:
        p_number_ptr[0] += int32(x & uint64(1))
        x >>= uint64(1)


@cfunc(map_sig_32, locals=dict(mode=int32, src=int32, dst=int32, out=uint32, occ=uint32, n_up=int32, n_down=int32))
def _flavour_flip_32(s, Nmodes, sign_ptr, args):
    L = Nmodes // 2
    out = uint32(0)
    n_up = 0
    n_down = 0
    for mode in range(Nmodes):
        src = Nmodes - mode - 1
        occ = (s >> src) & uint32(1)
        if occ:
            if mode < L:
                n_up += 1
            else:
                n_down += 1
            if mode < L:
                dst = Nmodes - (mode + L) - 1
            else:
                dst = Nmodes - (mode - L) - 1
            out |= uint32(1) << dst
    if (n_up * n_down) % 2:
        sign_ptr[0] = -sign_ptr[0]
    return out


@cfunc(map_sig_64, locals=dict(mode=int32, src=int32, dst=int32, out=uint64, occ=uint64, n_up=int32, n_down=int32))
def _flavour_flip_64(s, Nmodes, sign_ptr, args):
    L = Nmodes // 2
    out = uint64(0)
    n_up = 0
    n_down = 0
    for mode in range(Nmodes):
        src = Nmodes - mode - 1
        occ = (s >> src) & uint64(1)
        if occ:
            if mode < L:
                n_up += 1
            else:
                n_down += 1
            if mode < L:
                dst = Nmodes - (mode + L) - 1
            else:
                dst = Nmodes - (mode - L) - 1
            out |= uint64(1) << dst
    if (n_up * n_down) % 2:
        sign_ptr[0] = -sign_ptr[0]
    return out


def _make_pcon_dict(dtype: np.dtype, states: np.ndarray, Nparticles: int) -> dict:
    def get_s0_pcon(Nmodes: int, Nparticles: int) -> int:
        return int(states[0])

    def get_Ns_pcon(Nmodes: int, Nparticles: int) -> int:
        return int(states.size)

    if dtype == np.uint32:
        return {
            "Np": Nparticles,
            "next_state": _next_precomputed_32,
            "next_state_args": states,
            "get_Ns_pcon": get_Ns_pcon,
            "get_s0_pcon": get_s0_pcon,
            "count_particles": _count_particles_32,
            "count_particles_args": np.array([], dtype=np.int32),
            "n_sectors": 1,
        }
    return {
        "Np": Nparticles,
        "next_state": _next_precomputed_64,
        "next_state_args": states,
        "get_Ns_pcon": get_Ns_pcon,
        "get_s0_pcon": get_s0_pcon,
        "count_particles": _count_particles_64,
        "count_particles_args": np.array([], dtype=np.int32),
        "n_sectors": 1,
    }


def make_lz_user_basis(
    L: int,
    Nparticles: int,
    lz: int,
    nflav: int = 2,
    flavour_eigval: Optional[int] = None,
    parallel: bool = True,
    Ns_block_est: Optional[int] = None,
):
    """Construct a QuSpin ``user_basis`` restricted to fixed total particle number
    and fixed Fuzzy-sphere ``Lz``.

    If ``flavour_eigval`` is ``1`` or ``-1``, the basis is additionally reduced by
    the spin-flavour permutation symmetry.
    """
    if nflav not in (1, 2):
        raise ValueError(f"Unsupported nflav={nflav}.")
    if flavour_eigval not in (None, 1, -1):
        raise ValueError("`flavour_eigval` must be None, 1, or -1.")
    if flavour_eigval is not None and nflav != 2:
        raise ValueError("Flavour permutation is only defined for `nflav=2`.")

    Nmodes = nflav * L
    dtype = _basis_dtype(Nmodes)
    op = _fermion_op_32 if dtype == np.uint32 else _fermion_op_64
    states = _generate_lz_basis_states(L, Nparticles, lz, nflav, dtype)
    if states.size == 0:
        raise ValueError(
            f"The requested Lz sector is empty: L={L}, N={Nparticles}, "
            f"lz={lz}, nflav={nflav}."
        )

    blocks = {}
    if flavour_eigval is not None:
        q = 0 if flavour_eigval == 1 else 1
        flip = _flavour_flip_32 if dtype == np.uint32 else _flavour_flip_64
        blocks["flavour_block"] = (flip, 2, q, np.array([], dtype=dtype))

    if Ns_block_est is None:
        if flavour_eigval is not None:
            Ns_block_est = max(1, states.size // 2 + 2)
        else:
            Ns_block_est = max(1, states.size)

    return user_basis(
        dtype,
        Nmodes,
        dict(op=op, op_args=np.array([], dtype=dtype)),
        sps=2,
        pcon_dict=_make_pcon_dict(dtype, states, Nparticles),
        allowed_ops=set("+-nI"),
        parallel=parallel,
        Ns_block_est=int(Ns_block_est),
        **blocks,
    )


class LzUserBasisSymmetry(FuzzySphereSymmetry):
    """Quantax symmetry adapter whose QuSpin basis is an Lz-filtered user_basis."""

    def __init__(
        self,
        lz: int,
        eigval: int = 0,
        nflav: Optional[int] = None,
        parallel: bool = True,
        Ns_block_est: Optional[int] = None,
    ):
        if eigval not in (0, 1, -1):
            raise ValueError("`eigval` must be 0, 1, or -1.")
        sites = get_sites()
        if sites.particle_type not in (
            PARTICLE_TYPE.spinless_fermion,
            PARTICLE_TYPE.spinful_fermion,
        ):
            raise ValueError("Lz user bases are implemented only for fermions.")

        if eigval == 0 and nflav is None:
            nflav = 2 if sites.particle_type == PARTICLE_TYPE.spinful_fermion else 1
        elif eigval in (1, -1):
            if nflav not in (None, 2):
                raise ValueError("Flavour permutation requires `nflav=2`.")
            nflav = 2

        if sites.particle_type == PARTICLE_TYPE.spinless_fermion and nflav != 1:
            raise ValueError("Spinless fermions require `nflav=1`.")
        if sites.particle_type == PARTICLE_TYPE.spinful_fermion and nflav != 2:
            raise ValueError("Spinful fermions require `nflav=2`.")

        if eigval == 0:
            generator = None
            sector = 0
        else:
            if sites.particle_type != PARTICLE_TYPE.spinful_fermion:
                raise ValueError("Flavour permutation requires spinful fermions.")
            sector = 0 if eigval == 1 else 1
            generator = np.concatenate([np.arange(sites.Nsites, 2 * sites.Nsites), np.arange(sites.Nsites)])

        self._lz = int(lz)
        self._nflav = int(nflav)
        self._eigval = int(eigval)
        self._parallel = parallel
        self._Ns_block_est = Ns_block_est
        super().__init__(generator=generator, sector=sector)

    @property
    def lz(self) -> int:
        return self._lz

    @property
    def eigval(self) -> int:
        return self._eigval

    @property
    def basis(self):
        if self._basis is None:
            self._basis = make_lz_user_basis(
                self.Nsites,
                _total_particles(self.Nparticles),
                self._lz,
                nflav=self._nflav,
                flavour_eigval=None if self._eigval == 0 else self._eigval,
                parallel=self._parallel,
                Ns_block_est=self._Ns_block_est,
            )
        return self._basis

    def basis_make(self) -> None:
        if not self._is_basis_made:
            self.basis.make()
        self._is_basis_made = True
