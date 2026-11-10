from functools import partial
from typing import Sequence
import numpy as np
import sympy.physics.wigner as spw
import equinox as eqx
import jax
import jax.numpy as jnp
import quantax as qtx
from quantax.global_defs import get_default_dtype, get_sites, get_subkeys
from quantax.operator.operator import OpTerm, OpTermJAX, _apply_site_operator, _get_Olocx
from quantax.operator.update_mode_filters import DIAGONAL_OPS
from quantax.sampler import Samples
from quantax.state import State
from quantax.utils import chunk_map, get_replicated_sharding, to_distributed_array


def get_int_matrix(nm : int, ps_pot : Sequence[int]):
    """
        GetIntMatrix(nm :: Int64, ps_pot :: Vector{<:Number}) :: Array{ComplexF64, 3}

    Gives the interaction matrix ``U_{m_1,m_2,m_3,m_4}`` from the pseudopotentials.
        
    # Argument

    * `nm :: Int64` is the number of orbitals.
    * `ps_pot :: Vector{<:Number}` is the vector of non-zero pseudopotentials.

    # Output
    * A `nm`×`nm`×`nm` array giving the interaction matrix ``U_{m_1,m_2,m_3,-m_1-m_2-m_3}``.
    """
    int_el = np.zeros((nm, nm, nm))
    s = 0.5 * (nm - 1)
    for m1 in range(nm): 
        m1r = m1 - s
        for m2 in range(nm): 
            m2r = m2 - s
            for m3 in range(nm): 
                m3r = m3 - s
                m4 = m1 + m2 - m3 
                if m4 < 0 or m4 >= nm:
                    continue
                m4r = m4 - s
                for l in range(len(ps_pot)):
                    if (abs(m1r + m2r) > nm - l - 1) or (abs(m3r + m4r) > nm - l - 1):
                        break
                    int_el[m1, m2, m3] += ps_pot[l] * (2 * nm - 2 * l - 1) * \
                        spw.wigner_3j(s, s, nm - l - 1, m1r, m2r, -m1r - m2r) * \
                            spw.wigner_3j(s, s, nm - l - 1, m4r, m3r, -m3r - m4r)
    return int_el



def GetSpinlessDenIntTerms(nm : int, ps_pot : Sequence[int]) -> qtx.operator.Operator:
    """
        GetDenIntTerms(nm :: Int64, nf :: Int64[, ps_pot :: Vector{<:Number}][, mat_a :: Matrix{<:Number}[, mat_b :: Matrix{<:Number}]][ ; m_kept :: Vector{Int64}]) :: Terms

    Return the normal-ordered density-density term in the Hamiltonian 
    ```math 
    ∑_{\\{m_i,f_i\\}}U_{m_1m_2m_3m_4}M^A_{f_1f_4}M^B_{f_2f_3}c^{†}_{m_1f_1}c^{†}_{m_2f_2}c_{m_3f_3}c_{m_4f_4}.
    ```

    # Arguments 

    * `nm :: Int64` is the number of orbitals.
    * `ps_pot :: Vector{<:Number}` is a list of numbers specifying the pseudopotentials for the interacting matrix ``U_{m_1m_2m_3m_4}``. Facultative, `[1.0]` by default. 
    """
    H = 0
    interaction = get_int_matrix(nm, ps_pot)
    
    for m1 in range(nm):
        for m2 in range(nm):
            if m1 == m2:
                continue
            for m3 in range(nm):
                m4 = m1 + m2 - m3 
                if m4 < 0 or m4 >= nm or m3 == m4: 
                    continue
                val = interaction[m1, m2, m3]
                if (abs(val) < 1E-12):
                    continue  
                H += val * (qtx.operator.create(m1) @ qtx.operator.create(m2) @ qtx.operator.annihilate(m3) @ qtx.operator.annihilate(m4))
    return H

def qtx_create(f,m):
    if f == 0:
        return qtx.operator.create_u(m)
    else:
        return qtx.operator.create_d(m)
    

def qtx_annihilate(f,m):
    if f == 0:
        return qtx.operator.annihilate_u(m)
    else:
        return qtx.operator.annihilate_d(m)


def qtx_number(f,m):
    if f == 0:
        return qtx.operator.number_u(m)
    else:
        return qtx.operator.number_d(m)

def GetSpinfulDenIntTerms(nm : int, ps_pot : Sequence[int], mat_a : Sequence[int], mat_b : Sequence[int],
                          cutoff : float = 1e-15, return_dropped : bool = False) -> qtx.operator.Operator:
    """
        GetDenIntTerms(nm :: Int64, nf :: Int64[, ps_pot :: Vector{<:Number}][, mat_a :: Matrix{<:Number}[, mat_b :: Matrix{<:Number}]][ ; m_kept :: Vector{Int64}]) :: Terms

    Return the normal-ordered density-density term in the Hamiltonian
    ```math
    ∑_{\\{m_i,f_i\\}}U_{m_1m_2m_3m_4}M^A_{f_1f_4}M^B_{f_2f_3}c^{†}_{m_1f_1}c^{†}_{m_2f_2}c_{m_3f_3}c_{m_4f_4}.
    ```

    # Arguments

    * `nm :: Int64` is the number of orbitals.
    * `ps_pot :: Vector{<:Number}` is a list of numbers specifying the pseudopotentials for the interacting matrix ``U_{m_1m_2m_3m_4}``. Facultative, `[1.0]` by default.
    * `cutoff` drops the hopping terms (those changing the configuration, the cost of a local energy) with
      |coefficient| below it; the diagonal density terms are free to evaluate and always kept down to 1e-15.
    * `return_dropped` also returns the sum of the dropped |coefficients|: a bound on the norm of the dropped part,
      so on the shift of every eigenvalue and expectation value.
    """
    H = 0
    dropped = 0.0
    interaction = get_int_matrix(nm, ps_pot)

    for o1 in range(2*nm):
        m1 = o1//2
        f1 = o1%2
        for o2 in range(2*nm):
            m2 = o2//2
            f2 = o2%2
            if o1 == o2:
                continue
            for o3 in range(2*nm):
                m3 = o3//2
                f3 = o3%2
                if np.abs(mat_b[f2, f3]) < 1e-13:
                     continue
                m4 = m1 + m2 - m3 
                if m4 < 0 or m4 >= nm: 
                    continue
                for f4 in range(2):
                    if np.abs(mat_a[f1, f4]) < 1e-13:
                        continue
                    o4 = m4 * 2 + f4
                    #print(o1, o2, o3, o4)
                    if o3 == o4:
                        continue
                    val = mat_a[f1, f4] * mat_b[f2, f3] * interaction[m1, m2, m3]
                    #print(val)
                    if (np.abs(val) < 1E-15):
                        continue
                    # Emit true density operators when the term is diagonal so Quantax
                    # keeps it in the cheap `apply_diag` path instead of generating a
                    # connected configuration equal to s.  With Lz conservation
                    # (m4 = m1+m2-m3) the direct case is ~L^2 of the terms.
                    if (m1, f1) == (m4, f4) and (m2, f2) == (m3, f3):
                        # c+_1 c+_2 c_2 c_1 = n_1 n_2
                        H += val * (qtx_number(f1, m1) @ qtx_number(f2, m2))
                    elif (m1, f1) == (m3, f3) and (m2, f2) == (m4, f4):
                        # c+_1 c+_2 c_1 c_2 = -n_1 n_2
                        H += -val * (qtx_number(f1, m1) @ qtx_number(f2, m2))
                    elif np.abs(val) < cutoff:
                        dropped += float(np.abs(val))
                    else:
                        H += val * (qtx_create(f1, m1) @ qtx_create(f2, m2) @ qtx_annihilate(f3, m3) @ qtx_annihilate(f4, m4))
    return (H, dropped) if return_dropped else H


def GetSpinfulPolTerms(nm : int, mat : Sequence[int]) -> qtx.operator.Operator:
    H = 0
    for o1 in range(2*nm):
        m1 = o1//2
        f1 = o1%2
        for f2 in range(2):
            if np.abs(mat[f1, f2]) < 1E-13:
                continue
            H += mat[f1,f2] * (qtx_create(f1, m1) @ qtx_annihilate(f2, m1))

    return H


def GetSpinfulMultipoleTerms(
    nm: int,
    ell: int,
    m: int,
    mat: Sequence[int],
    normalize: bool = False,
) -> qtx.operator.Operator:
    r"""
    Return the spinful one-body multipole operator

        sum_{m1,m2,f1,f2} <S,m1|T_{ell,m}|S,m2> M_{f1,f2}
            c^\dagger_{m1,f1} c_{m2,f2}.

    The orbital matrix elements are taken from the standard rank-``ell`` tensor
    structure on the sphere. For variational warm starts, the overall normalization is
    usually irrelevant; only the relative coefficients matter.
    """
    if abs(m) > ell:
        raise ValueError(f"Need |m| <= ell, got ell={ell}, m={m}.")

    H = 0
    s = 0.5 * (nm - 1)
    prefactor = np.sqrt(2 * ell + 1)
    if normalize:
        prefactor /= np.sqrt(nm)

    for m1 in range(nm):
        m1r = m1 - s
        for m2 in range(nm):
            m2r = m2 - s
            coeff = prefactor * ((-1) ** int(round(s - m1r))) * spw.wigner_3j(
                s, ell, s, -m1r, m, m2r
            )
            coeff = float(coeff)
            if abs(coeff) < 1e-13:
                continue

            for f1 in range(2):
                for f2 in range(2):
                    if abs(mat[f1, f2]) < 1e-13:
                        continue
                    # Emit true density operators when the term is diagonal so Quantax
                    # can keep it in the cheap `apply_diag` path.
                    if m1 == m2 and f1 == f2:
                        if f1 == 0:
                            H += coeff * mat[f1, f2] * qtx.operator.number_u(m1)
                        else:
                            H += coeff * mat[f1, f2] * qtx.operator.number_d(m1)
                    else:
                        H += coeff * mat[f1, f2] * (
                            qtx_create(f1, m1) @ qtx_annihilate(f2, m2)
                        )
    return H


def GetGlobalSpinFlip(nm : int) -> qtx.operator.Operator:
    H = (qtx_create(0, 0) @ qtx_annihilate(1, 0) + qtx_create(1, 0) @ qtx_annihilate(0, 0))
    for m1 in range(1,nm):
        H = H @ (qtx_create(0, m1) @ qtx_annihilate(1, m1) + qtx_create(1, m1) @ qtx_annihilate(0, m1))
    return H 


def GetLzTerms(nm : int, nflav: int) -> qtx.operator.Operator:
    Lz = 0
    s = (nm - 1) / 2
    for o in range(nm * nflav):
        m = o//nflav
        f = o%nflav
        if nflav == 1:
            Lz += (m-s) * qtx.operator.number(m)
        else:
            Lz += (m-s) * (qtx_create(f, m) @ qtx_annihilate(f, m))

    return Lz


def GetLpTerms(nm : int, nflav: int) -> qtx.operator.Operator:
    Lp = 0
    for o in range(nflav, nm * nflav):
        m = o//nflav
        f = o%nflav
        if nflav == 1:
            Lp += np.sqrt(m * (nm - m)) * (qtx.operator.create(m) @ qtx.operator.annihilate(m-1))
        else:
            Lp += np.sqrt(m * (nm - m)) * (qtx_create(f, m) @ qtx_annihilate(f, m-1))

    return Lp


def GetL2Terms(nm : int, nflav: int) -> qtx.operator.Operator:
    Lz = GetLzTerms(nm, nflav)
    Lp = GetLpTerms(nm, nflav)
    Lm = Lp.H

    return Lz @ Lz - Lz + Lp @ Lm


def GetIdTerms(nm : int, nflav: int) -> qtx.operator.Operator:
    Op = 0
    for o in range(nm * nflav):
        m = o//nflav
        f = o%2
        if nflav == 1:
            Op += qtx.operator.number(m)
        else:
            Op += (qtx_create(f, m) @ qtx_annihilate(f, m))

    return Op


def _flip_set(opstr: str, indices) -> tuple:
    """The sites a term flips whenever it does not annihilate the configuration: those carrying an odd number of
    off-diagonal characters (+, -, x, y). Terms with the same flip set connect s to the same s'."""
    count = {}
    for op, i in zip(opstr, indices):
        if op not in DIAGONAL_OPS:
            count[int(i)] = count.get(int(i), 0) + 1
    return tuple(sorted(i for i, c in count.items() if c % 2))


@eqx.filter_jit
@partial(eqx.filter_vmap, in_axes=(0, None))
def _term_factors(s: jax.Array, op_terms: tuple) -> jax.Array:
    """Quantax's `_apply_off_diag` without the connected configurations: the factor every unit-strength term picks
    up on s (fermion sign, the 1/2 of x and y, ...), NaN where it annihilates s."""
    apply_fn = jax.vmap(_apply_site_operator, in_axes=(0, None, 0, 0))
    factors = []
    for op_term in op_terms:
        strength = op_term.strength
        s_conn = jnp.repeat(s[None, :], strength.size, axis=0)
        for op, idx in zip(reversed(op_term.opstr), op_term.indices.T[::-1]):
            s_conn, strength = apply_fn(s_conn, op, strength, idx)
        factors.append(strength)
    return jnp.concatenate(factors)


@partial(jax.jit, static_argnums=3)
def _group_weights(factors: jax.Array, term_group: jax.Array, term_J: jax.Array, ngroups: int) -> jax.Array:
    """<s'|O_c|s> of every flip-set group, (nsamples, ngroups, ncomp): the terms' factors times their strengths,
    summed over the terms of a group. A factor of 0 is an `n` that vanishes on s, NaN a term that annihilates s."""
    valid = jnp.isfinite(factors) & (factors != 0)
    w = jnp.where(valid[..., None], factors[..., None] * term_J, 0)
    W = jnp.zeros((factors.shape[0], ngroups, term_J.shape[1]), w.dtype)
    return W.at[:, term_group].add(w)


@jax.jit
def _thin(W: jax.Array, key: jax.Array, tail: float) -> jax.Array:
    """Keep every connection with probability p = min(1, max_c |W| / tail), weighted by 1/p: unbiased, exact above
    `tail`, and the minimum-variance choice for a given expected number of connections below it."""
    p = jnp.minimum(1.0, jnp.max(jnp.abs(W), axis=-1) / tail)
    keep = jax.random.uniform(key, p.shape) < p
    return jnp.where(keep[..., None], W / jnp.where(keep, p, 1.0)[..., None], 0)


@jax.jit
def _tail_g(W: jax.Array, tail: float) -> jax.Array:
    r"""The added variance the tail sampling of `_thin` puts on each sample, without the amplitudes:
    :math:`g_{cd}(s) = \sum_{s'} (1/p_{ss'} - 1)\, W^*_c W_d`, ``(nsamples, ncomp, ncomp)``, from the unthinned W.
    Its average over :math:`s \sim |\psi|^2` is exactly :math:`E_s \mathrm{Cov}_\xi` of the sampled components
    (the ratios cancel on relabelling, :math:`|O_{ss'}|` and p being symmetric), so a component combination
    :math:`a` gains :math:`\langle a^\dagger g\, a\rangle` of variance."""
    p = jnp.minimum(1.0, jnp.max(jnp.abs(W), axis=-1) / tail)
    f = jnp.where(p > 0, 1 / jnp.where(p > 0, p, 1.0) - 1, 0.0)
    return jnp.einsum("sg,sgc,sgd->scd", f, W.conj(), W, precision=jax.lax.Precision.HIGHEST)


@partial(jax.jit, static_argnums=1)
def _live_size(live: jax.Array, multiple: int | None) -> jax.Array:
    """The largest number of live connections on one device, rounded up to `multiple` (the state's forward chunk,
    as Quantax's `_get_conn_size` does, so that the compacted batch takes few distinct shapes)."""
    size = jnp.max(jnp.sum(live.reshape(jax.device_count(), -1), axis=1))
    if multiple is not None:
        size = (size + multiple - 1) // multiple * multiple
    return size


@partial(jax.jit, static_argnums=4)
def _compact(spins: jax.Array, W: jax.Array, live: jax.Array, flips: jax.Array, size: int):
    """The live connections, laid out as Quantax's `_get_conn` does: (segment, s', <s'|O_c|s>). s' is s with the
    group's flip set flipped, built for the live connections only; padding repeats s with zero weight."""
    ndevices = jax.device_count()
    nsamples, ngroups, ncomp = W.shape
    Nmodes = spins.shape[-1]

    def device_conn(spins, W, live):
        segment, group = jnp.nonzero(live, size=size, fill_value=-1)
        pad = segment == -1
        s_conn = spins[segment] * jnp.where(pad[:, None], 1, flips[group]).astype(spins.dtype)
        W = jnp.where(pad[:, None], 0, W[segment, group])
        return segment, s_conn, W

    segment, s_conn, W = jax.vmap(device_conn)(
        spins.reshape(ndevices, -1, Nmodes), W.reshape(ndevices, -1, ngroups, ncomp),
        live.reshape(ndevices, -1, ngroups))
    segment += jnp.arange(ndevices)[:, None] * (nsamples // ndevices)
    return segment.flatten(), s_conn.reshape(-1, Nmodes), W.reshape(-1, ncomp)


def _bit_codes(masks: np.ndarray) -> np.ndarray:
    """One integer per row of a boolean (n, Nmodes) array: bit m set where it is True (row bytes beyond 64 modes)."""
    masks = np.asarray(masks, bool)
    if masks.shape[1] <= 64:
        return (masks.astype(np.uint64) << np.arange(masks.shape[1], dtype=np.uint64)).sum(axis=1, dtype=np.uint64)
    packed = np.packbits(masks, axis=1)
    return np.array([row.tobytes() for row in packed], dtype=object)


def _by_opstr(keys) -> dict:
    """{opstr: [indices, ...]} of (opstr, indices) keys."""
    out = {}
    for opstr, inds in keys:
        out.setdefault(opstr, []).append(inds)
    return out


class PooledOperator:
    r"""
    Several operators evaluated together, so that every distinct connected configuration costs one forward pass.

    Quantax's ``Operator.__add__`` concatenates term lists and ``Oloc`` evaluates one connected configuration per
    term, so :math:`H + \lambda L^-L^+` pays twice for the moves the two share, a term like
    :math:`c^\dagger_a c_b c^\dagger_b c_a` pays a forward pass on :math:`s` itself, and every separate ``Oloc`` on
    the same samples enumerates the terms again. Here the off-diagonal terms of all `operators` (the components) are
    grouped by their flip set: a term that does not annihilate :math:`s` turns it into :math:`s` with that set
    flipped, so a group is one connected configuration, whatever its terms and whatever :math:`s`. Its matrix element
    is kept per component. The empty set (terms that come back to :math:`s`) needs no forward pass.

    ``Oloc(states, samples)`` then returns every component for every state from one enumeration per chunk of
    samples and one forward pass of each state per distinct connection: exact, the same numbers as the separate
    ``Oloc`` calls up to summation order.

    :param operators: the components, e.g. ``[H, L^- L^+]``.
    :param tail: sample the small matrix elements instead of evaluating them all: a connection is kept with
        probability :math:`p = \min(1, \max_c |\langle s'|O_c|s\rangle| / \mathrm{tail})` and weighted by
        :math:`1/p`. Unbiased, so exact on average, at an extra variance below
        :math:`\mathrm{tail}\sum_{s'} |O_{ss'}| |\psi(s')/\psi(s)|^2` over the sampled connections. 0 (the default)
        evaluates every connection.
    """

    def __init__(self, operators: Sequence[qtx.operator.Operator], tail: float = 0.0):
        self.operators = list(operators)
        self.tail = float(tail)
        ncomp = len(self.operators)
        terms, diag = {}, [{} for _ in range(ncomp)]
        for c, op in enumerate(self.operators):
            for term in op.op_list:
                for J, inds in zip(term.strength, term.indices):
                    key = (term.opstr, tuple(int(i) for i in inds))
                    if all(ch in DIAGONAL_OPS for ch in term.opstr):
                        diag[c][key] = diag[c].get(key, 0) + J
                    else:
                        terms.setdefault(key, np.zeros(ncomp, complex))[c] += J

        groups, by_opstr = {}, {}
        for (opstr, inds), J in terms.items():
            if np.any(J != 0):
                g = groups.setdefault(_flip_set(opstr, inds), len(groups))
                by_opstr.setdefault(opstr, []).append((inds, J, g))
        self._by_opstr = by_opstr
        self._groups = list(groups)
        self._empty = groups.get(())
        self._diag = [qtx.operator.Operator([OpTerm(opstr, [d[(opstr, i)] for i in ii], [list(i) for i in ii])
                                             for opstr, ii in _by_opstr(d).items()]) if d else None for d in diag]
        self._jax = None

    @property
    def ncomp(self) -> int:
        return len(self.operators)

    @property
    def nterms(self) -> int:
        """Distinct off-diagonal terms, every component's pooled."""
        return sum(len(t) for t in self._by_opstr.values())

    @property
    def ngroups(self) -> int:
        """Distinct moves (flip sets) among them, the empty one included."""
        return len(self._groups)

    def _arrays(self) -> dict:
        """The jax arrays, built on first use (under the dtype and sites then set, as Quantax does)."""
        if self._jax is None:
            sharding = get_replicated_sharding()
            dtype = get_default_dtype()
            op_terms, J, group = [], [], []
            for opstr, entries in self._by_opstr.items():
                ones = jnp.ones(len(entries), dtype, device=sharding)
                indices = jnp.asarray([e[0] for e in entries], jnp.int32, device=sharding)
                op_terms.append(OpTermJAX(opstr, ones, indices))
                J.extend(e[1] for e in entries)
                group.extend(e[2] for e in entries)
            J = np.asarray(J, complex).reshape(-1, self.ncomp)
            J = J if jnp.issubdtype(dtype, jnp.complexfloating) else J.real
            flips = np.ones((self.ngroups, get_sites().Nmodes), np.int8)
            for g, sites in enumerate(self._groups):
                flips[g, list(sites)] = -1
            self._jax = dict(op_terms=tuple(op_terms), J=jnp.asarray(J, dtype, device=sharding),
                             group=jnp.asarray(group, jnp.int32, device=sharding),
                             flips=jnp.asarray(flips, device=sharding))
        return self._jax

    def _weights(self, spins: jax.Array, key=None, with_g: bool = False):
        """(W, live, ratio-one part) of a batch: the group weights, with the tail sampled when `key` is given;
        with `with_g` also the per-sample `_tail_g` of that sampling (zeros when nothing is sampled)."""
        a = self._arrays()
        if a["op_terms"]:
            W = _group_weights(_term_factors(spins, a["op_terms"]), a["group"], a["J"], self.ngroups)
        else:
            W = jnp.zeros((spins.shape[0], 0, self.ncomp), get_default_dtype())
        extra = jnp.zeros((spins.shape[0], self.ncomp), W.dtype)
        if self._empty is not None:
            extra = W[:, self._empty]
            W = W.at[:, self._empty].set(0)
        g = jnp.zeros((spins.shape[0], self.ncomp, self.ncomp), W.dtype) if with_g else None
        if key is not None and self.tail > 0 and W.shape[1]:
            if with_g:
                g = _tail_g(W, self.tail)
            W = _thin(W, key, self.tail)
        live = jnp.any(jnp.abs(W) > 1e-8, axis=-1)      # Quantax's isclose(H_conn, 0)
        return (W, live, extra, g) if with_g else (W, live, extra)

    def _diagonal(self, spins: jax.Array, dtype) -> jax.Array:
        cols = [jnp.zeros(spins.shape[0], dtype) if d is None else d.apply_diag(spins).astype(dtype)
                for d in self._diag]
        return jnp.stack(cols, axis=-1)

    def _group_codes(self):
        """The flip sets as sorted integer codes (bit m set where mode m flips) and the group of each."""
        if getattr(self, "_codes", None) is None:
            Nmodes = get_sites().Nmodes
            masks = np.zeros((self.ngroups, Nmodes), bool)
            for g, sites in enumerate(self._groups):
                masks[g, list(sites)] = True
            codes = _bit_codes(masks)
            order = np.argsort(codes, kind="stable")
            self._codes = (codes[order], order)
        return self._codes

    def tail_g_between(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        r"""
        :math:`(1/p - 1)\, W^*_c W_d` of the connection :math:`a \to b` for each pair of configurations, zero
        where b is not one move from a (or a = b) or nothing is sampled: ``(npairs, ncomp, ncomp)``, numpy.

        `_tail_g` sums this over every connection of a. A trace over a tuple of NES slots does not see the
        connections of slot i that land on another slot j, where its determinant ratio vanishes; the trace's added
        variance is therefore :math:`\sum_i g(x_i) - \sum_{i\neq j} c(x_i, x_j)`. Only the pairs that are one move
        apart are evaluated (no forward pass), so this is nearly free when such pairs are rare.
        """
        a, b = np.asarray(a), np.asarray(b)
        nc = self.ncomp
        out = np.zeros((a.shape[0], nc, nc))
        if self.tail <= 0 or a.shape[0] == 0:
            return out
        codes, order = self._group_codes()
        diff = a != b
        pair = _bit_codes(diff)
        pos = np.minimum(np.searchsorted(codes, pair), len(codes) - 1)
        hit = np.flatnonzero((codes[pos] == pair) & diff.any(axis=1))
        if hit.size == 0:
            return out
        groups = order[pos[hit]]
        # padded to a power of 2, so the jitted enumeration sees few distinct shapes
        size = 1 << int(np.ceil(np.log2(hit.size)))
        rows = np.concatenate([hit, np.full(size - hit.size, hit[0])])
        W = np.asarray(self._weights(jnp.asarray(a[rows]))[0])[np.arange(hit.size), groups]
        p = np.minimum(1.0, np.max(np.abs(W), axis=-1) / self.tail)
        f = np.where(p > 0, 1 / np.where(p > 0, p, 1.0) - 1, 0.0)
        c = f[:, None, None] * np.einsum("kc,kd->kcd", W.conj(), W)
        out = out.astype(np.result_type(out, c))
        out[hit] = c
        return out

    def connection_count(self, spins: jax.Array, sampled: bool = False) -> jax.Array:
        """Forward passes per sample: the distinct connected configurations s' != s; with `sampled`, those kept by
        one draw of the tail sampling."""
        spins = to_distributed_array(jnp.asarray(spins))
        _, live, _ = self._weights(spins, get_subkeys() if sampled else None)
        return jnp.sum(live, axis=1)

    def Oloc(self, states, samples, psi=None, tail_variance: bool = False):
        r"""
        :math:`\sum_{s'} \langle s|O_c|s'\rangle \psi(s')/\psi(s)` of every component and state.

        :param states: a state, or a sequence of states sharing the samples (the members of an NES set).
        :param samples: `Samples` or configurations.
        :param psi: the amplitudes on the samples, one per state (a list for a sequence of states); computed where
            missing. For a single state it defaults to ``samples.psi``.
        :param tail_variance: also return the per-sample `_tail_g` of this call's tail sampling,
            ``(nsamples, ncomp, ncomp)``: from the matrix elements already enumerated, no forward pass. The draw
            is shared by the states, so every state's sampled values carry the same g. Zeros without a tail.
        :return: ``(nsamples, ncomp)`` for a single state, ``(nsamples, nstates, ncomp)`` for a sequence; with
            `tail_variance` the pair ``(that, g)``.
        """
        single = isinstance(states, State)
        states = [states] if single else list(states)
        if not isinstance(samples, Samples):
            samples = Samples(to_distributed_array(jnp.asarray(samples)))
        spins = samples.spins
        if psi is None:
            psi = [samples.psi if single else None] * len(states)
        elif single:
            psi = [psi]
        psi = [state(spins) if p is None else p for state, p in zip(states, psi)]

        nc, ns = self.ncomp, len(states)
        if any(state.use_ref for state in states):
            # reference updates need Quantax's per-term update modes: no pooling, every operator on its own (and
            # no tail sampling)
            out = jnp.stack([jnp.stack([op.Oloc(state, Samples(spins, p)) for op in self.operators], axis=-1)
                             for state, p in zip(states, psi)], axis=1)
            out = out[:, 0] if single else out
            return (out, jnp.zeros((spins.shape[0], nc, nc), out.dtype)) if tail_variance else out

        forward_chunk = getattr(states[0], "forward_chunk", None)
        chunk = getattr(states[0], "ref_chunk", forward_chunk)
        sampled = self.tail > 0
        nchunks = 1 if chunk is None else -(-spins.shape[0] // chunk)
        keys = iter(jax.random.split(get_subkeys(), nchunks)) if sampled else None

        def fn(spins, psi):
            weights = self._weights(spins, next(keys) if sampled else None, with_g=tail_variance)
            W, live, extra = weights[:3]
            size = int(_live_size(live, forward_chunk).item())
            if size > 0:
                segment, s_conn, W_conn = _compact(spins, W, live, self._arrays()["flips"], size)
            out = []
            for state, p in zip(states, psi):
                if size == 0:
                    out.append(jnp.zeros((spins.shape[0], self.ncomp), W.dtype))
                    continue
                psi_conn = state(s_conn)
                out.append(jnp.stack([_get_Olocx(p, segment, psi_conn, W_conn[:, c]) for c in range(self.ncomp)],
                                     axis=-1))
            out = jnp.stack(out, axis=1)
            out = out + (self._diagonal(spins, out.dtype) + extra.astype(out.dtype))[:, None]
            if tail_variance:       # one array per chunk, as `chunk_map` stacks it: g packed after the values
                n = out.shape[0]
                out = jnp.concatenate([out.reshape(n, -1), weights[3].reshape(n, -1).astype(out.dtype)], axis=1)
            return out

        out = chunk_map(fn, chunk_size=chunk)(spins, psi)
        if tail_variance:
            n = out.shape[0]
            g = out[:, ns * nc:].reshape(n, nc, nc)
            out = out[:, :ns * nc].reshape(n, ns, nc)
        out = out[:, 0] if single else out
        return (out, g) if tail_variance else out
