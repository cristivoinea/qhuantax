r"""Backflow models closed under the global flavour rotation :math:`R(\theta)`.

:math:`R(\theta)=\exp\big(\frac\theta2\sum_m(c^\dagger_{\downarrow m}c_{\uparrow m}-\mathrm{h.c.})\big)`
only reassigns the flavour of singly occupied orbitals, so it preserves the orbital occupations
:math:`n`. When the backflow features depend on a configuration only through :math:`n`
(*charge-only*), every configuration that :math:`R` mixes sees the same orbital matrix
:math:`M(n)=U_0+X(n)W^T`, and :math:`R` acts on the state by rotating the rows of :math:`M`:

.. math::

    (R\psi)(s) = \det\big([R_1 M(n)]_{\mathrm{occ}(s)}\big),

with :math:`R_1` mixing the (up :math:`m`, down :math:`m`) rows by
:math:`\begin{pmatrix}c&-s\\s&c\end{pmatrix}`, :math:`c,s=\cos,\sin\frac\theta2`
(``quantumhall_excitons.rotation_orbitals``). The rotation is one extra :math:`2\times2` mixing per
orbital and is exact; ``theta`` is a leaf, so it can be fixed (mask its gradient) or trained.

With ``charge_only=False`` the network sees the flavours and the same formula is just a
flavour-aware backflow written in a rotated frame: no less expressive than ``DetBackflow``
(at ``theta=0`` it is identical to it), but no longer exactly
:math:`R(\theta)` of the ``theta=0`` state. ``untie`` switches a charge-only model over without
changing a single amplitude, which is the transfer from the exactly rotated :math:`h=0` reference to a
model trained at :math:`h>0`.

Charge-only is implemented on the input: the network is fed the configuration with every singly
occupied orbital put in the up flavour, so ``Embedding`` row 2 (down only) is never read and the
tree is that of the ordinary ``Transformer``. ``untie`` copies row 1 (up only) into row 2.

Limitation of a single term, from ``docs``/the k=2 tests: on the configurations without charge
fluctuations (one electron per orbital, which carry ~90% of the :math:`h=0`, :math:`k=2` state) a
charge-only model is one *fixed* determinant, which gives amplitudes :math:`T(p,p')` of rank
:math:`\le4` there; ``RotatedMultiDetBackflow`` adds terms to lift this.
"""

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from quantax.global_defs import get_sites, get_subkeys
from quantax.nn import fermion_idx, fermion_inverse_sign, lecun_normal
from quantax.utils import LogArray, ScaleArray

MAX_EXPONENT = 60.0         # largest backflow feature e^60 ~ 1e26, well inside float32


def charge_only_spins(s: jax.Array) -> jax.Array:
    """The configuration with the flavour of every singly occupied orbital set to up: a function of
    the orbital occupations alone. Quantax layout, up modes then down modes, :math:`\\pm1`."""
    nm = s.shape[-1] // 2
    up, dn = s[..., :nm] > 0, s[..., nm:] > 0
    occ = jnp.concatenate([up | dn, up & dn], axis=-1)
    return jnp.where(occ, 1, -1).astype(s.dtype)


def rotate_rows(M: jax.Array, theta: jax.Array) -> jax.Array:
    r""":math:`R_1(\theta)M` for an orbital matrix with up rows first: up row
    :math:`\to c\,\mathrm{up}-s\,\mathrm{down}`, down row :math:`\to s\,\mathrm{up}+c\,\mathrm{down}`
    (``rotation_orbitals(nm, theta) @ M``)."""
    nm = M.shape[0] // 2
    c, s = jnp.cos(theta / 2), jnp.sin(theta / 2)
    up, dn = M[:nm], M[nm:]
    return jnp.concatenate([c * up - s * dn, s * up + c * dn])


class _RotatedBackflow(eqx.Module):
    net: Callable[[jax.Array], jax.Array]
    U0: jax.Array
    W: jax.Array
    theta: jax.Array
    charge_only: bool = eqx.field(static=True)
    dtype: DTypeLike = eqx.field(static=True)

    def _init_common(self, net, d, U0, W, theta, charge_only, dtype, rescale):
        self.net = net
        # float64 becomes float32 when x64 is off (NES runs load the float64 members of the ansatz script)
        self.dtype = dtype = jax.dtypes.canonicalize_dtype(dtype)
        self.charge_only = bool(charge_only)
        U0 = jnp.asarray(U0, dtype)
        if U0.shape[0] != get_sites().Nfmodes:
            raise ValueError(f"U0 must have {get_sites().Nfmodes} rows, got {U0.shape}")
        self.U0 = U0 / jnp.std(U0) if rescale else U0
        if W is None:
            if d is None:
                raise ValueError("give the channel dimension d or W")
            W = lecun_normal(get_subkeys(), (U0.shape[1], d // 2), dtype=dtype) / 10
        self.W = jnp.asarray(W, dtype)
        self.theta = jnp.asarray(theta, dtype)

    def features(self, s: jax.Array) -> jax.Array:
        """Backflow features of every mode, ``(2*nm, d/2)``: output row ``2c`` of the network feeds
        the up modes of channel ``c``, row ``2c+1`` the down modes (the ``DetBackflow`` layout).

        The network's exp output (``exp_by_scale``: ``significand * e^xmax``) is materialised with ``xmax``
        capped at ``MAX_EXPONENT``, unchanged below it. Beyond float32's e^88 the features would be inf, and
        a member with W = 0 NaN (inf * 0) where it should be its exact determinant: an untrained network
        can get there on rare strongly charge-fluctuating configurations (N=14, k=1 members)."""
        x = self.net(charge_only_spins(s) if self.charge_only else s)
        if isinstance(x, ScaleArray):
            x = x.significand * jnp.exp(jnp.minimum(x.exponent, MAX_EXPONENT))
        return jnp.asarray(x).reshape(-1, get_sites().Nfmodes).T.astype(self.dtype)

    def orbitals(self, s: jax.Array) -> jax.Array:
        r"""The full rotated orbital matrix :math:`R_1(\theta)(U_0+X(s)W^T)`, all :math:`2n_m` rows."""
        return rotate_rows(self.U0 + self.features(s) @ self.W.T, self.theta)

    def _fields(self) -> dict:
        return dict(net=self.net, U0=self.U0, W=self.W, theta=self.theta,
                    charge_only=self.charge_only, dtype=self.dtype)

    def rebuild(self, **changes):
        """A copy with some fields replaced (static ones included), no rescaling."""
        return type(self)(**{**self._fields(), **changes}, rescale=False)


class RotatedDetBackflow(_RotatedBackflow):
    r""":math:`\psi(s)=\det\big([R_1(\theta)(U_0+X(s)W^T)]_{\mathrm{occ}(s)}\big)`.

    :param net: backflow network with ``DetBackflow``'s output layout (e.g. ``Transformer`` with
        ``final_sum=False``).
    :param d: channel dimension of the network output (``W`` is ``(ncols, d/2)``).
    :param U0: unrotated reference orbitals, ``(2*nm, Ntotal)``.
    :param theta: rotation angle.
    :param charge_only: feed the network the orbital occupations only (exactly rotation-closed).
    :param W: backflow weights; random (``DetBackflow``'s init) when ``None``.
    :param rescale: divide ``U0`` by its std, as ``DetBackflow`` does.
    """

    def __init__(self, net, U0, d=None, theta=0.0, charge_only=True, W=None,
                 dtype=jnp.float64, rescale=True):
        self._init_common(net, d, U0, W, theta, charge_only, dtype, rescale)

    def __call__(self, s: jax.Array) -> LogArray:
        idx = fermion_idx(s)
        sign, logabs = jnp.linalg.slogdet(self.orbitals(s)[idx])
        return LogArray(sign, logabs) * fermion_inverse_sign(s)


class RotatedMultiDetBackflow(_RotatedBackflow):
    r""":math:`\psi(s)=\sum_a c_a\det\big([R_1(\theta)(U_0^a+X(s)W_a^T)]_{\mathrm{occ}(s)}\big)`: one
    network, one reference and one ``W`` per determinant.

    Every term rotates exactly when the features are charge-only, so the sum does. Sharing the
    features costs nothing where a single term is limited: on the configurations without charge
    fluctuations :math:`X` is one fixed matrix whatever the number of networks, and each block term
    adds rank :math:`\le4` there.

    :param U0: ``(ndets, 2*nm, Ntotal)``; each term is divided by its own std (``rescale``).
    :param W: ``(ndets, Ntotal, d/2)``; random when ``None``.
    :param coeffs: ``(ndets,)``, default :math:`1/n_\text{dets}`.
    """

    coeffs: jax.Array

    def __init__(self, net, U0, d=None, theta=0.0, charge_only=True, W=None, coeffs=None,
                 dtype=jnp.float64, rescale=True):
        self.net = net
        self.dtype = dtype = jax.dtypes.canonicalize_dtype(dtype)    # float32 when x64 is off
        self.charge_only = bool(charge_only)
        U0 = jnp.asarray(U0, dtype)
        if U0.ndim != 3 or U0.shape[1] != get_sites().Nfmodes:
            raise ValueError(f"U0 must be (ndets, {get_sites().Nfmodes}, ncols), got {U0.shape}")
        self.U0 = U0 / jnp.std(U0, axis=(1, 2), keepdims=True) if rescale else U0
        ndets, _, ncols = U0.shape
        if W is None:
            if d is None:
                raise ValueError("give the channel dimension d or W")
            W = lecun_normal(get_subkeys(), (ndets, ncols, d // 2), dtype=dtype) / 10
        self.W = jnp.asarray(W, dtype)
        self.coeffs = jnp.full((ndets,), 1.0 / ndets, dtype) if coeffs is None else jnp.asarray(coeffs, dtype)
        self.theta = jnp.asarray(theta, dtype)

    @property
    def ndets(self) -> int:
        return self.U0.shape[0]

    def orbitals(self, s: jax.Array) -> jax.Array:
        """``(ndets, 2*nm, Ntotal)``, every term rotated."""
        M = self.U0 + jnp.einsum("mf,acf->amc", self.features(s), self.W)
        return jax.vmap(rotate_rows, in_axes=(0, None))(M, self.theta)

    def _fields(self) -> dict:
        return {**super()._fields(), "coeffs": self.coeffs}

    def __call__(self, s: jax.Array) -> LogArray:
        idx = fermion_idx(s)
        M = self.orbitals(s)[:, idx]
        # A term with no weight next to the largest (singular, or below e^-80 of it) adds 0, but the gradient of its
        # log-determinant goes through M^-1 and 0 * inf = nan reaches every parameter: such terms are swapped for the
        # identity before the differentiated determinant, and likewise a configuration where every term vanishes
        _, logabs = jax.lax.stop_gradient(jnp.linalg.slogdet(M))
        top = jnp.max(logabs)
        top = jnp.where(jnp.isfinite(top), top, 0.0)    # every term exactly 0 outside sector k
        live = logabs > top - 80.0
        sign, logabs = jnp.linalg.slogdet(jnp.where(live[:, None, None], M, jnp.eye(M.shape[-1], dtype=M.dtype)))
        val = jnp.sum(jnp.where(live, self.coeffs * sign * jnp.exp(logabs - top), 0.0))
        safe = jnp.where(val == 0, 1.0, val)
        return LogArray(jnp.sign(val), jnp.where(val == 0, -jnp.inf, jnp.log(jnp.abs(safe)) + top)) \
            * fermion_inverse_sign(s)


def untie(model: _RotatedBackflow) -> _RotatedBackflow:
    """Charge-only -> flavour-aware, amplitudes unchanged: the ``Embedding`` row of down-only
    orbitals (2), never read while charge-only, becomes a copy of the up-only row (1). From then on
    the two rows train apart, since they are read on different configurations."""
    if not model.charge_only:
        return model
    E = model.net.layers[0].E
    net = eqx.tree_at(lambda n: n.layers[0].E, model.net, E.at[2].set(E[1]))
    return model.rebuild(net=net, charge_only=False)
