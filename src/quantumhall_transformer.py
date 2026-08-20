from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import quantax as qtx
from quantax.global_defs import get_sites, get_subkeys
from quantax.nn import (
    lecun_normal,
    he_normal,
    glorot_normal,
    Sequential,
    pair_cpl,
    exp_by_scale,
)
from quantax.utils import PsiArray


class Embedding(eqx.Module):
    """Embedding layer."""

    E: jax.Array
    P: jax.Array

    def __init__(self, d: int, dtype=jnp.float32):
        """
        Initialize the embedding layer
        
        :param d: Dimension of the embedding
        :param dtype: Data type of the embedding
        """

        self.E = jr.normal(get_subkeys(), (4, d), dtype=dtype)
        self.P = jr.normal(get_subkeys(), (get_sites().Nsites, d), dtype=dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.reshape(-1, get_sites().Nsites)
        x = (2 * (x[0] < 0) + (x[1] < 0)).astype(jnp.uint8)
        out = self.E[x] + self.P
        return out.T


class MHSA(eqx.Module):
    layer_norm: eqx.nn.LayerNorm
    WQ: jax.Array
    WK: jax.Array
    WV: jax.Array
    W0: jax.Array

    def __init__(self, heads: int, d: int, dtype=jnp.float32):
        dH = d // heads
        lecun_init = jax.nn.initializers.lecun_normal(
            in_axis=1, out_axis=2, batch_axis=0, dtype=dtype
        )
        self.WQ = lecun_init(get_subkeys(), (heads, d, dH))
        self.WK = lecun_init(get_subkeys(), (heads, d, dH))
        self.WV = lecun_init(get_subkeys(), (heads, d, dH))
        self.W0 = lecun_normal(get_subkeys(), (d, d), dtype)
        N = get_sites().Nsites
        self.layer_norm = eqx.nn.LayerNorm((d, N), use_weight=False, use_bias=False)

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x

        x = self.layer_norm(x)
        Q = jnp.einsum("hcd,ci->hdi", self.WQ, x)
        K = jnp.einsum("hcd,ci->hdi", self.WK, x)
        dot = jnp.einsum("hdi,hdj->hij", Q, K)
        alpha = jax.nn.softmax(dot / jnp.sqrt(self.WK.shape[-1]))

        V = jnp.einsum("hcd,ci->hdi", self.WV, x)
        attention = jnp.einsum("hij,hdj->hdi", alpha, V)
        N = attention.shape[2]
        attention = attention.reshape(-1, N)
        attention = self.W0 @ attention
        return attention + residual


class FFN(eqx.Module):
    layer_norm: eqx.nn.LayerNorm
    W1: jax.Array
    b1: jax.Array
    W2: jax.Array
    b2: jax.Array

    def __init__(self, d: int, dtype=jnp.float32):
        N = get_sites().Nsites
        self.layer_norm = eqx.nn.LayerNorm((d, N), use_weight=False, use_bias=False)

        self.W1 = he_normal(get_subkeys(), (4 * d, d), dtype)
        self.b1 = jnp.zeros((4 * d, 1), dtype=dtype)

        self.W2 = glorot_normal(get_subkeys(), (d, 4 * d), dtype)
        self.b2 = jnp.zeros((d, 1), dtype=dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        N = get_sites().Nsites
        x = x.reshape(-1, N)
        residual = x
        x = self.layer_norm(x)

        x = self.W1 @ x + self.b1
        x = jax.nn.silu(x)
        x = self.W2 @ x + self.b2
        return x + residual


class Transformer(Sequential):
    nblocks: int
    d: int
    heads: int
    final_activation: Callable[[jax.Array], PsiArray]
    final_sum: bool
    dtype: jnp.dtype
    out_dtype: jnp.dtype
    layers: Tuple[Callable, ...]
    holomorphic: bool

    def __init__(
        self,
        nblocks: int,
        d: int,
        heads: int = 4,
        final_activation: Optional[Callable[[jax.Array], PsiArray]] = None,
        final_sum: bool = True,
        dtype: jnp.dtype = jnp.float32,
        out_dtype: Optional[jnp.dtype] = None,
    ):
        self.nblocks = nblocks
        self.d = d
        self.heads = heads
        if final_activation is None:
            final_activation = exp_by_scale
        self.final_activation = final_activation
        self.final_sum = final_sum
        self.dtype = dtype
        if out_dtype is None:
            out_dtype = dtype
        self.out_dtype = out_dtype

        layers = [Embedding(d, dtype)]
        for l in range(nblocks):
            layers.append(MHSA(heads, d, dtype))
            layers.append(FFN(d, dtype))

        def final_layer(x):
            x /= jnp.sqrt(nblocks + 1)
            if jnp.issubdtype(out_dtype, jnp.complexfloating):
                x = pair_cpl(x)
            x = x.astype(out_dtype)
            x = final_activation(x)
            if final_sum:
                x = x.sum()
            return x

        layers = [*layers, final_layer]
        super().__init__(layers)


def transformer_backflow_state(
    index: int,
    L: int,
    N: int,
    d: int,
    nb: int,
    nh: int,
    symm,
    pf_backflow: bool,
    U: np.ndarray,
    coeffs: Optional[Sequence[float]] = None,
    orbital_noise: float = 5e-2,
    rng: np.random.Generator = np.random.default_rng(),
    param_file: Union[None, str, Path] = None,
    backflow_scale: float = 1.0,
    nterms: int = 1,
):
    """A `Transformer`-backflow state: one determinant or Pfaffian, or a sum of them.

    `pf_backflow` picks Pfaffian over determinant, `nterms` picks a sum over a single
    term, and the four combinations are the four models a run can be built from. Every
    script that reads a checkpoint has to reproduce the same choice, since the tree is
    what the checkpoint is keyed to and a mismatch does not raise -- hence one definition
    here rather than a copy per script, and `check_param_file` on the way in.

    `coeffs` switches on the multi-determinant mean-field reference: `U` is then the determinant
    stack shared by every member (shape `(ndets, 2L, N)`) and `coeffs` is this member's vector over
    it, so the members differ by their coefficients rather than by noise, and the term count comes
    from that vector. Without it, `nterms` sets how many perturbed copies of `U` this member sums.
    """
    from qhuantax.quantumhall_meanfield import backflow_coeffs
    from qhuantax.quantumhall_models import MultiDetBackflow, MultiPfBackflow
    from qhuantax.quantumhall_utils import check_param_file, scale_backflow

    def wrap(model):
        # `param_file` replaces every leaf of the model, U0 and W included, so the orbital
        # noise and the backflow scale above are irrelevant whenever one is given.
        model = scale_backflow(model, backflow_scale)
        if param_file is not None:
            check_param_file(model, param_file)
        return qtx.state.Variational(model, param_file=param_file, symm=symm,
                                     max_parallel=16384, use_ref=False)

    U_state = U.copy()
    if orbital_noise > 0:
        U_state = U_state + orbital_noise * rng.normal(size=U_state.shape)

    if coeffs is not None:
        if pf_backflow:
            raise ValueError("--mean-field builds determinants; drop --pf-backflow")
        # Only the determinants this member actually uses. Carrying the others would give each a
        # network whose coefficient is zero, so its gradient vanishes identically -- a singular SR
        # solve, and `ndets` times the parameters. Without --orthogonalize the mean-field driver
        # writes one reference per state, so this is usually a small subset of the stack.
        used = np.flatnonzero(np.asarray(coeffs))
        U_state, coeffs = U_state[used], np.asarray(coeffs)[used]

        if len(used) == 1:
            net = Transformer(nblocks=nb, d=d, heads=nh, final_sum=False)
            return wrap(qtx.model.DetBackflow(net, U0=jnp.asarray(U_state[0]), d=d))

        nets = [Transformer(nblocks=nb, d=d, heads=nh, final_sum=False) for _ in U_state]
        return wrap(MultiDetBackflow(
            nets,
            U0=jnp.asarray(U_state),
            # `DetBackflow` divides each determinant by its own std, which would destroy the
            # relative weights; `backflow_coeffs` undoes exactly that.
            coeffs=jnp.asarray(backflow_coeffs(U_state, coeffs)),
            d=d,
        ))

    # One perturbed copy of the reference per term. The copies must differ, or the terms are
    # identical and the sum is rank deficient; `orbital_noise` above separates the *members*, not
    # the terms within one.
    U0s = [U_state if nterms == 1 else U_state + 1e-2 * rng.normal(size=U_state.shape)
           for _ in range(nterms)]
    if pf_backflow:
        for k, U_alpha in enumerate(U0s):
            U_pf = jnp.zeros((2 * L, 2 * L))
            for i in range(N):
                U_pf = U_pf.at[:, 2 * i].add(U_alpha[:, i])
            U0s[k] = U_pf

    nets = [Transformer(nblocks=nb, d=d, heads=nh, final_sum=False) for _ in U0s]
    if nterms > 1:
        multi = MultiPfBackflow if pf_backflow else MultiDetBackflow
        return wrap(multi(nets=nets, U0=jnp.stack([jnp.asarray(u) for u in U0s]),
                          coeffs=jnp.ones(nterms) / nterms, d=d))
    if pf_backflow:
        return wrap(qtx.model.PfBackflow(nets[0], U0=U0s[0], d=d))
    return wrap(qtx.model.DetBackflow(nets[0], U0=U0s[0], d=d))
