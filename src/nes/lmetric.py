r"""
L-aware metric for NES on the fuzzy sphere: Gram rows of the curvature of :math:`\left<L^- L^+\right>`.

NES is VMC on the tuple amplitude :math:`\Psi(X) = \det A(X)`, :math:`A_{la} = \psi_a(x_l)`, with the summed operator
:math:`\hat A = \sum_l A_l` acting on each slot of the tuple, so :math:`\mathrm{tr}(S^{-1} L^- L^+) =
\langle\Psi|\hat A|\Psi\rangle / \langle\Psi|\Psi\rangle` for :math:`A = L^- L^+ = (L^+)^\dagger L^+`. A step
:math:`d\theta` changes it at second order by :math:`d\theta^T Q\, d\theta`,

.. math::

    Q_{pq} = \sum_l \sum_T \overline{(L^+_l \tilde\partial_p\Psi)(T)}\,(L^+_l \tilde\partial_q\Psi)(T)
    \,/\, \langle\Psi|\Psi\rangle, \qquad \tilde\partial_p\Psi = \partial_p\Psi - \langle O_p\rangle\Psi,

a sum of squares over the raised tuples :math:`T` (slot :math:`l` in the :math:`L_z + 1` sector). The SR metric holds
nothing of the loss, so :math:`S + \mu Q` adds the penalty's Gauss-Newton curvature, which plain SR drops.

`lplus_gram_rows` estimates :math:`Q = \sum_n u_n u_n^T` from the NES samples :math:`X_n`: each draws a slot :math:`l`
and an allowed :math:`L^+` hop of :math:`x_l` uniformly, giving :math:`T_n`. Its proposal density
:math:`q(T) = \sum_{X'} \pi(X')\, p(T|X')` runs over the :math:`L^-` neighbours :math:`X'` of :math:`T` in slot
:math:`l` (:math:`\sim N/2`, :math:`X_n` among them), which is also all :math:`(L^+_l f)(T)` needs, so

.. math::

    u_n = \frac{\sum_j a_j\, \Psi(X'_j)\, (O(X'_j) - \langle O\rangle)/\Psi(X_n)}
    {\sqrt{n\, r_n^{-1} \sum_j |\Psi(X'_j)/\Psi(X_n)|^{w} / (K h_j)}}

with :math:`a_j = \sqrt{m(N-m)}` the :math:`L^+` element of the hop, :math:`h_j` the number of :math:`L^+` hops of the
neighbour's slot configuration, :math:`w` the sampler's reweight exponent and :math:`r_n` its reweight factor. It is
unbiased and positive semidefinite by construction. `SubspaceLMetricSolver` uses them to weigh the combinations of
SR's own directions (not as extra rows under :math:`\bar O`, which failed; see there).

The tuple derivatives need the members' Jacobians on the sample's own configurations and on the new slot
configuration of each neighbour, i.e. :math:`K + N/2` Jacobians per sample and member: about :math:`1 + N/(2K)` times
the cost of :math:`\bar O`. The row replacement is handled by Sherman-Morrison on the equilibrated matrix, so a
neighbour where the tuple vanishes (:math:`\Psi(X') = 0`) needs no inverse of its own.

:math:`L^+` hops move one electron between adjacent modes of one flavour (up modes :math:`0..N-1`, down modes
:math:`N..2N-1`), so they carry no fermion sign. With ``coeff = 1`` in place of :math:`\tilde\partial\Psi` the same
estimator returns :math:`\mathrm{tr}(S^{-1} L^- L^+)` itself, which `lplus_gram_rows` also reports as a check.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
from quantax.sampler import Samples
from quantax.utils import LogArray

from .optimizer import _scale_psi_matrix
from .state_set import NaturalStateSet


@jax.jit
def _subspace_solve(A: jax.Array, b: jax.Array, R: jax.Array, mu: float, rshift: float, ashift: float):
    """MinSR with every eigen-direction of A A^T damped by its leak, in float64 (see `SubspaceLMetricSolver`); also
    the share of MinSR's |A x|^2 (the step's 1 - F on this batch) that is kept. |x|^2 itself is no measure: MinSR's
    is dominated by the weakly determined directions it amplifies, which the damping removes first."""
    input_dtype = A.dtype
    with jax.enable_x64():
        A, b, R = (jnp.asarray(v, jnp.float64) for v in (A, b, R))
        T = jnp.matmul(A, A.T, precision="highest")
        n = T.shape[0]
        eps = rshift * jnp.trace(T) / jnp.sqrt(n) + ashift
        # orthonormal coordinates w of the row space, x = A^T U Sigma^-1 w (T = U Sigma^2 U^T), where the problem
        # (Sigma^2 + eps + mu W^T W) w = Sigma U^T b is conditioned by eps; T's null directions (Obar is centred) are
        # dropped, as MinSR's solution has no component there either
        lam, U = jnp.linalg.eigh(T)
        keep = lam > 1e-12 * lam[-1]
        inv_sigma = jnp.where(keep, 1 / jnp.sqrt(jnp.where(keep, lam, 1.0)), 0.0)
        W = jnp.matmul(R, A.T, precision="highest") @ U * inv_sigma[None, :]     # leak rows of the unit directions
        leak = jnp.sum(W**2, axis=0)                                             # |R v_j|^2
        rhs = jnp.sqrt(jnp.clip(lam, 0)) * (U.T @ b) * keep
        w = rhs / (jnp.where(keep, lam, 1.0) + eps + mu * leak)
        w0 = rhs / (jnp.where(keep, lam, 1.0) + eps)
        x = jnp.matmul(A.T, U @ (w * inv_sigma), precision="highest")
        kept = jnp.sum(lam * keep * w**2) / jnp.sum(lam * keep * w0**2)      # of the step's |Obar x|^2, its 1 - F
    return x.astype(input_dtype), kept


class SubspaceLMetricSolver:
    r"""
    MinSR with each of its directions damped by its own leak. With :math:`\bar O = U \Sigma V^T`, MinSR's step is
    :math:`x = \sum_j w_j v_j`, :math:`w_j = \sigma_j b_j / (\sigma_j^2 + \epsilon)`, :math:`b = U^T \bar\epsilon`;
    here

    .. math::

        w_j = \frac{\sigma_j b_j}{\sigma_j^2 + \epsilon + \mu\, |R v_j|^2},

    the diagonal of the metric :math:`S + \mu Q` in MinSR's directions. :math:`|R v_j|^2 / \sigma_j^2` is the
    :math:`L^- L^+` a step along :math:`v_j` creates per unit of its 1 - F, so a direction is damped once that exceeds
    about :math:`1/\mu`. No component grows, and none can cancel another's leak. At :math:`\mu = 0` (or ``rows``
    None) this is quantax's ``minnorm_shift_eig`` up to the null space of :math:`T`.

    Two fuller versions failed at N=8 (K=4, 512 samples, ~8500 parameters), both by letting the step cancel the
    sampled leak with directions the sampled metric does not see:

    - stacking :math:`\sqrt\mu R` under :math:`\bar O` (the full :math:`S + \mu Q`): the step is a combination of
      the rows it is given, and the Gram rows -- the directions that move L -- became step directions (junk 3x the
      plain run's, energy stalled);
    - minimising over combinations of MinSR's directions with the cross terms of :math:`Q`: weakly determined
      directions (small :math:`\sigma_j`, cheap in the fit) cancelled the leak of the others, and the first step
      moved the span by 1 - F 0.25 while ``fs`` read 1e-3 (junk 20-29).

    This one, same N=8 test (30 sweeps, max-fs 1e-3, exact junk tr L^2 / energy above ED at the end): mu 0 0.44 /
    0.53, mu 0.01 0.39 / 0.69, mu 0.1 0.11 / 1.02, mu 1 0.016 / 1.19; junk per unit energy gained 1.26, 0.72, 0.38,
    0.32. Less junk at every mu, slower energy beyond ~0.01; mu 0.01 led in energy until sweep ~13 (no early junk
    spike).

    ``rows`` (and ``mu``) are set by the caller every step; take them from an independent batch (the previous
    sweep's), since :math:`G` from the step's own batch fits its noise. The call is not jitted as a whole, so that
    new rows are seen; accepts ``diag_preconditioner`` (AdamSR's second solve) and returns the step in the original
    coordinates, as quantax expects of such solvers.
    """

    def __init__(self, rshift: float = 1e-12, ashift: float = 1e-6):
        self.rshift, self.ashift = rshift, ashift
        self.rows: Optional[jax.Array] = None
        self.mu = 0.0
        self.kept = np.nan          # share of MinSR's 1 - F kept, of the last call (AdamSR's second solve)

    def __call__(self, A: jax.Array, b: jax.Array, diag_preconditioner: Optional[jax.Array] = None, **kwargs):
        V = 1.0 if diag_preconditioner is None else diag_preconditioner
        R = jnp.zeros((1, A.shape[1]), A.dtype) if self.rows is None or self.mu <= 0 else self.rows
        x, kept = _subspace_solve(A / V, b, R / V, float(self.mu if self.rows is not None else 0.0), self.rshift,
                                  self.ashift)
        self.kept = float(kept)
        return x / V


def _jacobian(state, spins: np.ndarray) -> jax.Array:
    """``state.jacobian`` on a batch padded to a whole number of configurations per device."""
    n, ndev = len(spins), jax.device_count()
    pad = -n % ndev
    if pad:
        spins = np.concatenate([spins, np.repeat(spins[:1], pad, axis=0)])
    return state.jacobian(jnp.asarray(spins))[:n]


def _raise_and_neighbours(occ: np.ndarray, rng: np.random.Generator):
    r"""
    For slot configurations ``occ`` (n, 2, N) bool: one uniformly drawn :math:`L^+` hop each, the :math:`L^-`
    neighbours of the raised configuration (padded to the largest count), their :math:`L^+` elements (0 on padding)
    and their own numbers of :math:`L^+` hops.
    """
    n, _, N = occ.shape
    up = occ[:, :, :-1] & ~occ[:, :, 1:]              # (f, m-1) -> (f, m) allowed, indexed by m-1
    h = up.reshape(n, -1).sum(1)
    pick = np.minimum((rng.random(n) * h).astype(int), np.maximum(h - 1, 0))
    flat = np.argsort(~up.reshape(n, -1), axis=1, kind="stable")[np.arange(n), pick]     # the pick-th allowed hop
    f, m0 = flat // (N - 1), flat % (N - 1)
    raised = occ.copy()
    raised[np.arange(n), f, m0] = False
    raised[np.arange(n), f, m0 + 1] = True
    raised = np.where((h > 0)[:, None, None], raised, occ)  # no hop (rows dropped later): keep a valid configuration

    down = raised[:, :, 1:] & ~raised[:, :, :-1]      # (f, m) -> (f, m-1) allowed, indexed by m-1
    count = down.reshape(n, -1).sum(1)
    J = max(int(count.max()), 1)
    order = np.argsort(~down.reshape(n, -1), axis=1, kind="stable")[:, :J]
    valid = np.arange(J)[None, :] < count[:, None]
    fj, mj = order // (N - 1), order % (N - 1) + 1     # the electron at (fj, mj) moves to mj - 1
    neigh = np.repeat(occ[:, None], J, axis=1)          # padding: the sample's own slot
    for j in range(J):
        rows = np.flatnonzero(valid[:, j])
        neigh[rows, j] = raised[rows]
        neigh[rows, j, fj[rows, j], mj[rows, j]] = False
        neigh[rows, j, fj[rows, j], mj[rows, j] - 1] = True
    a = np.where(valid, np.sqrt(mj * (N - mj)), 0.0)
    h_neigh = (neigh[:, :, :, :-1] & ~neigh[:, :, :, 1:]).reshape(n, J, -1).sum(2)
    return neigh, a, np.maximum(h_neigh, 1), h > 0


def lplus_gram_rows(
    states: NaturalStateSet,
    samples: Samples,
    Omean: jax.Array,
    reweight: float,
    rng: np.random.Generator,
    chunk: int = 128,
) -> tuple[jax.Array, float]:
    r"""
    Rows ``u`` (nsamples, nparams) with :math:`Q \approx u^T u` (see the module docstring), in the parameter order of
    ``states`` (member blocks), and the same estimator's :math:`\mathrm{tr}(S^{-1} L^- L^+)`.

    :param Omean:
        :math:`\langle O \rangle` over the batch, as the optimizer's centering of :math:`\bar O` uses it.

    :param reweight:
        The sampler's exponent: tuples were drawn from :math:`|\det A|^\mathrm{reweight}`.

    Rows that come out non-finite (tuples at a node of a member) are zeroed, as `get_Obar` does with its own.
    """
    if not all(states.trainable):
        raise NotImplementedError("lplus_gram_rows assumes every member is trainable")
    spins = np.asarray(samples.spins)
    n, K, Nmodes = spins.shape
    N = Nmodes // 2
    slot = rng.integers(K, size=n)
    occ = spins[np.arange(n), slot].reshape(n, 2, N) > 0
    neigh_occ, a, h_neigh, has_hop = _raise_and_neighbours(occ, rng)
    J = a.shape[1]
    neigh = np.where(neigh_occ, 1, -1).astype(spins.dtype).reshape(n, J, Nmodes)

    # equilibrated tuple matrix A_s = D^-1 A C^-1 and B_s = A_s^-1; the neighbours' rows on the same scales
    A_s, row_shift, col_shift = _scale_psi_matrix(states.psi_matrix(jnp.asarray(spins)))
    B_s = jnp.linalg.pinv(A_s)
    vals = LogArray.from_value(states.amplitudes(jnp.asarray(neigh.reshape(-1, Nmodes)))).reshape(n, J, K)
    shift = row_shift[jnp.arange(n), slot][:, None, None] + col_shift[:, None, :]
    r_s = vals.sign * jnp.exp(vals.logabs - shift)                            # (n, J, K)

    Bi = B_s[jnp.arange(n), :, slot]                                         # B_s[:, a, i]   (n, K)
    Ai = A_s[jnp.arange(n), slot, :]                                         # A_s[i, :]      (n, K)
    rho = jnp.einsum("nja,na->nj", r_s, Bi)                                  # Psi(X'_j) / Psi(X)
    v = jnp.einsum("nja,nal->njl", r_s - Ai[:, None, :], B_s)
    # rho_j A'_j^-1 by Sherman-Morrison, finite where A'_j is singular
    M = rho[:, :, None, None] * B_s[:, None] - Bi[:, None, :, None] * v[:, :, None, :]      # (n, J, a, l)
    a_j = jnp.asarray(a, dtype=r_s.dtype)
    is_slot = jnp.arange(K)[None, :] == jnp.asarray(slot)[:, None]           # (n, l)
    # coefficients of the members' log-derivatives: on the sample's slots l != i, and on each neighbour's slot i
    kappa = jnp.where(is_slot[:, None, :], 0.0, jnp.einsum("nj,njal,nla->nal", a_j, M, A_s))  # (n, a, l)
    M_i = M[jnp.arange(n), :, :, slot]                                       # (n, J, a)
    nu = jnp.einsum("nj,nja,nja->naj", a_j, M_i, r_s)                       # (n, a, j)
    gamma = jnp.sum(a_j * rho, axis=1)                                       # (L^+_i Psi)(T) / Psi(X)

    rf = jnp.ones(n) if samples.reweight_factor is None else samples.reweight_factor
    valid = a_j > 0
    density = jnp.sum(jnp.where(valid, jnp.abs(rho) ** reweight, 0.0) / jnp.asarray(h_neigh), axis=1) / K
    den = jnp.sqrt(n * density / rf)
    good = jnp.asarray(has_hop) & jnp.isfinite(den) & (den > 0) & jnp.all(jnp.isfinite(M), axis=(1, 2, 3))
    den = jnp.where(good, den, 1.0)
    penalty = float(jnp.sum(jnp.where(good, gamma**2 / den**2, 0.0)))

    rows, offset = [], 0
    for index, state in enumerate(states.states):
        p = state.nparams
        Om = Omean[offset:offset + p]
        offset += p
        block = []
        for c in range(0, n, chunk):
            s = slice(c, min(c + chunk, n))
            nc = s.stop - s.start
            J_x = _jacobian(state, spins[s].reshape(-1, Nmodes)).reshape(nc, K, p)
            J_n = _jacobian(state, neigh[s].reshape(-1, Nmodes)).reshape(nc, J, p)
            u = (jnp.einsum("nl,nlp->np", kappa[s, index], J_x) + jnp.einsum("nj,njp->np", nu[s, index], J_n)
                 - gamma[s, None] * Om[None, :])
            block.append(u / den[s, None])
        rows.append(jnp.concatenate(block))
    rows = jnp.concatenate(rows, axis=1)
    rows = jnp.where((good & jnp.all(jnp.isfinite(rows), axis=1))[:, None], rows, 0.0)
    return rows.real, penalty
