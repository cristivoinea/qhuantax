from typing import Optional, Union, Callable
import numpy as np
import jax
import jax.lax as lax
import jax.numpy as jnp
import quantax as qtx
from quantax.global_defs import get_default_dtype, is_default_cpl
from quantax.operator.operator import _get_conn_size, _get_conn, _get_Olocx
from quantax.optimizer import EnergyGrad, OverlapGrad
from quantax.utils import chunk_map, ints_to_array


class ERDipoleCons(qtx.optimizer.QNGD):
    r"""
    Exact reconfiguration, performed by a full summation in the whole Hilbert space.
    This is only available in small systems.
    """

    def __init__(
        self,
        state: qtx.state.Variational,
        hamiltonian: qtx.operator.Operator,
        lz_mask: jnp.array,
        imag_time: bool = True,
        solver: Optional[Callable] = None,
        symm: Optional[qtx.symmetry.Symmetry] = None,
    ):
        r"""
        :param state:
            Variational state to be optimized.

        :param hamiltonian:
            The Hamiltonian for the evolution.

        :param imag_time:
            Whether to use imaginary-time evolution, default to True.

        :param solver:
            The numerical solver for the matrix inverse, default to `~quantax.optimizer.auto_pinv_eig`.

        :param symm:
            Symmetry used to construct the Hilbert space, default to be the symmetry
            of the variational state.
        """
        super().__init__(state, EnergyGrad(hamiltonian), imag_time=imag_time, solver=solver)

        self._hamiltonian = hamiltonian
        self._energy = None
        self._Omean = None

        self._symm = state.symm if symm is None else symm
        self._symm.basis_make()
        basis = self._symm.basis
        basis_states = np.asarray(basis.states)
        lz_mask = np.asarray(lz_mask)
        if lz_mask.dtype == np.bool_:
            if lz_mask.shape != (basis_states.size,):
                raise ValueError(
                    "`lz_mask` as a boolean mask must have one entry per basis state."
                )
            lz_indices = np.flatnonzero(lz_mask)
        elif np.issubdtype(lz_mask.dtype, np.integer):
            lz_indices = lz_mask.astype(np.int64).reshape(-1)
        else:
            raise TypeError("`lz_mask` must be a boolean mask or integer indices.")
        if np.any(lz_indices < 0) or np.any(lz_indices >= basis_states.size):
            raise ValueError("`lz_mask` contains indices outside the basis.")
        if np.unique(lz_indices).size != lz_indices.size:
            raise ValueError("`lz_mask` contains duplicate basis indices.")
        if lz_indices.size == 0:
            raise ValueError("`lz_mask` selects an empty Lz sector.")

        self._basis_size = basis_states.size
        self._lz_mask = jnp.asarray(lz_indices)
        self._spins = ints_to_array(basis_states[lz_indices])
        self._symm_norm = jnp.asarray(basis.get_amp(basis_states[lz_indices]))
        if not is_default_cpl():
            self._symm_norm = self._symm_norm.real

    @property
    def hamiltonian(self) -> qtx.operator.Operator:
        """The Hamiltonian for the evolution."""
        return self._hamiltonian

    @property
    def energy(self) -> Optional[float]:
        """Energy of the current step."""
        return self._energy

    def get_Ebar(self, psi: jax.Array) -> jax.Array:
        r"""Compute :math:`\bar \epsilon` in the full Hilbert space."""
        full_psi = (
            jnp.zeros(self._basis_size, dtype=psi.dtype).at[self._lz_mask].set(psi)
        )
        dense = qtx.state.DenseState(full_psi, self._symm)
        H_psi = self._hamiltonian @ dense
        energy = dense @ H_psi
        Ebar = H_psi - dense * energy
        self._energy = energy.real
        return jnp.asarray(Ebar.psi)[self._lz_mask]

    def get_Obar(self, psi: jax.Array) -> jax.Array:
        r"""Compute :math:`\bar O` in the full Hilbert space."""
        Omat = self._state.jacobian(self._spins) * psi[:, None]
        Omat = jnp.where(jnp.isnan(Omat), 0, Omat)
        self._Omean = jnp.einsum("s,sk->k", psi.conj(), Omat)
        Omean = jnp.einsum("s,k->sk", psi, self._Omean)
        return Omat - Omean

    def get_step(self) -> jax.Array:
        r"""
        Obtain the optimization step by solving the equation :math:`\bar O \dot \theta = \bar \epsilon`.
        """
        psi = self._state(self._spins) / self._symm_norm
        psi /= jnp.linalg.norm(psi)
        Ebar = self.get_Ebar(psi)
        Obar = self.get_Obar(psi)
        step, self._buffers = self.solve(Obar, Ebar, self._buffers)
        return step


class FuzzySphereSupervised(qtx.optimizer.QNGD):
    def __init__(
        self,
        state: qtx.state.Variational,
        target_state: qtx.state.State,
        solver: Optional[Callable[[jax.Array, jax.Array], jax.Array]] = None,
    ):
        super().__init__(state, OverlapGrad(target_state), solver=solver)
        self._target_state = target_state

        self._loss_mean = None
        self._loss_variance = None

    @property
    def loss_mean(self) -> Optional[float]:
        """Loss function for the current step."""
        return self._loss_mean
    
    @property
    def loss_variance(self) -> Optional[float]:
        """Loss function for the current step."""
        return self._loss_variance

    def get_Ebar(self, samples: qtx.sampler.Samples) -> jax.Array:
        phi = self._target_state(samples.spins)
        psi = samples.psi
        ratio = phi / psi
        reweight = samples.reweight_factor

        ratio_mean = jnp.mean(ratio * reweight)
        ratio_var = jnp.abs(ratio - ratio_mean) ** 2
        self._loss_mean = ratio_mean.real
        self._loss_variance = jnp.mean(ratio_var * samples.reweight_factor).real

        ratio = ratio / ratio_mean - 1
        Ebar = -ratio * jnp.sqrt(reweight / samples.nsamples)
        return Ebar



class Supervised_KL_Sign(qtx.optimizer.StochasticQNGD):
    """Supervised loss on log-amplitude + sign, in the spirit of arXiv:2507.13322.

    Base class note: this must derive from ``StochasticQNGD``, not ``QNGD`` --
    ``get_Obar`` and ``get_step`` are defined on ``StochasticQNGD``, so with ``QNGD``
    as the base the optimizer has no ``get_step`` at all and cannot run.

    The local loss is
        2*(log|psi| - log|phi|)  +  sign_weight * |sign(psi) - sign(phi)|^2
    and ``get_Ebar`` centers it, so the *unsquared* log term is the KL-divergence
    estimator D_KL(|psi|^2 || |phi|^2): centering cancels the arbitrary additive
    constant coming from psi and phi being unnormalized, which is why the unsquared
    form needs no explicit normalization handling. (The paper's Eq. 2 instead uses the
    *squared* log-ratio, which is not offset-invariant -- see
    ``Supervised_LogSq_Sign`` for that variant, where the mean log-ratio must be
    subtracted before squaring.)
    """

    def __init__(
        self,
        state: qtx.state.Variational,
        target_state: qtx.state.State,
        sign_weight: float,
        solver: Optional[Callable[[jax.Array, jax.Array], jax.Array]] = None,
        zero_penalty: float = 4.0,
    ):
        super().__init__(state, OverlapGrad(target_state), solver=solver)
        self._target_state = target_state
        self._sign_weight = sign_weight
        self._zero_penalty = zero_penalty
        self._frac_offsupport = None

    @property
    def frac_offsupport(self) -> Optional[float]:
        """Fraction of sampled configs where the target is exactly zero."""
        return self._frac_offsupport

    @property
    def loss_total(self) -> Optional[float]:
        """Loss function for the current step."""
        return self._loss_total
    
    @property
    def loss_density(self) -> Optional[float]:
        """Loss function for the current step."""
        return self._loss_density
    
    @property
    def loss_sign(self) -> Optional[float]:
        """Loss function for the current step."""
        return self._loss_sign

    @property
    def loss_var(self) -> Optional[float]:
        """Variance for the loss function estimate of the current step."""
        return self._loss_var


    def get_Ebar(self, samples: qtx.sampler.Samples) -> jax.Array:
        target = jnp.asarray(self._target_state(samples.spins))
        target_sign = jnp.sign(target)

        # The Laughlin model state is supported only on its squeezed set, so the
        # target is EXACTLY zero on many sampled configs and log|phi| = -inf there.
        # Mask the log term to the support; off-support configs instead take a flat
        # `zero_penalty`, which through the score-function route pushes probability
        # weight off them -- i.e. "train the non-squeezed configs toward zero".
        on_support = jnp.abs(target) > 0
        safe_logabs = jnp.where(on_support, jnp.log(jnp.abs(jnp.where(
            on_support, target, 1.0))), 0.0)

        psi_logabs = samples.psi.logabs
        psi_sign = samples.psi.sign

        kl_div = jnp.where(on_support, 2 * (psi_logabs - safe_logabs),
                           self._zero_penalty)
        sign_div = jnp.abs(psi_sign - target_sign)**2

        loss = kl_div + self._sign_weight * sign_div
        reweight = samples.reweight_factor
        if reweight is None:
            reweight = 1.0

        loss_mean = jnp.mean(loss * reweight)

        self._loss_total = loss_mean
        self._loss_density = jnp.mean(
            jnp.where(on_support, kl_div, 0.0) * reweight)
        self._loss_sign = jnp.mean(sign_div * reweight)
        self._frac_offsupport = jnp.mean((~on_support).astype(jnp.float32))
        loss_var = jnp.abs(loss - loss_mean) ** 2
        self._loss_var = jnp.mean(loss_var * reweight).real

        loss = loss - loss_mean
        Ebar = loss * jnp.sqrt(reweight / samples.nsamples)
        return Ebar



class Supervised_LogSq_Sign(qtx.optimizer.StochasticQNGD):
    r"""The paper's density loss (arXiv:2507.13322 Eq. 2) adapted to real amplitudes:

        L_rho = E_{|psi|^2} [ ( ln|psi|^2 - ln|phi|^2 - c )^2 ],
        L_sign = E_{|psi|^2} [ |sign(psi) - sign(phi)|^2 ],
        L = L_rho + sign_weight * L_sign

    The offset ``c`` is the sample mean of the log-ratio. It is *required*: psi and phi
    are unnormalized, so ln|psi|^2 - ln|phi|^2 carries an arbitrary additive constant.
    Without removing it the squared loss penalizes the overall normalization rather
    than the shape, and the minimum is not at psi proportional to phi. (For the unsquared/KL form in
    ``Supervised_KL_Sign`` the constant cancels automatically when ``get_Ebar``
    centers, which is why that variant does not need this.)

    Since ``sign()`` is piecewise constant, the sign term contributes no ordinary
    derivative; it acts here only through the score-function (REINFORCE) route implicit
    in the QNGD covariance estimator, i.e. by moving probability weight off
    sign-mismatched configurations rather than by flipping signs directly.
    """

    def __init__(
        self,
        state: qtx.state.Variational,
        target_state: qtx.state.State,
        sign_weight: float,
        solver: Optional[Callable[[jax.Array, jax.Array], jax.Array]] = None,
        zero_penalty: float = 4.0,
    ):
        super().__init__(state, OverlapGrad(target_state), solver=solver)
        self._target_state = target_state
        self._sign_weight = sign_weight
        self._zero_penalty = zero_penalty
        self._loss_total = self._loss_density = self._loss_sign = None
        self._frac_offsupport = None

    @property
    def frac_offsupport(self):
        return self._frac_offsupport

    @property
    def loss_total(self):
        return self._loss_total

    @property
    def loss_density(self):
        return self._loss_density

    @property
    def loss_sign(self):
        return self._loss_sign

    def get_Ebar(self, samples: qtx.sampler.Samples) -> jax.Array:
        target = jnp.asarray(self._target_state(samples.spins))
        target_sign = jnp.sign(target)

        # target is exactly zero off the squeezed set -> log|phi| = -inf there; mask
        # the density term to the support and penalize off-support configs flatly.
        on_support = jnp.abs(target) > 0
        safe_logabs = jnp.where(on_support, jnp.log(jnp.abs(jnp.where(
            on_support, target, 1.0))), 0.0)

        psi_logabs = samples.psi.logabs
        psi_sign = samples.psi.sign

        reweight = samples.reweight_factor
        if reweight is None:
            reweight = 1.0

        log_ratio = 2 * (psi_logabs - safe_logabs)
        # offset kills the arbitrary normalization; average over the SUPPORT only
        w_sup = on_support.astype(log_ratio.dtype)
        offset = (jnp.sum(log_ratio * w_sup * reweight)
                  / jnp.clip(jnp.sum(w_sup * reweight), 1e-12, None))
        density = jnp.where(on_support, (log_ratio - offset) ** 2,
                            self._zero_penalty)
        sign_div = jnp.abs(psi_sign - target_sign) ** 2

        loss = density + self._sign_weight * sign_div
        loss_mean = jnp.mean(loss * reweight)
        self._loss_total = loss_mean
        self._loss_density = jnp.mean(jnp.where(on_support, density, 0.0) * reweight)
        self._loss_sign = jnp.mean(sign_div * reweight)
        self._frac_offsupport = jnp.mean((~on_support).astype(jnp.float32))

        Ebar = (loss - loss_mean) * jnp.sqrt(reweight / samples.nsamples)
        return Ebar


class SupervisedExact_KL_Sign(qtx.optimizer.SupervisedExact):
    def __init__(
        self,
        state: qtx.state.Variational,
        target_state: qtx.state.State,
        sign_weight : int,
        solver: Optional[Callable] = None,
        symm: Optional[qtx.symmetry.Symmetry] = None,
        restricted_to: Optional[jax.Array] = None,
    ):
        self._sign_weight = sign_weight

        super().__init__(state, target_state, solver=solver, symm=symm)
        if restricted_to is None:
            restricted_to = jnp.arange(self._Ns)
        self._resctricted_to = restricted_to
        self._target_psi = jnp.asarray(target_state.todense(self._symm).psi)[
            self._resctricted_to
        ]

    @property
    def loss_fn(self) -> Optional[float]:
        """Loss function for the current step."""
        return self._loss_fn

    def get_epsilon(self, psi: jax.Array) -> jax.Array:
        kl_div = (np.log(np.abs(psi)) - np.log(np.abs(self._target_psi)))**2
        self._loss_fn = np.sum(np.abs(psi)**2 * kl_div)
        return kl_div

    def get_Obar(self, psi: jax.Array) -> jax.Array:
        Omat = self._state.jacobian(self._spins[self._resctricted_to]) * psi[:, None]
        self._Omean = jnp.einsum("s,sk->k", psi.conj(), Omat)
        Omean = jnp.einsum("s,k->sk", psi, self._Omean)
        return Omat - Omean

    def get_step(self) -> jax.Array:
        psi = self._state(self._spins) / self._symm_norm
        self._psi = psi / jnp.linalg.norm(psi)
        psi = self._psi[self._resctricted_to]
        epsilon = self.get_epsilon(psi)
        Obar = self.get_Obar(psi)
        step = self.solve(Obar, epsilon)
        return step


def _squeezed_dominance_from_spins(
    spins: jax.Array,
    hopping_particle: int,
    n_particles: int,
    root_cumsum: jax.Array,
) -> jax.Array:
    Nmodes = spins.shape[-1]
    # Keep only occupied positions and extract the N largest directly.
    # This avoids sorting the full mode axis for every connected configuration.
    pos = jnp.where(spins == hopping_particle, jnp.arange(Nmodes), -1)
    pos_desc, _ = lax.top_k(pos, n_particles)
    return jnp.all(jnp.cumsum(pos_desc, axis=-1) <= root_cumsum, axis=-1)


class _SqueezedEnergyMixin:
    def _init_squeezed_energy(
        self, root_partition: jax.Array, hopping_particle: int
    ) -> None:
        # Cache the root prefix sums once. The dominance test then becomes
        # a simple cumulative-sum comparison on connected configurations.
        root_desc = jnp.sort(jnp.asarray(root_partition))[::-1]
        self._root_partition = root_desc
        self._root_cumsum = jnp.cumsum(root_desc)
        self._n_particles = root_desc.shape[0]
        self._hopping_particle = hopping_particle

    def _squeezed_Oloc(self, samples: qtx.sampler.Samples) -> jax.Array:
        """Compute the local energy after projecting H onto the squeezed sector."""
        state = self._state
        forward_chunk = getattr(state, "forward_chunk", None)
        ref_chunk = getattr(state, "ref_chunk", None)

        spins = samples.spins
        psi = samples.psi
        internal = samples.state_internal

        Oloc = self._hamiltonian.apply_diag(spins)
        off_diags = self._hamiltonian.apply_off_diag(spins)

        for nflips, (s_conn, H_conn) in off_diags.items():
            # Project H onto the squeezed sector by discarding matrix elements
            # that connect the sampled state to configurations outside the
            # root-dominated basis.
            valid_conn = _squeezed_dominance_from_spins(
                s_conn,
                self._hopping_particle,
                self._n_particles,
                self._root_cumsum,
            )
            H_conn = jnp.where(valid_conn, H_conn, 0)
            conn_size = _get_conn_size(H_conn, forward_chunk).item()

            if conn_size == 0:
                continue

            def get_Oloc_terms(spins, psi, s_conn, H_conn, internal):
                segment, s_conn, H_conn = _get_conn(s_conn, H_conn, conn_size)
                if internal is None:
                    internal = state.init_internal(spins)
                psi_conn = state.segment_ref_forward(
                    s_conn, spins, {"nflips": nflips}, segment, internal
                )
                return _get_Olocx(psi, segment, psi_conn, H_conn)

            in_axes = (0, 0, 0, 0, None) if internal is None else 0
            get_Oloc_terms = chunk_map(get_Oloc_terms, in_axes, chunk_size=ref_chunk)
            Oloc += get_Oloc_terms(spins, psi, s_conn, H_conn, internal)

        return Oloc

    def get_Ebar(self, samples: qtx.sampler.Samples) -> jax.Array:
        Eloc = self._squeezed_Oloc(samples).astype(get_default_dtype())
        Emean = jnp.mean(Eloc * samples.reweight_factor)
        self._energy = Emean.real
        Evar = jnp.abs(Eloc - Emean) ** 2
        self._VarE = jnp.mean(Evar * samples.reweight_factor).real

        Eloc -= jnp.mean(Eloc)
        Eloc *= jnp.sqrt(samples.reweight_factor / samples.nsamples)
        return Eloc


class SqueezedSR(_SqueezedEnergyMixin, qtx.optimizer.SR):
    def __init__(
        self,
        state: qtx.state.Variational,
        hamiltonian: qtx.operator.Operator,
        root_partition: jax.Array,
        hopping_particle: int = 1,
        imag_time: bool = True,
        solver: Optional[Callable] = None,
    ):
        super().__init__(state, hamiltonian, imag_time, solver)
        self._init_squeezed_energy(root_partition, hopping_particle)


class SqueezedAdamSR(_SqueezedEnergyMixin, qtx.optimizer.AdamSR):
    def __init__(
        self,
        state: qtx.state.Variational,
        hamiltonian: qtx.operator.Operator,
        root_partition: jax.Array,
        hopping_particle: int = 1,
        imag_time: bool = True,
        solver: Optional[Callable] = None,
        mu: float = 0.95,
        beta: float = 0.995,
        file: Union[None, str] = None,
    ):
        super().__init__(
            state,
            hamiltonian,
            imag_time=imag_time,
            solver=solver,
            mu=mu,
            beta=beta,
            file=file,
        )
        self._init_squeezed_energy(root_partition, hopping_particle)
