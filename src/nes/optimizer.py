from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Callable, Optional, Union

import jax
import jax.numpy as jnp
import quantax as qtx
from quantax.global_defs import get_default_dtype
from quantax.sampler import Samples
from quantax.utils import LogArray

from .state_set import NaturalStateSet


def _scaled_psi_matrix(
    states: NaturalStateSet,
    tuple_spins: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    r"""
    Materialize the NES matrix with per-row stabilization.

    If ``A[i, a] = psi_a(s_i)`` and ``D[i, i] = exp(row_shift[i])``, this returns
    ``A_scaled = D^{-1} A``. The same row scaling must be used for ``B`` in the
    local-energy trace, so that ``solve(A_scaled, D^{-1} B) = A^{-1} B``.
    """
    psi_matrix = LogArray.from_value(states.psi_matrix(tuple_spins))
    row_shift = jnp.max(psi_matrix.logabs, axis=-1)
    row_shift = jnp.where(jnp.isfinite(row_shift), row_shift, 0.0)
    scaled = psi_matrix.sign * jnp.exp(psi_matrix.logabs - row_shift[..., None])
    return scaled, row_shift


class NaturalTraceEnergyGrad(qtx.optimizer.EnergyGrad):
    """"""

    _Eloc_max_dev = None

    @property
    def Eloc_max_dev(self) -> Optional[jax.Array]:
        r"""
        :math:`\max_s |E_{loc}(s) - \bar E|` over the last batch.

        Compared against ``sqrt(VarE)``, this says whether :math:`\bar \epsilon` is
        dominated by a single sample -- the SR right-hand side is used one realization at
        a time, not in expectation, so a lone outlier translates directly into a step.
        """
        return self._Eloc_max_dev

    def local_energy(self, states: NaturalStateSet, samples: Samples) -> jax.Array:
        tuple_spins = jnp.asarray(samples.spins)
        nsamples, Nstates, Nmodes = tuple_spins.shape
        if Nstates != states.Nstates or Nmodes != states.Nmodes:
            raise ValueError(
                "Expected sample spins with shape "
                f"(nsamples, {states.Nstates}, {states.Nmodes}), got "
                f"{tuple_spins.shape}."
            )

        flat_spins = tuple_spins.reshape(nsamples * Nstates, Nmodes)
        S_scaled, row_shift = _scaled_psi_matrix(states, tuple_spins)

        H_columns = []
        for state in states.states:
            psi = jnp.asarray(state(flat_spins))
            Eloc = self.hamiltonian.Oloc(state, flat_spins).astype(get_default_dtype())
            H_columns.append(Eloc * psi)

        H = jnp.stack(H_columns, axis=-1).reshape(nsamples, Nstates, Nstates)
        H_scaled = H * jnp.exp(-row_shift[..., None])
        Sinv_H = jnp.linalg.pinv(S_scaled) @ H_scaled
        return jnp.trace(Sinv_H, axis1=-2, axis2=-1).astype(get_default_dtype())

    def ebar(self, states: NaturalStateSet, samples: Samples) -> jax.Array:
        Eloc = self.local_energy(states, samples)

        if samples.reweight_factor is None:
            reweight_factor = jnp.ones(samples.nsamples)
        else:
            reweight_factor = samples.reweight_factor

        Emean = jnp.mean(Eloc * reweight_factor)
        self._energy = Emean.real
        Evar = jnp.abs(Eloc - Emean) ** 2
        self._VarE = jnp.mean(Evar * reweight_factor).real
        # Free: `Evar` is already materialized, so the max is one more reduction over it.
        self._Eloc_max_dev = jnp.sqrt(jnp.max(Evar))

        Eloc -= jnp.mean(Eloc)
        Eloc *= jnp.sqrt(reweight_factor / samples.nsamples)
        return Eloc


class NaturalExcitedSR(qtx.optimizer.StochasticQNGD):
    r"""
    SR optimizer for natural excited state determinants.

    The Quantax stochastic optimizer stack still owns the SR equation solve, updater
    buffers, solver defaults, checkpointing, and non-holomorphic bookkeeping. This
    class provides only the NES-specific centered local-energy vector and
    determinant logarithmic Jacobian.
    """

    def __init__(
        self,
        states: NaturalStateSet,
        hamiltonian: qtx.operator.Operator,
        imag_time: bool = True,
        solver: Optional[Callable[[jax.Array, jax.Array], jax.Array]] = None,
        file: Union[None, str, Path, BinaryIO] = None,
        updater: Optional[qtx.optimizer.Updater] = None,
        diagnostics: bool = False,
        diagnostics_every: int = 5,
    ):
        r"""
        :param updater:
            The update strategy, default to `~quantax.optimizer.PlainUpdater`, i.e. plain
            SR. Pass a configured `~quantax.optimizer.Adam` to get the AdamSR behaviour;
            its hyperparameters belong on that object rather than here, since they are
            meaningless for the other updaters.

        :param diagnostics:
            Whether to record solve-conditioning scalars in
            `~qhuantax.nes.NaturalExcitedSR.diagnostics`. Inert by default.

        :param diagnostics_every:
            How often to refresh the expensive entries (the Jacobian Gram spectrum).
            The cheap entries refresh every step.
        """
        if not isinstance(states, NaturalStateSet):
            raise TypeError("NaturalExcitedSR expects a NaturalStateSet.")
        if states.nparams is None:
            raise TypeError(
                "NaturalExcitedSR requires all trainable member states to expose "
                "`nparams`."
            )
        for index, state in enumerate(states.states):
            # Frozen states never contribute a Jacobian block, so they are allowed to
            # be any state type; everything else must be consistent across the set,
            # since all members share the psi matrix.
            if states.trainable[index] and not hasattr(state, "jacobian"):
                raise TypeError(
                    "NaturalExcitedSR requires trainable member states with "
                    f"`jacobian`; state {index} has none."
                )
            if not hasattr(state, "vs_type"):
                raise TypeError(
                    "NaturalExcitedSR requires member states with Quantax "
                    f"`vs_type`; state {index} has none."
                )
            if state.vs_type != states.vs_type:
                raise ValueError(
                    "All member states must have the same `vs_type`; "
                    f"state {index} differs from state 0."
                )
            if state.dtype != states.dtype:
                raise ValueError(
                    "All member states must have the same parameter dtype; "
                    f"state {index} differs from state 0."
                )

        grad = NaturalTraceEnergyGrad(hamiltonian)
        if updater is None:
            updater = qtx.optimizer.PlainUpdater()
        super().__init__(
            states,
            grad,
            imag_time=imag_time,
            solver=solver,
            updater=updater,
            file=file,
        )
        self._Omean = None
        self._diagnostics_enabled = diagnostics
        self._diagnostics_every = diagnostics_every
        self._diagnostics_calls = 0
        self._diagnostics = {}

    @property
    def states(self) -> NaturalStateSet:
        return self.state

    @property
    def Omean(self) -> Optional[jax.Array]:
        return self._Omean

    @property
    def Eloc_max_dev(self) -> Optional[jax.Array]:
        r"""Largest deviation of a single sample's local energy, see the gradient source."""
        return getattr(self._grad, "Eloc_max_dev", None)

    # Gram eigenvalues below this fraction of the largest are treated as unresolved.
    # Comfortably above the float32 eigensolver noise floor (~1e-8 relative), while still
    # a tight rank criterion.
    _RANK_RTOL = 1e-6

    @property
    def diagnostics(self) -> dict:
        r"""
        Conditioning scalars for the last `~NaturalExcitedSR.get_Obar`, empty unless
        ``diagnostics`` was enabled.

        ``obar_fro2`` refreshes every step; ``sigma_max``, ``sigma_min`` and ``rank_eff``
        only every ``diagnostics_every`` steps, since they need the Jacobian Gram matrix.
        """
        return dict(self._diagnostics)

    def _record_diagnostics(self, Obar: jax.Array) -> None:
        r"""
        Measure how well conditioned the SR system is.

        :math:`\bar O` is ``(nsamples, nparams)`` with ``nsamples << nparams``, so the
        solve runs through the ``nsamples x nsamples`` Gram matrix
        :math:`T = \bar O \bar O^\dagger`, whose eigenvalues are the squared singular
        values. ``Tr(T) = \|\bar O\|_F^2`` also sets the solver's diagonal shift, so it is
        recorded every step; the spectrum itself costs an extra Gram formation
        (``O(nsamples^2 nparams)``) and is therefore sampled.

        The analytic gain bound ``1/(2 sqrt(eps))`` is not a substitute: Adam's second
        ``core_solve`` runs on ``Obar / V``, so its trace, shift and bound all differ, and
        ``V`` is exactly the quantity under suspicion.

        .. note::

            :math:`\bar O` is centered, so its column sums vanish and the all-ones vector
            is *exactly* a left null vector -- one Gram eigenvalue is zero by construction.
            The rank threshold is therefore relative to the largest eigenvalue, and
            ``sigma_min`` reports the smallest direction above it. An absolute threshold
            would sit below the eigensolver's noise floor and pin ``rank_eff`` at
            ``nsamples`` in every run.
        """
        self._diagnostics["obar_fro2"] = jnp.sum(jnp.abs(Obar) ** 2)

        refresh = self._diagnostics_calls % self._diagnostics_every == 0
        self._diagnostics_calls += 1
        if not refresh:
            # Cleared rather than left stale: a repeated value would read as "measured and
            # unchanged" instead of "not measured".
            self._diagnostics.update(
                sigma_max=float("nan"), sigma_min=float("nan"), rank_eff=float("nan")
            )
            return

        eigvals = jnp.linalg.eigvalsh(Obar @ Obar.conj().T)
        eigvals = jnp.clip(eigvals, 0.0)  # tiny negatives are roundoff
        resolved = eigvals > self._RANK_RTOL * eigvals[-1]
        rank_eff = jnp.count_nonzero(resolved)
        self._diagnostics["sigma_max"] = jnp.sqrt(eigvals[-1])
        self._diagnostics["sigma_min"] = jnp.sqrt(eigvals[-rank_eff])
        self._diagnostics["rank_eff"] = rank_eff

    def get_Ebar(self, samples: Samples) -> jax.Array:
        r"""Compute Quantax-style centered/scaled local energies for the NES determinant."""
        return self._grad.ebar(self.states, samples)

    def get_Obar(self, samples: Samples) -> jax.Array:
        r"""
        Compute determinant logarithmic Jacobians in member-state parameter order.

        For :math:`A[i, a] = \psi_a(s_i)`, the derivative with respect to state ``a``'s
        parameters is :math:`\sum_i (A^{-1})_{ai} A_{ia} O_a(s_i)`, with :math:`O_a` the
        state's own logarithmic Jacobian. Only trainable states contribute a column
        block, but the :math:`A^{-1}` prefactor is built from the *full*
        ``Nstates`` x ``Nstates`` matrix -- the members are coupled through the
        determinant, so a frozen state still shapes the coefficients of the trainable
        blocks. Freezing therefore drops columns from :math:`\bar O` without altering
        the ansatz or approximating the remaining gradients.
        """
        tuple_spins = jnp.asarray(samples.spins)
        nsamples, Nstates, Nmodes = tuple_spins.shape
        if Nstates != self.states.Nstates or Nmodes != self.states.Nmodes:
            raise ValueError(
                "Expected sample spins with shape "
                f"(nsamples, {self.states.Nstates}, {self.states.Nmodes}), got "
                f"{tuple_spins.shape}."
            )

        A_scaled, _ = _scaled_psi_matrix(self.states, tuple_spins)
        Ainv = jnp.linalg.pinv(A_scaled)

        # `jacobians` skips frozen states, so the yield order is not the member-state
        # order: pair each Jacobian with its own index in the full set.
        trainable_indices = [
            index for index, flag in enumerate(self.states.trainable) if flag
        ]

        blocks = []
        for index, jacobian in zip(trainable_indices, self.states.jacobians(tuple_spins)):
            coeff = Ainv[:, index, :] * A_scaled[:, :, index]
            blocks.append(
                jnp.einsum(
                    "ni,nip->np", coeff, jacobian.reshape(nsamples, Nstates, -1)
                )
            )
            # Drop the reference before the generator computes the next Jacobian,
            # otherwise two of them overlap on device.
            del jacobian

        Omat = jnp.concatenate(blocks, axis=-1)
        del blocks
        if samples.reweight_factor is None:
            reweight = jnp.ones(samples.nsamples)
        else:
            reweight = samples.reweight_factor
        self._Omean = jnp.mean(Omat * reweight[:, None], axis=0)
        factor = jnp.sqrt(reweight / samples.nsamples)[:, None]
        Obar = (Omat - jnp.mean(Omat, axis=0, keepdims=True)) * factor
        if self._diagnostics_enabled:
            self._record_diagnostics(Obar)
        return Obar
