from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Callable, Optional, Union

import jax
import jax.numpy as jnp
import quantax as qtx
from quantax.global_defs import get_default_dtype
from quantax.sampler import Samples
from quantax.utils import LogArray, PsiArray, to_distributed_array

from .state_set import NaturalStateSet, _stack_psi_arrays


def _scale_psi_matrix(psi_matrix: PsiArray) -> tuple[jax.Array, jax.Array]:
    r"""
    Apply the per-row stabilization to an evaluated NES matrix.

    If ``A[i, a] = psi_a(s_i)`` and ``D[i, i] = exp(row_shift[i])``, this returns
    ``A_scaled = D^{-1} A``. The same row scaling must be used for ``B`` in the
    local-energy trace, so that ``solve(A_scaled, D^{-1} B) = A^{-1} B``.

    It takes the matrix rather than ``(states, spins)`` because the local energy
    already holds the amplitudes it is built from, and must not evaluate them twice.
    """
    psi_matrix = LogArray.from_value(psi_matrix)
    row_shift = jnp.max(psi_matrix.logabs, axis=-1)
    row_shift = jnp.where(jnp.isfinite(row_shift), row_shift, 0.0)
    scaled = psi_matrix.sign * jnp.exp(psi_matrix.logabs - row_shift[..., None])
    return scaled, row_shift


class NaturalTraceEnergyGrad(qtx.optimizer.EnergyGrad):
    r"""
    NES trace gradient, optionally against :math:`H + \lambda P`.

    The penalty is kept as its own operator rather than folded into ``hamiltonian``,
    so that the bare :math:`\left< H \right>` and the penalty are recorded alongside
    the quantity being minimized. The two operators reach the same connected
    configurations as their sum, but Quantax pays a per-``Oloc`` overhead of order
    10--20% for the second call; ``penalty_every`` amortizes it.
    """

    _Eloc_max_dev = None
    _energy_H = None
    _VarE_H = None
    _penalty_value = None
    _VarPenalty = None
    _nonfinite = None

    def __init__(
        self,
        hamiltonian: qtx.operator.Operator,
        penalty: Optional[qtx.operator.Operator] = None,
        penalty_coeff: float = 0.0,
        penalty_every: int = 1,
    ):
        r"""
        :param penalty:
            Operator to measure, and -- when ``penalty_coeff`` is nonzero -- to add to
            the Hamiltonian as ``penalty_coeff * penalty``, e.g. :math:`L^- L^+` to
            lift the :math:`L > 0` levels out of the target window.

        :param penalty_coeff:
            The multiplier :math:`\lambda`. Zero (the default) reproduces the plain
            :math:`\mathrm{tr}(S^{-1} H)` gradient exactly, and leaves ``penalty``
            as a pure measurement.

        :param penalty_every:
            How often to resolve :math:`\left< H \right>` and :math:`\left< P \right>`
            separately. ``0`` never does, so ``penalty`` is never evaluated and the run
            costs exactly what it would without one. ``1`` does it every step, for the
            cost of the second ``Oloc`` call. Above ``1``, the intermediate steps
            evaluate the summed operator instead -- the same gradient either way, but
            `energy_H` and `penalty_value` read ``nan`` in between.

        Measuring and penalizing are independent: ``penalty_coeff=0`` with
        ``penalty_every=1`` labels the members by :math:`\left< P \right>` without
        biasing them, and ``penalty_every=0`` penalizes without measuring.
        """
        super().__init__(hamiltonian)
        self._penalty_coeff = float(penalty_coeff)
        self._penalty_every = max(0, int(penalty_every))
        self._penalty_op = penalty if self._penalty_every else None
        # Only built when it is actually reached, since summing the operators is not
        # free -- and with a zero coefficient the sum is just the Hamiltonian.
        self._merged_op = (
            hamiltonian + self._penalty_coeff * penalty
            if penalty is not None and self._penalty_coeff and self._penalty_every != 1
            else None
        )
        self._steps = 0

    @property
    def penalty(self) -> Optional[qtx.operator.Operator]:
        """The measured penalty operator, ``None`` when ``penalty_every`` is 0."""
        return self._penalty_op

    @property
    def penalty_coeff(self) -> float:
        r"""The multiplier :math:`\lambda` on `~NaturalTraceEnergyGrad.penalty`."""
        return self._penalty_coeff

    @property
    def penalty_every(self) -> int:
        """How often `energy_H` and `penalty_value` are refreshed."""
        return self._penalty_every

    @property
    def energy_H(self) -> Optional[jax.Array]:
        r"""
        :math:`\mathrm{tr}(S^{-1} H)` of the current step, without the penalty.

        Equal to `energy` whenever nothing is added to the Hamiltonian, ``nan`` on a
        step that ``penalty_every`` skipped. This is the quantity to compare against
        ED: with :math:`\lambda \neq 0`, `energy` sums the *shifted* levels.
        """
        return self._energy_H

    @property
    def VarE_H(self) -> Optional[jax.Array]:
        r"""
        Sample variance of :math:`\mathrm{tr}(S^{-1} H)` over the last batch.

        The zero-variance property belongs to :math:`H` alone, so this is the one that
        has to fall to zero at convergence; `VarE` also carries
        :math:`\lambda^2 \mathrm{Var}(P)` and their covariance, which is what makes it
        the wrong thing to read when :math:`\lambda` is large. ``nan`` on a step that
        ``penalty_every`` skipped.
        """
        return self._VarE_H

    @property
    def penalty_value(self) -> Optional[jax.Array]:
        r"""
        :math:`\mathrm{tr}(S^{-1} P)` of the current step.

        For :math:`P = L^- L^+` at :math:`L_z = 0` this is :math:`\sum_k L_k(L_k+1)`
        over the members, so it reaches zero exactly when every member has converged
        to :math:`L = 0` -- the convergence check for the penalty itself. ``nan``, not
        zero, on a step where it was not measured.
        """
        return self._penalty_value

    @property
    def VarPenalty(self) -> Optional[jax.Array]:
        r"""Sample variance of :math:`\mathrm{tr}(S^{-1} P)` over the last batch."""
        return self._VarPenalty

    @property
    def Eloc_max_dev(self) -> Optional[jax.Array]:
        r"""
        :math:`\max_s |E_{loc}(s) - \bar E|` over the last batch, discarded samples
        excluded.

        Compared against ``sqrt(VarE)``, this says whether :math:`\bar \epsilon` is
        dominated by a single sample -- the SR right-hand side is used one realization at
        a time, not in expectation, so a lone outlier translates directly into a step.
        """
        return self._Eloc_max_dev

    @property
    def nonfinite(self) -> Optional[jax.Array]:
        r"""
        How many samples of the last batch were discarded for a non-finite local energy.

        See `~NaturalTraceEnergyGrad.ebar` for why they arise and why dropping them is
        cheaper than it looks. Persistently nonzero means the members are running close
        enough to their nodes that the float32 amplitude ratio saturates, which is worth
        knowing even though the estimator absorbs it.
        """
        return self._nonfinite

    def local_energies(
        self, states: NaturalStateSet, samples: Samples, resolve: bool = True
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        r"""
        Return ``(tr(S^-1 (H + lambda P)), tr(S^-1 H), tr(S^-1 P))``.

        Both ``Oloc`` and the trace are linear in the operator, so with ``resolve`` the
        two traces come out of one pass: they share the psi matrix, the single
        ``pinv``, and the connected configurations, since Quantax's
        ``Operator.__add__`` concatenates term lists without merging them and
        ``_Oloc`` emits one connected configuration per term. ``(H + lambda P).Oloc``
        therefore visits exactly the configurations that ``H.Oloc`` and ``P.Oloc``
        visit between them -- measured equal, not assumed.

        What the second call does duplicate is Quantax's per-``Oloc`` overhead: its
        internal evaluation of ``psi(s)`` on the sampled configurations, the host sync
        in ``_get_conn_size``, and the rounding of the connected batch up to
        ``max_parallel``. The first of those is removed here by computing the
        amplitudes once per member and passing them down through `Samples`, so they
        serve as the psi matrix, the column prefactor and the denominator inside both
        calls; the rest is what ``penalty_every`` exists to amortize.

        Without ``resolve`` the summed operator is evaluated instead, for the same
        gradient in one call, and the two components come back as ``nan``.
        """
        tuple_spins = jnp.asarray(samples.spins)
        nsamples, Nstates, Nmodes = tuple_spins.shape
        if Nstates != states.Nstates or Nmodes != states.Nmodes:
            raise ValueError(
                "Expected sample spins with shape "
                f"(nsamples, {states.Nstates}, {states.Nmodes}), got "
                f"{tuple_spins.shape}."
            )

        # `_Oloc` distributes its own input, so do it up front and keep the amplitudes
        # on the same sharding as the configurations they came from.
        flat_spins = to_distributed_array(tuple_spins.reshape(nsamples * Nstates, Nmodes))
        psi_columns = [state(flat_spins) for state in states.states]
        psi_matrix = _stack_psi_arrays(psi_columns, axis=-1)
        S_scaled, row_shift = _scale_psi_matrix(
            psi_matrix.reshape(nsamples, Nstates, Nstates)
        )

        resolve = resolve and self._penalty_op is not None
        if resolve:
            operators = [self.hamiltonian, self._penalty_op]
        else:
            # `_merged_op` is None whenever the sum degenerates to the Hamiltonian.
            operators = [self._merged_op or self.hamiltonian]
        columns = [[] for _ in operators]

        for state, psi_column in zip(states.states, psi_columns):
            flat_samples = Samples(flat_spins, psi_column)
            psi = jnp.asarray(psi_column)
            for operator, operator_columns in zip(operators, columns):
                Oloc = operator.Oloc(state, flat_samples).astype(get_default_dtype())
                operator_columns.append(Oloc * psi)

        Sinv = jnp.linalg.pinv(S_scaled)
        row_scale = jnp.exp(-row_shift[..., None])

        traces = []
        for operator_columns in columns:
            matrix = jnp.stack(operator_columns, axis=-1).reshape(
                nsamples, Nstates, Nstates
            )
            Sinv_O = Sinv @ (matrix * row_scale)
            traces.append(
                jnp.trace(Sinv_O, axis1=-2, axis2=-1).astype(get_default_dtype())
            )

        if not resolve:
            # `nan` rather than 0 for the penalty: it was not measured, which is not
            # the same statement as "it vanished".
            Eloc = traces[0]
            unresolved = jnp.full_like(Eloc, jnp.nan)
            # With nothing added to the Hamiltonian the optimized energy is the bare one.
            return Eloc, (unresolved if self._penalty_coeff else Eloc), unresolved

        Eloc_H, Eloc_P = traces
        return Eloc_H + self._penalty_coeff * Eloc_P, Eloc_H, Eloc_P

    def local_energy(self, states: NaturalStateSet, samples: Samples) -> jax.Array:
        """The optimized local energy, i.e. the first entry of `local_energies`."""
        return self.local_energies(states, samples)[0]

    def ebar(self, states: NaturalStateSet, samples: Samples) -> jax.Array:
        r"""
        Centered, reweighted local energies, with non-finite samples discarded.

        A configuration sitting close to a node of one member makes Quantax's ``_Oloc``
        overflow: it materializes :math:`\psi(s')/\psi(s)` in the network's dtype
        (float32 here), so the ratio saturates once the log amplitudes differ by more
        than ``log(finfo.max)``, even though the ``Oloc * psi`` product that
        `local_energies` forms from it is finite -- the divergence cancels analytically,
        just one step too late to survive the intermediate. Sampling
        :math:`|\det A|^\mathrm{reweight}` with ``reweight < 2`` deliberately visits
        those tails more often, which is the point, so this has to be survivable rather
        than avoided.

        Such a sample carries a reweight factor of order ``VarE * nsamples /
        Eloc_max_dev**2``, i.e. :math:`10^{-3}` or less in practice, so it contributes
        nothing to any of the weighted estimators below and dropping it is free. Keeping
        it is not: the centering at the end is an *unweighted* mean, so one ``nan``
        propagates to every entry of the returned vector, the SR solve returns a
        ``nan`` step, and a converged run dies on a single unlucky draw.
        """
        resolve = bool(self._penalty_every) and self._steps % self._penalty_every == 0
        self._steps += 1
        Eloc, Eloc_H, Eloc_P = self.local_energies(states, samples, resolve=resolve)

        if samples.reweight_factor is None:
            reweight_factor = jnp.ones(samples.nsamples)
        else:
            reweight_factor = samples.reweight_factor

        # Taken from `Eloc` alone, so that the components stay `nan` on the steps
        # `penalty_every` skips -- there they are unmeasured, not overflowed, and
        # `Eloc` is finite throughout. On a resolved step an overflow in either
        # component reaches `Eloc` through the sum, so the masks agree.
        finite = jnp.isfinite(Eloc)
        nfinite = jnp.count_nonzero(finite)
        self._nonfinite = samples.nsamples - nfinite
        # Zero weight is what actually removes a sample; zeroing the value only keeps
        # `nan * 0 = nan` from resurrecting it in the reductions.
        reweight_factor = jnp.where(finite, reweight_factor, 0.0)
        Eloc = jnp.where(finite, Eloc, 0.0)
        Eloc_H = jnp.where(finite, Eloc_H, 0.0)
        Eloc_P = jnp.where(finite, Eloc_P, 0.0)

        Emean = jnp.mean(Eloc * reweight_factor)
        self._energy = Emean.real
        Evar = jnp.abs(Eloc - Emean) ** 2
        self._VarE = jnp.mean(Evar * reweight_factor).real
        # Free: `Evar` is already materialized, so the max is one more reduction over it.
        # Discarded samples sit at `Eloc = 0`, whose deviation is a spurious `|Emean|`.
        self._Eloc_max_dev = jnp.sqrt(jnp.max(jnp.where(finite, Evar, 0.0)))

        # Same batch, same weights: reductions over arrays already in hand.
        Hmean = jnp.mean(Eloc_H * reweight_factor)
        self._energy_H = Hmean.real
        self._VarE_H = jnp.mean(jnp.abs(Eloc_H - Hmean) ** 2 * reweight_factor).real
        Pmean = jnp.mean(Eloc_P * reweight_factor)
        self._penalty_value = Pmean.real
        self._VarPenalty = jnp.mean(jnp.abs(Eloc_P - Pmean) ** 2 * reweight_factor).real

        # Over the surviving samples: `jnp.mean` would divide by `nsamples` and pull the
        # centre towards zero by whatever the discarded ones would have contributed.
        Eloc -= jnp.sum(Eloc) / nfinite
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
        penalty: Optional[qtx.operator.Operator] = None,
        penalty_coeff: float = 0.0,
        penalty_every: int = 1,
        diagnostics: bool = False,
        diagnostics_every: int = 5,
    ):
        r"""
        :param penalty:
            Operator optimized as ``hamiltonian + penalty_coeff * penalty``, kept
            separate so that `~qhuantax.nes.NaturalExcitedSR.energy_H` and
            `~qhuantax.nes.NaturalExcitedSR.penalty_value` are available alongside
            the optimized energy. Pass :math:`L^- L^+` to select an :math:`L` sector.

        :param penalty_coeff:
            The multiplier on ``penalty``; zero leaves the gradient untouched.

        :param penalty_every:
            How often to separate the two, see
            `~NaturalTraceEnergyGrad.local_energies`. The gradient does not depend on
            it; raising it trades resolution of the components for the ~10-20% that
            the second ``Oloc`` call costs.

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

        grad = NaturalTraceEnergyGrad(
            hamiltonian, penalty, penalty_coeff, penalty_every
        )
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
        self._nonfinite_grad = 0
        self._finite_grad = True
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

    @property
    def nonfinite(self) -> Optional[jax.Array]:
        r"""Samples discarded for a non-finite local energy, see the gradient source."""
        return getattr(self._grad, "nonfinite", None)

    @property
    def nonfinite_grad(self) -> jax.Array:
        r"""
        Samples discarded from the last `~NaturalExcitedSR.get_Obar` for a non-finite
        logarithmic derivative.

        Counted separately from `~NaturalExcitedSR.nonfinite` because the two mean
        different things: that one says a local energy overflowed, this one says a
        member state was sampled close enough to a node for its log-derivative to
        diverge while its amplitude stayed representable. This one is the more
        informative of the two -- it rises with the peakedness of the Jacobian
        spectrum, so a run where it climbs is a run whose learning rate is outpacing
        the ansatz.
        """
        return self._nonfinite_grad

    @property
    def energy_H(self) -> Optional[jax.Array]:
        r"""
        The energy without the penalty, see the gradient source.

        `~qhuantax.nes.NaturalExcitedSR.energy` and
        `~qhuantax.nes.NaturalExcitedSR.VarE` describe the quantity actually being
        minimized, :math:`\mathrm{tr}(S^{-1}(H + \lambda P))`; this one is the
        physical energy, and the two coincide when ``penalty_coeff`` is zero.
        """
        return getattr(self._grad, "energy_H", None)

    @property
    def VarE_H(self) -> Optional[jax.Array]:
        r"""Variance of the energy without the penalty, see the gradient source."""
        return getattr(self._grad, "VarE_H", None)

    @property
    def penalty_value(self) -> Optional[jax.Array]:
        r""":math:`\mathrm{tr}(S^{-1} P)`, see the gradient source."""
        return getattr(self._grad, "penalty_value", None)

    @property
    def VarPenalty(self) -> Optional[jax.Array]:
        r"""Sample variance of :math:`\mathrm{tr}(S^{-1} P)`, see the gradient source."""
        return getattr(self._grad, "VarPenalty", None)

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

        A_scaled, _ = _scale_psi_matrix(self.states.psi_matrix(tuple_spins))
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

        # The same near-node configurations that `ebar` guards against, reaching the
        # solve by the other side -- and this side has no cancellation to rely on.
        # `Oloc * psi` keeps a local energy finite however small the amplitude gets, but
        # Obar *is* the logarithmic derivative: ``d log|psi| / d theta`` grows like the
        # reciprocal distance to the node while the amplitude only shrinks. So a tuple
        # can give a perfectly finite `Eloc` and a non-finite Jacobian row on the same
        # sweep, and the centering below then spreads that one row over all of Obar.
        # Zeroing the row removes the sample from the Gram matrix outright.
        finite = jnp.all(jnp.isfinite(Omat), axis=1)
        nfinite = jnp.count_nonzero(finite)
        self._nonfinite_grad = samples.nsamples - nfinite
        self._finite_grad = finite
        Omat = jnp.where(finite[:, None], Omat, 0.0)
        reweight = jnp.where(finite, reweight, 0.0)

        self._Omean = jnp.mean(Omat * reweight[:, None], axis=0)
        factor = jnp.sqrt(reweight / samples.nsamples)[:, None]
        # Over the surviving rows, as in `ebar`: `jnp.mean` would divide by `nsamples`
        # and pull the centre towards zero by whatever the discarded rows would have
        # contributed.
        Obar = (Omat - jnp.sum(Omat, axis=0, keepdims=True) / nfinite) * factor
        if self._diagnostics_enabled:
            self._record_diagnostics(Obar)
        return Obar

    def get_step(self, samples: Samples) -> jax.Array:
        r"""
        Solve the SR equation, reconciling the two sides' discarded samples.

        `get_Obar` can discard a sample that `~NaturalTraceEnergyGrad.ebar` kept, since a
        node diverges the logarithmic derivative while leaving the local energy finite.
        A zeroed Obar row already drops that sample from the Gram matrix, but its
        `Ebar` entry has to go with it, or the solver is asked to account for a residual
        along a direction it no longer holds.

        `Ebar` keeps the centering `ebar` gave it, computed before this second mask was
        known. The leftover is a component along the all-ones vector, which Obar's own
        centering makes an exact left null vector -- so the solve annihilates it.
        """
        Ebar = self.get_Ebar(samples)
        Obar = self.get_Obar(samples)
        Ebar = jnp.where(self._finite_grad, Ebar, 0.0)
        step, self._buffers = self.solve(Obar, Ebar, self._buffers)
        return step
