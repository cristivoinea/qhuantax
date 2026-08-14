import argparse
import json
from datetime import datetime
from pathlib import Path

import jax.numpy as jnp

import numpy as np
import quantax as qtx
import scipy as sp
from qhuantax.quantumhall_transformer import Transformer

from qhuantax.nes import (
    NaturalExcitedSR,
    NaturalLzDetSampler,
    NaturalStateSet,
    dense_reduced_matrices,
)
from qhuantax.quantumhall_operators import (
    GetLpTerms,
    GetSpinfulDenIntTerms,
    GetSpinfulPolTerms,
)
from qhuantax.quantumhall_meanfield import backflow_coeffs
from qhuantax.quantumhall_models import MultiDetBackflow, MultiPfBackflow
from qhuantax.quantumhall_symmetries import FlavourPermQH, IdentityQH, ParticleHoleQH
from qhuantax.quantumhall_utils import (
    MF_BACKFLOW_SCALE,
    adaptive_learning_rate_inv,
    generate_spin_configs,
    scale_backflow,
)


S1 = np.array([[1, 0], [0, 0]])
S2 = np.array([[0, 0], [0, 1]])
SX = np.array([[0, 1], [1, 0]])



def build_state(index, L, N, d, nb, nh, symm, pf_backflow, U, coeffs=None, orbital_noise=5e-2, rng=np.random.default_rng(), param_file=None, backflow_scale=1.0, nterms=1):
    """One NES member.

    `coeffs` switches on the multi-determinant mean-field reference: `U` is then the determinant
    stack shared by every member (shape `(ndets, 2L, N)`) and `coeffs` is this member's vector over
    it, so the members differ by their coefficients rather than by noise, and the term count comes
    from that vector. Without it, `nterms` sets how many perturbed copies of `U` this member sums.
    """
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
            model = qtx.model.DetBackflow(net, U0=jnp.asarray(U_state[0]), d=d)
        else:
            nets = [Transformer(nblocks=nb, d=d, heads=nh, final_sum=False) for _ in U_state]
            model = MultiDetBackflow(
                nets,
                U0=jnp.asarray(U_state),
                # `DetBackflow` divides each determinant by its own std, which would destroy the
                # relative weights; `backflow_coeffs` undoes exactly that.
                coeffs=jnp.asarray(backflow_coeffs(U_state, coeffs)),
                d=d,
            )
        return qtx.state.Variational(scale_backflow(model, backflow_scale), param_file=param_file,
                                     symm=symm, max_parallel=16384, use_ref=False)

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
        model = multi(nets=nets, U0=jnp.stack([jnp.asarray(u) for u in U0s]),
                      coeffs=jnp.ones(nterms) / nterms, d=d)
    elif pf_backflow:
        model = qtx.model.PfBackflow(nets[0], U0=U0s[0], d=d)
    else:
        model = qtx.model.DetBackflow(nets[0], U0=U0s[0], d=d)

    # `param_file` replaces every leaf of the model, U0 and W included, so the orbital noise
    # and the backflow scale above are irrelevant whenever one is given.
    return qtx.state.Variational(scale_backflow(model, backflow_scale), param_file=param_file,
                                 symm=symm, max_parallel=16384, use_ref=False)



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-n", action="store", required=True,
                    help="number of particles")
parser.add_argument("-s", action="store", required=True,
                    help="number of orbitals in the system (2s)")
parser.add_argument("--nbr-states", action="store", default=2, 
                    help="number of desired states")
parser.add_argument("--lz-sect", action="store", required=True,
                    help="Lz symmetry sector")
parser.add_argument("--z2-sect", action="store", default=0,
                    help="Z2 symmetry sector")
parser.add_argument("--ph-sect", action="store", default=0,
                    help="PH symmetry sector (without spin flip)")

parser.add_argument("--init-state-file", action="store", default=None,
                    help="path to a .eqx state used as starting parameters of the ground state")
parser.add_argument("--incremental", action="store_true", default=False,
                    help="grow the state set one state at a time, training --nbr-sweeps per stage")
parser.add_argument("--warmup-sweeps", action="store", default=None,
                    help="opening sweeps of each stage with already-trained states held fixed; "
                         "defaults to --nbr-sweeps//2")
parser.add_argument("--updater", action="store", default="plain",
                    choices=("plain", "adam", "spring", "march"),
                    help="SR update strategy; the learning rate does not transfer between these")
parser.add_argument("--norm-clip", action="store", default=None,
                    help="cap on the norm of the step accumulated into momentum; needs an "
                         "updater with momentum, so not available for --updater plain")
parser.add_argument("--diagnostics", action="store_true", default=False,
                    help="write per-sweep step norms and SR conditioning to data_diagnostics_*.txt")
parser.add_argument("--diagnostics-every", action="store", default=5,
                    help="how often to refresh the Jacobian Gram spectrum in the diagnostics file")
parser.add_argument("--mean-field", action="store", default=None,
                    help="path to a .npz written by FuzzySphereMeanField.py: its determinant "
                         "stack becomes the shared U0 of every member, and its k-th "
                         "Rayleigh-Ritz vector initialises member k")


parser.add_argument("--exact-diag", action="store_true", default=False,
                    help="perform exact diagonalization and track energy, energy variance and overlap with ground state")
parser.add_argument("--lmlp-coeff", action="store", default=0,
                    help="coefficient in front of the L^- L^+ term added to the Hamiltonian")
parser.add_argument("--lmlp-freq", action="store", default=1,
                    help="how often to resolve <H> and <L^- L^+> separately rather than only "
                         "their optimized sum; 1 (every sweep) costs the ~10-20% overhead of a "
                         "second Oloc call, and the gradient is the same either way")

parser.add_argument("--multi", action="store", default=None,
                    help="number of determinants each state is built from, either one value for "
                         "all of them or one per state, e.g. '1,1,2' -- which is what a mean-field "
                         "file written without --orthogonalize gives, since there each state "
                         "carries only its own reference. Set by the mode spec, so this is only "
                         "cross-checked against the file")
parser.add_argument("--pf-backflow", action="store_true", default=False,
                    help="change the ansatz structure from PfBackflow to DetBackflow")
parser.add_argument("--nbr-heads", action="store", default=4,
                    help="number of attention heads in the transformer")
parser.add_argument("--attn-dim", action="store", default=16,
                    help="attention dimension of the transformer")
parser.add_argument("--nbr-blocks", action="store", default=1,
		    help="number of layers in the transformer")
parser.add_argument("--nbr-sweeps", action="store", default=500,
		    help="number of training iterations")
parser.add_argument("--nbr-samples", action="store", default=256,
		    help="number of samples for each training iteration")
parser.add_argument("--lr", action="store", default=1e-2,
		    help="starting value of the learning rate")
parser.add_argument("--reweight", action="store", default=2.0,
		    help="reweight factor for training sampling")
parser.add_argument("--run-id", action="store", default=1,
                    help="")
parser.add_argument("--path", action="store", required=True,
                    help="path")

args = vars(parser.parse_args())

N = int(args["n"])
L = int(args["s"])+1
nstates = int(args["nbr_states"])
lz = int(args["lz_sect"])
z2 = int(args["z2_sect"])
ph = int(args["ph_sect"])
id = int(args["run_id"])
path = str(args["path"])
run_id = f"nes_n_{N}_2s_{L-1}_lz_{lz}_z2_{z2}_ph_{ph}_id0{id}"

init_state_file = args["init_state_file"]
incremental = bool(args["incremental"])
if incremental and init_state_file is None:
    parser.error("--incremental requires --init-state-file for the K=1 state")
mf_file = args["mean_field"]


do_ED = bool(args["exact_diag"])
LmLp_coeff = float(args["lmlp_coeff"])
LmLp_freq = int(args["lmlp_freq"])

# "2" means every state, "1,1,2" one entry per state. Without --mean-field this builds that many
# determinants per state; with one it is only cross-checked against the file.
nterms_arg = (None if args["multi"] is None
              else [int(v) for v in str(args["multi"]).replace(",", " ").split()])
if nterms_arg is not None and len(nterms_arg) not in (1, nstates):
    parser.error(f"--multi needs 1 or {nstates} entries, got {len(nterms_arg)}")
nterms = ([1] * nstates if nterms_arg is None
          else nterms_arg * nstates if len(nterms_arg) == 1 else list(nterms_arg))
pf_backflow = bool(args["pf_backflow"])
nsweeps = int(args["nbr_sweeps"])
nsamples = int(args["nbr_samples"])
nb = int(args["nbr_blocks"])
nh = int(args["nbr_heads"])
d = int(args["attn_dim"])

lr0 = float(args["lr"])
t0 = 10*N
rw = float(args["reweight"])
model_type = "DetBackflow"

warmup_sweeps = nsweeps // 2 if args["warmup_sweeps"] is None else int(args["warmup_sweeps"])
diagnostics = bool(args["diagnostics"])
diagnostics_every = int(args["diagnostics_every"])

updater_name = str(args["updater"])
norm_clip = None if args["norm_clip"] is None else float(args["norm_clip"])
updater = {"plain": qtx.optimizer.PlainUpdater,
           "adam": qtx.optimizer.Adam,
           "spring": qtx.optimizer.Spring,
           "march": qtx.optimizer.March}[updater_name]()


def stage_phases(K):
    """[(trainable_mask, nsweeps), ...] for a stage holding K member states.

    Already-trained states are pinned for the opening `warmup_sweeps`, then everything 
    iis released.
    """
    nfrozen = (K - 1) if incremental else (1 if init_state_file else 0)
    if nfrozen == 0:                       # nothing trained yet to protect
        return [((True,) * K, nsweeps)]

    warm = min(warmup_sweeps, nsweeps)
    phases = []
    if warm > 0:
        phases.append(((False,) * nfrozen + (True,) * (K - nfrozen), warm))
    if nsweeps - warm > 0:
        phases.append(((True,) * K, nsweeps - warm))
    return phases


Path(path).mkdir(parents=True, exist_ok=True)

lattice = qtx.sites.Chain(L=L, boundary=0, particle_type="spinful_fermion", Nparticles=N)

if z2 != 0:
    symm = FlavourPermQH(eigval=z2)
else:
    symm = IdentityQH()
if ph != 0:
    symm = ParticleHoleQH(eigval=ph) @ symm


# quantum Hall Hamiltonian
pspot_inter = np.array([4.75, 1])
pspot_intra = np.array([0])
transverse_fld = -3.16
tms_H = GetSpinfulDenIntTerms(nm = L, ps_pot=2*pspot_inter, mat_a = S1, mat_b = S2)
tms_H += transverse_fld * GetSpinfulPolTerms(nm=L, mat = SX)


# Kept out of `tms_H`: the optimizer takes it as a separate operator, so <H> and
# <L^- L^+> come back separately at every step for the same cost as the sum, and the
# spectrum readout below stays on the unshifted Hamiltonian.
tms_Lp = GetLpTerms(L, 2)
tms_LmLp = tms_Lp.H @ tms_Lp

# Mean-field reference: a determinant stack shared by every member, with one Rayleigh-Ritz
# coefficient vector per state.
mf_coeffs = None
if mf_file is not None:
    with np.load(mf_file) as mf:
        U = np.asarray(mf["U0"])
        mf_coeffs = np.asarray(mf["coeffs"])
        mf_meta = json.loads(str(mf["meta"]))
    if (mf_meta["N"], mf_meta["nm"]) != (N, L):
        raise ValueError(f"{mf_file} was built for N={mf_meta['N']}, 2s+1={mf_meta['nm']}, "
                         f"but this run is N={N}, 2s+1={L}")
    if (mf_meta["lz"], mf_meta["z2"]) != (lz, z2):
        raise ValueError(f"{mf_file} is for sector (lz,z2)=({mf_meta['lz']},{mf_meta['z2']}), "
                         f"but this run is ({lz},{z2})")
    if len(mf_coeffs) < nstates:
        raise ValueError(f"{mf_file} holds {len(mf_coeffs)} states, need {nstates}; rerun "
                         f"FuzzySphereMeanField.py with a larger --nbr-states and more modes")
    # Members keep only the determinants their own coefficient vector uses, so the term count is
    # per state and set by the mode spec. --multi can only agree or disagree: an assertion.
    mf_nterms = [int(np.count_nonzero(c)) for c in mf_coeffs[:nstates]]
    if nterms_arg is not None and nterms != mf_nterms:
        raise ValueError(f"{mf_file} gives its states {mf_nterms} determinants, but --multi "
                         f"asks for {nterms}; the mode spec sets this, so drop --multi")
    nterms = mf_nterms
    print(f"mean field: {U.shape[0]} determinants, {nterms} per state, "
          f"modes {mf_meta['modes']}, "
          f"energies {[round(e, 6) for e in mf_meta['energies'][:nstates]]}, "
          f"residual |<L^2>-L(L+1)| {mf_meta['residual']}")
else:
    U = np.zeros((2*L, N))
    U[:N,:N] = np.eye(N)
    if lz != 0:
        U[0,0] = 0
        U[N,lz] = 1

start_time = datetime.now()

def init_args(index):
    if mf_coeffs is not None:
        # The members already differ by their Rayleigh-Ritz coefficients, so no noise is needed
        # -- and any would spoil the tuned reference. Checked first because it also fixes the
        # model structure, which `param_file` has to match.
        pf = init_state_file if (init_state_file is not None and index == 0) else None
        return dict(coeffs=mf_coeffs[index], orbital_noise=0, param_file=pf,
                    backflow_scale=MF_BACKFLOW_SCALE)
    if init_state_file is not None and index == 0:
        return dict(orbital_noise=0, param_file=init_state_file)
    return dict(orbital_noise=(0 if index == 1 else 1e-1), param_file=None)

member_states = [
    build_state(index, L, N, d, nb, nh, symm, pf_backflow, U, nterms=nterms[index],
                **init_args(index))
    for index in range(nstates)
]
# The last phase of the last stage holds every state, so this is the set the meta block
# and the downstream scripts describe.
state_set = NaturalStateSet(member_states, trainable=stage_phases(nstates)[-1][0])


with open(f"{path}/meta_{run_id}.txt", "w") as f:
  f.write(f"V^inter: {pspot_inter}\n")
  f.write(f"V^intra: {pspot_intra}\n")
  f.write(f"transverse field: {transverse_fld}\n")
  f.write(f"exact diagonalization: {do_ED}\n")
  f.write(f"L^- L^+ coeff: {LmLp_coeff}\n")
  f.write(f"L^- L^+ meas. frequency: {LmLp_freq}\n")
  f.write(f"optimizer: {updater_name}\n")
  f.write(f"norm clip: {norm_clip}\n")
  f.write(f"nbr. iter.: {nsweeps}\n")
  f.write(f"learning rate: {lr0}\n")
  f.write(f"lr schedule: inv\n")
  f.write(f"t0: {t0}\n")
  f.write(f"sampler: DipoleCons\n")
  f.write(f"nbr. samples NN: {nsamples}\n")
  f.write(f"reweight: {rw}\n")
  f.write(f"model: {'PfBackflow' if pf_backflow else 'DetBackflow'}\n")
  f.write(f"net: Transformer\n")
  f.write(f"nbr. blocks: {nb}\n")
  f.write(f"nbr. heads: {nh}\n")
  f.write(f"attndim: {d}\n")
  f.write(f"nbr. params per state: {state_set.nparams_per_state}\n")
  f.write(f"nterms per state: {nterms}\n")
  f.write(f"nbr. states: {nstates}\n")
  f.write(f"init state file: {init_state_file}\n")
  f.write(f"mean field file: {mf_file}\n")
  if mf_coeffs is not None:
      f.write(f"backflow scale: {MF_BACKFLOW_SCALE}\n")
  f.write(f"incremental: {incremental}\n")
  f.write(f"warmup sweeps: {warmup_sweeps}\n")


# `energy`/`VarE` describe tr(S^-1 (H + lambda L^- L^+)), the quantity the optimizer
# actually minimizes; `energy_H` is the physical tr(S^-1 H) to compare against ED, and
# reads nan on the sweeps --lmlp-freq skips. The three coincide with the old two-column
# output when --lmlp-coeff is 0.
energy = qtx.utils.DataTracer()
VarE = qtx.utils.DataTracer()
energy_H = qtx.utils.DataTracer()
LmLp_tracer = qtx.utils.DataTracer()
LmLp_var_tracer = qtx.utils.DataTracer()


DIAGNOSTICS_COLUMNS = (
    "sweep",          # matches the first column of data_energy_*.txt
    "step_norm",      # norm of the solve output, before the learning rate
    "step_max",       # is that norm concentrated in a few parameters or spread out
    "obar_fro2",      # Tr(Obar Obar^H); sets the solver's diagonal shift
    "eloc_max_dev",   # max|Eloc - Emean|; vs sqrt(VarE), is Ebar one-sample-dominated
    "sigma_max",      # largest singular value of Obar
    "sigma_min",      # smallest *resolved* one; Obar is centered so one is exactly zero
    "rank_eff",       # resolved directions, at most nsamples - 1
)


def append_diagnostics(sweep, step, optimizer):
    """Append one line to the diagnostics file.

    Every entry is either the loop counter or a reduction over an array already in hand.
    The last three columns need the Jacobian Gram matrix, so they only refresh every
    ``--diagnostics-every`` sweeps and read `nan` in between.
    """
    diag = optimizer.diagnostics
    row = (
        sweep,
        float(jnp.linalg.norm(step)),
        float(jnp.max(jnp.abs(step))),
        float(diag.get("obar_fro2", np.nan)),
        float(optimizer.Eloc_max_dev),
        float(diag.get("sigma_max", np.nan)),
        float(diag.get("sigma_min", np.nan)),
        float(diag.get("rank_eff", np.nan)),
    )
    with open(f"{path}/data_diagnostics_{run_id}.txt", "a") as f:
        f.write(" ".join(repr(value) for value in row) + "\n")


def train_stage(stage_set, nsweeps_phase, sweep0):
    """Train one phase, appending to the shared tracers.

    The sampler and the optimizer are rebuilt per phase: the sampler bakes ``Nstates``
    into its chain shape, and Quantax sizes the Adam buffers from ``stage_set.nparams``,
    which counts trainable states only. ``sweep0`` keeps the learning-rate schedule
    continuous across the phases of a stage.
    """
    init_configs = generate_spin_configs(
        L, N, lz, nsamples * stage_set.Nstates).reshape(nsamples, stage_set.Nstates, 2 * L)

    sampler = NaturalLzDetSampler(
        stage_set,
        nsamples,
        n_neighbor=np.arange(1, 3),
        initial_spins=init_configs,
        reweight=rw)

    optimizer = NaturalExcitedSR(
        stage_set,
        tms_H,
        updater=updater,
        penalty=tms_LmLp,
        penalty_coeff=LmLp_coeff,
        penalty_every=LmLp_freq,
        diagnostics=diagnostics,
        diagnostics_every=diagnostics_every)

    for i in range(nsweeps_phase):
        samples = sampler.sweep()
        step = optimizer.get_step(samples)

        # Checked before `update` so the saved checkpoints remain the last good state.
        if not (jnp.all(jnp.isfinite(step)) and np.isfinite(optimizer.energy)):
            raise RuntimeError(
                f"Non-finite update at sweep {len(energy.data)} (K={stage_set.Nstates}, "
                f"{stage_set.nparams} trainable params); energy={optimizer.energy}, "
                f"VarE={optimizer.VarE}."
            )

        # Cap what actually moves the parameters. The updaters' own `norm_clip` bounds
        # only the momentum input, not the emitted step, so it cannot do this.
        applied = step
        if norm_clip is not None:
            step_norm = jnp.linalg.norm(step)
            applied = jnp.where(step_norm > norm_clip,
                                step * (norm_clip / step_norm), step)

        lr = adaptive_learning_rate_inv(lr0, t0, sweep0 + i)
        stage_set.update(stage_set.split_step(applied * lr))
        energy.append(optimizer.energy)
        VarE.append(optimizer.VarE)
        energy_H.append(optimizer.energy_H)
        LmLp_tracer.append(optimizer.penalty_value)
        LmLp_var_tracer.append(optimizer.VarPenalty)

        # Columns 0-2 are unchanged, so the existing readers keep working; the penalty
        # columns are appended. `LmLp` is sum_k L_k(L_k+1) over the members, i.e. 0 once
        # every state is L = 0.
        np.savetxt(
            f"{path}/data_energy_{run_id}.txt",
            np.vstack((energy.time, energy.data, VarE.data, energy_H.data,
                       LmLp_tracer.data, LmLp_var_tracer.data)).T,
            header="sweep E_total VarE E_H LmLp VarLmLp",
        )
        if diagnostics:
            append_diagnostics(len(energy.data) - 1, step, optimizer)

        for index, state in enumerate(stage_set.states):
            state.save(f"{path}/state{index}_{run_id}.eqx")


def record_spectrum(K):
    """Diagonalize the reduced (S, H) pencil in the span of the first K states.
    """
    if not do_ED:
        return
    span = NaturalStateSet(member_states[:K])
    S, H_reduced = dense_reduced_matrices(span, tms_H, lz, z2)
    eigval = sp.linalg.eigh(H_reduced, S, eigvals_only=True)
    print("Measured eigenvalues: ", np.sort(eigval))
    with open(f"{path}/data_spectrum_{run_id}.txt", "a") as f:
        f.write(" ".join(str(x) for x in (K, *np.sort(eigval))) + "\n")


if diagnostics:
    with open(f"{path}/data_diagnostics_{run_id}.txt", "w") as f:
        f.write("# " + " ".join(DIAGNOSTICS_COLUMNS) + "\n")
        f.write(f"# sigma_max sigma_min rank_eff refresh every {diagnostics_every} sweeps, "
                "nan otherwise\n")

for K in range(2 if incremental else nstates, nstates + 1):
    sweep0 = 0
    for mask, nsweeps_phase in stage_phases(K):
        stage_set = NaturalStateSet(member_states[:K], trainable=mask)
        print(f"stage K={K}, {nsweeps_phase} sweeps: training {stage_set.nparams} of "
              f"{sum(stage_set.nparams_per_state)} parameters", flush=True)
        train_stage(stage_set, nsweeps_phase, sweep0)
        sweep0 += nsweeps_phase
        record_spectrum(K)

print("NES training completed in: ", datetime.now() - start_time)

with open(f"{path}/meta_{run_id}.txt", "a") as f:
    f.write(f"time: {datetime.now() - start_time}\n")
