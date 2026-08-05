import argparse
from datetime import datetime
from pathlib import Path

import jax.numpy as jnp

import numpy as np
import quantax as qtx
import scipy as sp
from qhuantax.quantumhall_transformer import Transformer

from qhuantax.nes import (
    NaturalExcitedAdamSR,
    NaturalLzDetSampler,
    NaturalStateSet,
    dense_reduced_matrices,
)
from qhuantax.nes.optimizer import _scaled_psi_matrix
from qhuantax.quantumhall_operators import (
    GetLpTerms,
    GetSpinfulDenIntTerms,
    GetSpinfulPolTerms,
)
from qhuantax.quantumhall_symmetries import FlavourPermQH, IdentityQH, ParticleHoleQH
from qhuantax.quantumhall_utils import adaptive_learning_rate, generate_spin_configs


S1 = np.array([[1, 0], [0, 0]])
S2 = np.array([[0, 0], [0, 1]])
SX = np.array([[0, 1], [1, 0]])



def build_state(index, L, N, d, nb, nh, symm, pf_backflow, U, orbital_noise=5e-2, rng=np.random.default_rng(), param_file=None):
    U_state = U.copy()
    if orbital_noise > 0:
        U_state = U_state + orbital_noise * rng.normal(size=U_state.shape)

    net = Transformer(nblocks=nb, d=d, heads=nh, final_sum=False)
    if pf_backflow:
        U_pf = jnp.zeros((2 * L, 2 * L))
        for i in range(N):
            U_pf = U_pf.at[:, 2 * i].add(U_state[:, i])
        model = qtx.model.PfBackflow(net, U0=U_pf, d=d)
    else:
        model = qtx.model.DetBackflow(net, U0=U_state, d=d)

    # `param_file` replaces every leaf of the model, U0 included, so the orbital noise
    # above is irrelevant whenever one is given.
    return qtx.state.Variational(model, param_file=param_file, symm=symm, max_parallel=16384, use_ref=False)



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
parser.add_argument("--diagnostics", action="store_true", default=False,
                    help="write per-sweep step norms and SR conditioning to data_diagnostics_*.txt")
parser.add_argument("--diagnostics-every", action="store", default=5,
                    help="how often to refresh the Jacobian Gram spectrum in the diagnostics file")
parser.add_argument("--mean-field", action="store_true", default=False,
                    help="use mean-field ansatz as initial starting point")
parser.add_argument("--nbr-sweeps-mf", action="store", default=500,
		    help="number of iterations for the MF optimization")
parser.add_argument("--lr-mf", action="store", default=1e-2,
		    help="starting value of the learning rate for the MF optimization")


parser.add_argument("--exact-diag", action="store_true", default=False,
                    help="perform exact diagonalization and track energy, energy variance and overlap with ground state")
parser.add_argument("--lmlp-coeff", action="store", default=0,
                    help="coefficient in front of L^- L^+ term")
parser.add_argument("--lmlp-freq", action="store", default=5,
                    help="measurement frequency of L^- L^+ term")

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
do_MF = bool(args["mean_field"])
nsweeps_MF = int(args["nbr_sweeps_mf"])
lr_MF = int(args["lr_mf"])


do_ED = bool(args["exact_diag"])
LmLp_coeff = float(args["lmlp_coeff"])
LmLp_freq = float(args["lmlp_freq"])

pf_backflow = bool(args["pf_backflow"])
nsweeps = int(args["nbr_sweeps"])
nsamples = int(args["nbr_samples"])
nb = int(args["nbr_blocks"])
nh = int(args["nbr_heads"])
d = int(args["attn_dim"])

lr0 = float(args["lr"])
baseline = lr0/5
delay = nsweeps//5
decay = 2*np.log(2)/delay
rw = float(args["reweight"])
model_type = "DetBackflow"

warmup_sweeps = nsweeps // 2 if args["warmup_sweeps"] is None else int(args["warmup_sweeps"])
diagnostics = bool(args["diagnostics"])
diagnostics_every = int(args["diagnostics_every"])


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


tms_Lp = GetLpTerms(L, 2)
tms_LmLp = tms_Lp.H @ tms_Lp
if LmLp_coeff:
    tms_H = tms_H + LmLp_coeff * tms_LmLp

# MF pre training & load orbitals
if do_MF:
    path_MF = Path(f"{path}/state_MF_{run_id}.txt")
    if path_MF.exists():
        print(f"File already exists: {path}/state_MF_{run_id}.txt")
        U = np.loadtxt(f"{path}/state_MF_{run_id}.txt")[:2*L,:]
    else:
        energy_MF = qtx.utils.DataTracer()

        t = 1.0
        U = np.zeros((2, 2*L, N))
        U[0, :N,:N] = np.eye(N)*np.cos(t/2)
        U[0, N:2*N,:N] = np.eye(N)*np.sin(t/2)
        U[1, :N,:N] = np.eye(N)*np.cos(np.pi/2 - t/2)
        U[1, N:2*N,:N] = np.eye(N)*np.sin(np.pi/2 - t/2)

        model_MF = qtx.model.MultiDet(ndets = 2, U=U, coeffs = jnp.array([1, z2]))
        state_MF = qtx.state.MultiDetState(model_MF)

        for i in range(nsweeps_MF):
            step = state_MF.get_step(tms_H)
            state_MF.update(step * lr_MF)
            energy_MF.append(state_MF.energy)

            np.savetxt(f"{path}/data_MF_{run_id}.txt", np.vstack((energy_MF.time, energy_MF.data)).T)
            np.savetxt(f"{path}/state_MF_{run_id}.txt", np.vstack((state_MF.model.U_full[0,:,:], state_MF.model.U_full[1,:,:])))
        U = U[0, :, :]
else:
    U = np.zeros((2*L, N))
    U[:N,:N] = np.eye(N)
    if lz != 0:
        U[0,0] = 0
        U[N,lz] = 1

start_time = datetime.now()

def init_args(index):
    if init_state_file is not None and index == 0:
        return dict(orbital_noise=0, param_file=init_state_file)
    return dict(orbital_noise=(0 if index == 1 else 1e-1), param_file=None)

member_states = [
    build_state(index, L, N, d, nb, nh, symm, pf_backflow, U, **init_args(index))
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
  f.write(f"optimizer: AdamSR\n")
  f.write(f"nbr. iter.: {nsweeps}\n")
  f.write(f"learning rate: {lr0}\n")
  f.write(f"decay: {decay}\n")
  f.write(f"delay: {delay}\n")
  f.write(f"baseline: {baseline}\n")
  f.write(f"sampler: DipoleCons\n")
  f.write(f"nbr. samples NN: {nsamples}\n")
  f.write(f"reweight: {rw}\n")
  f.write(f"model: {'PfBackflow' if pf_backflow else 'DetBackflow'}\n")
  f.write(f"net: Transformer\n")
  f.write(f"nbr. blocks: {nb}\n")
  f.write(f"nbr. heads: {nh}\n")
  f.write(f"attndim: {d}\n")
  f.write(f"nbr. params per state: {state_set.states[0].nparams}\n")
  f.write(f"nbr. states: {nstates}\n")
  f.write(f"init state file: {init_state_file}\n")
  f.write(f"incremental: {incremental}\n")
  f.write(f"warmup sweeps: {warmup_sweeps}\n")


energy = qtx.utils.DataTracer()
VarE = qtx.utils.DataTracer()
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

    optimizer = NaturalExcitedAdamSR(
        stage_set,
        tms_H,
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

        lr = adaptive_learning_rate(lr0, delay, decay, baseline, sweep0 + i)
        stage_set.update(stage_set.split_step(step * lr))
        energy.append(optimizer.energy)
        VarE.append(optimizer.VarE)

        np.savetxt(
            f"{path}/data_energy_{run_id}.txt",
            np.vstack((energy.time, energy.data, VarE.data)).T,
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
