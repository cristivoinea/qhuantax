import argparse
from datetime import datetime
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import quantax as qtx

from qhuantax.nes import NaturalExcitedAdamSR, NaturalLzDetSampler, NaturalStateSet
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



def build_state(index, L, N, d, nb, nh, symm, pf_backflow, U, orbital_noise=5e-2, rng=np.random.default_rng()):
    U_state = U.copy()
    if orbital_noise > 0:
        U_state = U_state + orbital_noise * rng.normal(size=U_state.shape)

    net = qtx.model.Transformer(nblocks=nb, d=d, heads=nh, final_sum=False)
    if pf_backflow:
        U_pf = jnp.zeros((2 * L, 2 * L))
        for i in range(N):
            U_pf = U_pf.at[:, 2 * i].add(U_state[:, i])
        model = qtx.model.PfBackflow(net, U0=U_pf, d=d)
    else:
        model = qtx.model.DetBackflow(net, U0=U_state, d=d)

    return qtx.state.Variational(model, symm=symm, max_parallel=16384)



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
else:
    U = np.zeros((2*L, N))
    U[:N,:N] = np.eye(N)
    if lz != 0:
        U[0,0] = 0
        U[N,lz] = 1

start_time = datetime.now()

member_states = tuple(
    build_state(index, L, N, d, nb, nh, symm, pf_backflow, U)
    for index in range(nstates)
)
state_set = NaturalStateSet(member_states)


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
  f.write(f"model: {"PfBackflow" if pf_backflow else "DetBackflow"}\n")
  f.write(f"net: Transformer\n")
  f.write(f"nbr. blocks: {nb}\n")
  f.write(f"nbr. heads: {nh}\n")
  f.write(f"attndim: {d}\n")
  f.write(f"nbr. params per state: {state_set.states[0].nparams}\n")


init_configs = generate_spin_configs(L, N, lz, nsamples * nstates)
init_configs = init_configs.reshape(nsamples, nstates, 2 * L)

sampler = NaturalLzDetSampler(
    state_set,
    nsamples,
    n_neighbor=np.arange(1, 3),
    initial_spins=init_configs,
    reweight=rw)

optimizer = NaturalExcitedAdamSR(state_set, tms_H)

energy = qtx.utils.DataTracer()
VarE = qtx.utils.DataTracer()
LmLp_tracer = qtx.utils.DataTracer()
LmLp_var_tracer = qtx.utils.DataTracer()

for i in range(nsweeps):
    samples = sampler.sweep()
    step = optimizer.get_step(samples)
    lr = adaptive_learning_rate(lr0, delay, decay, baseline, i)
    state_set.update(state_set.split_step(step * lr))

    energy.append(optimizer.energy)
    VarE.append(optimizer.VarE)

    np.savetxt(
        f"{path}/data_energy_{run_id}.txt",
        np.vstack((energy.time, energy.data, VarE.data)).T,
    )
    
    for index, state in enumerate(state_set.states):
        state.save(f"{path}/state{index}_{run_id}.eqx")

print("NES training completed in: ", datetime.now() - start_time)

with open(f"{path}/meta_{run_id}.txt", "a") as f:
    f.write(f"time: {datetime.now() - start_time}\n")
