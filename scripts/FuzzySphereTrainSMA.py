import quantax as qtx
import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import matplotlib.pyplot as plt
from qhuantax.quantumhall_operators import GetSpinlessDenIntTerms, GetSpinfulDenIntTerms, GetSpinfulPolTerms, GetLpTerms, GetIdTerms
from qhuantax.quantumhall_samplers import FermionTwoBodyDipoleCons, GetLzDenseProjector
from qhuantax.quantumhall_utils import adaptive_learning_rate, generate_spin_configs, read_meta_file
from qhuantax.quantumhall_symmetries import ParticleHoleQH, FlavourPermQH, IdentityQH
from quspin.basis import spinful_fermion_basis_1d

from qhuantax.quantumhall_operators import GetSpinfulMultipoleTerms
from qhuantax.quantumhall_states import OperatedState, OffDiagonalOperatedState

from datetime import datetime
from pathlib import Path

S1 = np.array([[1,0],[0,0]])
S2 = np.array([[0,0],[0,1]])
SX = np.array([[0,1],[1,0]])
SZ = np.array([[1,0],[0,-1]])


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-N", action="store", required=True,
                    help="number of particles")
parser.add_argument("-L", action="store", required=True,
                    help="number of flux quanta")
parser.add_argument("--lz-sect", action="store", required=True,
                    help="Lz symmetry sector")
parser.add_argument("--z2-sect", action="store", default=0,
                    help="Z2 symmetry sector")
parser.add_argument("--ph-sect", action="store", default=0,
                    help="PH symmetry sector (without spin flip)")
parser.add_argument("--lmlp-coeff", action="store", default=0,
                    help="coefficient in front of L^- L^+ term")
parser.add_argument("--lmlp-freq", action="store", default=5,
                    help="measurement frequency of L^- L^+ term")
parser.add_argument("--run-id", action="store", default=1,
                    help="ID of current run")
parser.add_argument("--ground-run-id", action="store", default=1,
                    help="ID of ground state run")
parser.add_argument("--ground-path", action="store", required=True,
                    help="path of ground state file")

args = vars(parser.parse_args())

N = int(args["N"])
L = int(args["L"])
lz = int(args["lz_sect"])
z2 = int(args["z2_sect"])
ph = int(args["ph_sect"])
id = int(args["run_id"])
gs_id = int(args["ground_run_id"])
gs_path = str(args["ground_path"])
path = gs_path + "/from_gs"

run_id = f"n_{N}_2s_{L-1}_lz_{lz}_z2_{z2}_ph_{ph}_id0{id}"
gs_run_id = f"n_{N}_2s_{L-1}_lz_0_z2_1_ph_0_id0{gs_id}"

LmLp_coeff = float(args["lmlp_coeff"])
LmLp_freq = float(args["lmlp_freq"])


meta_dict = read_meta_file(gs_run_id, gs_path)


nsamples = 4096 #meta_dict["nbr. samples NN"]
nb = meta_dict["nbr. blocks"]
nh = meta_dict["nbr. heads"]
d = meta_dict["attndim"]

nsweeps = 3000
lr0 = 5e-4
baseline = 1e-4
delay = nsweeps//5
decay = 2*np.log(2)/delay

model_type = meta_dict["model"]
rw = meta_dict["reweight"]

lattice = qtx.sites.Chain(L=L, boundary=0, particle_type="spinful_fermion", Nparticles=N)
gs_symm = FlavourPermQH(eigval=1)


# V1 quantum Hall Hamiltonian
tms_H = GetSpinfulDenIntTerms(nm = L, ps_pot=2*np.array([4.75,1.]), mat_a = S1, mat_b = S2)
tms_H -= 3.16 * GetSpinfulPolTerms(nm=L, mat = SX)
#print(H)


startTime = datetime.now()


tms_Lp = GetLpTerms(L, 2)
tms_Lm = tms_Lp.H
tms_LmLp = tms_Lm @ tms_Lp
if LmLp_coeff:
    tms_H = tms_H + LmLp_coeff * tms_LmLp

with open(f"{path}/meta_{run_id}.txt", "w") as f:
  f.write(f"GS id: {gs_id}\n")
  f.write(f"L^- L^+ coeff: {LmLp_coeff}\n")
  f.write(f"L^- L^+ meas. frequency: {LmLp_freq}\n")


# start NN training
net = qtx.model.Transformer(nblocks=nb, d=d, heads=nh, final_sum=False)
model = qtx.model.DetBackflow(net, d=d)

nz_lm =GetSpinfulMultipoleTerms(L, lz, lz, (SZ if z2 == -1 else SX))
if lz == 0 and z2 == -1:
    state = OperatedState(model, param_file=f"{gs_path}/state_{gs_run_id}.eqx", operator=nz_lm, base_symm=gs_symm, max_parallel=2048)
else:
    state = OffDiagonalOperatedState(model, param_file=f"{gs_path}/state_{gs_run_id}.eqx", operator=nz_lm, base_symm=gs_symm, max_parallel=1024)


with open(f"{path}/meta_{run_id}.txt", "a") as f:
  f.write(f"optimizer: AdamSR\n")
  f.write(f"nbr. iter.: {nsweeps}\n")
  f.write(f"learning rate: {lr0}\n")
  f.write(f"decay: {decay}\n")
  f.write(f"delay: {delay}\n")
  f.write(f"baseline: {baseline}\n")
  f.write(f"sampler: DipoleCons\n")
  f.write(f"nbr. samples NN: {nsamples}\n")
  f.write(f"reweight: {rw}\n")
  f.write(f"model: {model_type}\n")
  f.write(f"net: Transformer\n")
  f.write(f"nbr. blocks: {nb}\n")
  f.write(f"nbr. heads: {nh}\n")
  f.write(f"attndim: {d}\n")
  f.write(f"nbr. params: {state.nparams}\n")


init_configs = generate_spin_configs(L, N, lz, nsamples)

sampler = FermionTwoBodyDipoleCons(state, nsamples, n_neighbor=np.arange(1,3), initial_spins=init_configs, reweight = rw, thermal_steps = 200*L)
optimizer = qtx.optimizer.AdamSR(state, tms_H)

energy = qtx.utils.DataTracer()
VarE = qtx.utils.DataTracer()
LmLp_tracer = qtx.utils.DataTracer()
LmLp_var_tracer = qtx.utils.DataTracer()

for i in range(nsweeps):
    samples = sampler.sweep()
    step = optimizer.get_step(samples)
    state.update(step * adaptive_learning_rate(lr0, delay, decay, baseline, i))

    energy.append(optimizer.energy)
    VarE.append(optimizer.VarE)

    if i % LmLp_freq == 0:
        expval, var = tms_LmLp.expectation(state, samples, return_var=True)
        LmLp_tracer.append(expval)
        LmLp_var_tracer.append(var)
        np.savetxt(f"{path}/data_LmLp_{run_id}.txt", np.vstack((LmLp_tracer.time, LmLp_tracer.data, LmLp_var_tracer.data)).T)
        
    np.savetxt(f"{path}/data_energy_{run_id}.txt", np.vstack((energy.time, energy.data, VarE.data)).T)
    state.save(f"{path}/state_{run_id}.eqx")


print(datetime.now() - startTime)
with open(f"{path}/meta_{run_id}.txt", "a") as f:
  f.write(f"time: {datetime.now() - startTime}\n")

