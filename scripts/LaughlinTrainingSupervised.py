import quantax as qtx
import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import matplotlib.pyplot as plt
from qhuantax.quantumhall_operators import GetSpinlessDenIntTerms
from qhuantax.quantumhall_samplers import FermionTwoBodyDipoleCons, GetLzDenseProjector
from qhuantax.quantumhall_utils import adaptive_learning_rate, generate_spin_configs
from quspin.basis import spinful_fermion_basis_1d
from quantax.model.fermion_mf import _init_spinless_orbs

from datetime import datetime
from pathlib import Path


N = 5
L = 3*N-2
lz = 0
id = 1
run_id = f"n_{N}_2s_{L-1}_lz_{lz}_id0{id}"
path = f"./data/laughlin/supervised"

do_ED = True

nsamples = 256
nb = 4
nh = 4
d = 32

nsweeps = 1024
lr0 = 2e-2
delay = 200
decay = 1e-2
baseline = 5e-3
rw = 1.0


lattice = qtx.sites.Chain(L=L, boundary=0, particle_type="spinless_fermion", Nparticles=N)

# V1 quantum Hall Hamiltonian
H = GetSpinlessDenIntTerms(nm = L, ps_pot=np.array([0,1.]))
#print(H)

# Exact diagonalization
if do_ED:
    E, wf = H.diagonalize(k=10)
    print(E)

    proj_mask = GetLzDenseProjector(L, N, lz, 1)
    print(f"(Lz = {lz}) dim = ", proj_mask.size)
    print("GS norm = ",np.dot(wf[proj_mask,0],wf[proj_mask,0]))

startTime = datetime.now()


laughlin_exact = qtx.state.DenseState(wf[:,0])

# start NN training
net = qtx.model.Transformer(nblocks=nb, d=d, heads=nh, final_sum=False)
U = _init_spinless_orbs(jnp.float64)[:,:lattice.Ntotal]
model = qtx.model.DetBackflow(net, U0=U, d=d)
state = qtx.state.Variational(model, max_parallel=16384, use_ref=False)#, param_file="./data/3d_ising/iqh/state_DetBackflow_DipoleCons_n_12_2s_11_nsamples_1024_rw_1.0_lr_0.001_nb_2_nh_2_d_16.eqx")

with open(f"{path}/meta_supervised_{run_id}.txt", "a") as f:
  f.write(f"optimizer: Supervised\n")
  f.write(f"nbr. iter.: {nsweeps}\n")
  f.write(f"learning rate: {lr0}\n")
  f.write(f"decay: {decay}\n")
  f.write(f"delay: {delay}\n")
  f.write(f"baseline: {baseline}\n")
  f.write(f"sampler: DipoleCons\n")
  f.write(f"nbr. samples NN: {nsamples}\n")
  f.write(f"reweight: {rw}\n")
  f.write(f"model: DetBackflow\n")
  f.write(f"net: Transformer\n")
  f.write(f"nbr. blocks: {nb}\n")
  f.write(f"nbr. heads: {nh}\n")
  f.write(f"attndim: {d}\n")
  f.write(f"nbr. params: {state.nparams}\n")
  f.write(f"exact data: {do_ED}\n")


init_configs = np.array([1 if i%3 == 0 else -1 for i in range(L)])
sampler = FermionTwoBodyDipoleCons(state, nsamples, n_neighbor=np.arange(1,4), initial_spins=init_configs, reweight = rw)
optimizer = qtx.optimizer.Supervised(state, laughlin_exact)

energy = qtx.utils.DataTracer()
VarE = qtx.utils.DataTracer()
if do_ED:
    overlap = qtx.utils.DataTracer()
    exact_energy = qtx.utils.DataTracer()
    exact_variance = qtx.utils.DataTracer()


for i in range(nsweeps):
    samples0 = sampler.sweep()
    step = optimizer.get_step(samples0)
    state.update(step * adaptive_learning_rate(lr0, delay, decay, baseline, i))

    energy_samples, varE_samples = H.expectation(state, samples0, return_var=True)
    energy.append(energy_samples)
    VarE.append(varE_samples)


    if do_ED:
        dense = state.todense().normalize()
        new = dense.psi.value().at[:].set(0)
        norm = np.sqrt(dense.psi.value()[proj_mask] @ dense.psi.value()[proj_mask])
        new = qtx.state.DenseState(new.at[proj_mask].set(dense.psi.value()[proj_mask] / norm))

        overlap.append(abs( (new @ wf[:,0])**2 ))
        exact_energy.append((new @ H @ new))
        exact_variance.append(((new @ H @ H @ new) - (new @ H @ new)**2))

    if do_ED:
        np.savetxt(f"{path}/data_supervised_{run_id}.txt", np.vstack((energy.time, energy.data, VarE.data, exact_energy.data, exact_variance.data, overlap.data)).T)
        state.save(f"{path}/state_supervised_{run_id}.eqx")
    else:
        np.savetxt(f"{path}/data_supervised_{run_id}.txt", np.vstack((energy.time, energy.data, VarE.data)).T)
        state.save(f"{path}/state_supervised_{run_id}.eqx")


print(datetime.now() - startTime)
with open(f"{path}/meta_supervised_{run_id}.txt", "a") as f:
  f.write(f"time: {datetime.now() - startTime}\n")
