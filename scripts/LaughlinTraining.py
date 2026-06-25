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


N = 6
L = 3*N-2
lz = 0
id = 1
run_id = f"n_{N}_2s_{L-1}_lz_{lz}_id0{id}"
path = f"./data/laughlin"

do_ED = True

MF_ansatz = True
nsweeps_MF = 2000
lr_MF = 1e-2

nsamples = 2048
nb = 4
nh = 4
d = 64

nsweeps = 5000
lr0 = 1e-3
delay = 500
decay = 2e-3
baseline = 1e-4
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

with open(f"{path}/meta_{run_id}.txt", "w") as f:
  f.write(f"MF ansatz: MF_ansatz\n")

# MF pre training
if MF_ansatz:
    path_MF = Path(f"{path}/data_MF_{run_id}.txt")
    if path_MF.exists() is False:
        energy_MF = qtx.utils.DataTracer()

        if do_ED:
            overlap_MF = qtx.utils.DataTracer()
            exact_energy_MF = qtx.utils.DataTracer()
            exact_variance_MF = qtx.utils.DataTracer()

        U = _init_spinless_orbs(jnp.float64)[:,:lattice.Ntotal]

        model_MF = qtx.model.GeneralDet(U=U)
        state_MF = qtx.state.GeneralDetState(model_MF)

        for i in range(nsweeps_MF):
            step = state_MF.get_step(H)
            state_MF.update(step * lr_MF)
            energy_MF.append(state_MF.energy)

            if do_ED:
                dense = state_MF.todense().normalize()

                overlap_MF.append(abs( (dense@ wf[:,0])**2 ))
                exact_energy_MF.append((dense @ H @ dense))
                exact_variance_MF.append(((dense @ H @ H @ dense) - (dense @ H @ dense)**2))

            if do_ED:
                np.savetxt(f"{path}/data_MF_{run_id}.txt", np.vstack((energy_MF.time, energy_MF.data, exact_energy_MF.data, exact_variance_MF.data, overlap_MF.data)).T)
            else:
                np.savetxt(f"{path}/data_MF_{run_id}.txt", np.vstack((energy_MF.time, energy_MF.data)).T)
            np.savetxt(f"{path}/state_MF_{run_id}.txt", state_MF.model.U_full)


MF_endTime = datetime.now()
print(MF_endTime - startTime)

if MF_ansatz:
    with open(f"{path}/meta_{run_id}.txt", "a") as f:
      f.write(f"MF nbr. iter.: {nsweeps_MF}\n")
      f.write(f"MF learning rate: {lr_MF}\n")
      f.write(f"MF time: {MF_endTime - startTime}\n")


# start NN training
net = qtx.model.Transformer(nblocks=nb, d=d, heads=nh, final_sum=False)

if MF_ansatz:
    U = np.loadtxt(f"{path}/state_MF_n_{N}_2s_{L-1}_lz_{lz}_id0{id}.txt")
    #U = state_MF.model.U_full[0,:,:]
else:
    U = _init_spinless_orbs(jnp.float64)[:,:lattice.Ntotal]


model = qtx.model.DetBackflow(net, U0=U, d=d)
state = qtx.state.Variational(model, max_parallel=16384, use_ref=False)#, param_file="./data/3d_ising/iqh/state_DetBackflow_DipoleCons_n_12_2s_11_nsamples_1024_rw_1.0_lr_0.001_nb_2_nh_2_d_16.eqx")


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
  f.write(f"model: DetBackflow\n")
  f.write(f"net: Transformer\n")
  f.write(f"nbr. blocks: {nb}\n")
  f.write(f"nbr. heads: {nh}\n")
  f.write(f"attndim: {d}\n")
  f.write(f"nbr. params: {state.nparams}\n")
  f.write(f"exact data: {do_ED}\n")


init_configs = np.array([1 if i%3 == 0 else -1 for i in range(L)])

sampler = FermionTwoBodyDipoleCons(state, nsamples, n_neighbor=np.arange(1,4), initial_spins=init_configs, reweight = rw)
optimizer = qtx.optimizer.AdamSR(state, H)

energy = qtx.utils.DataTracer()
VarE = qtx.utils.DataTracer()
if do_ED:
    overlap = qtx.utils.DataTracer()
    exact_energy = qtx.utils.DataTracer()
    exact_variance = qtx.utils.DataTracer()


for i in range(nsweeps):
    samples = sampler.sweep()
    step = optimizer.get_step(samples)
    state.update(step * adaptive_learning_rate(lr0, delay, decay, baseline, i))

    energy.append(optimizer.energy)
    VarE.append(optimizer.VarE)


    if do_ED:
        dense = state.todense().normalize()
        new = dense.psi.value().at[:].set(0)
        norm = np.sqrt(dense.psi.value()[proj_mask] @ dense.psi.value()[proj_mask])
        new = qtx.state.DenseState(new.at[proj_mask].set(dense.psi.value()[proj_mask] / norm))

        overlap.append(abs( (new @ wf[:,0])**2 ))
        exact_energy.append((new @ H @ new))
        exact_variance.append(((new @ H @ H @ new) - (new @ H @ new)**2))

    if do_ED:
        np.savetxt(f"{path}/data_{run_id}.txt", np.vstack((energy.time, energy.data, VarE.data, exact_energy.data, exact_variance.data, overlap.data)).T)
        state.save(f"{path}/state_{run_id}.eqx")
    else:
        np.savetxt(f"{path}/data_{run_id}.txt", np.vstack((energy.time, energy.data, VarE.data)).T)
        state.save(f"{path}/state_{run_id}.eqx")


print(datetime.now() - MF_endTime)
with open(f"{path}/meta_{run_id}.txt", "a") as f:
  f.write(f"time: {datetime.now() - MF_endTime}\n")
