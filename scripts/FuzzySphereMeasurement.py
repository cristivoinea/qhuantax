import quantax as qtx
import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import matplotlib.pyplot as plt
from qhuantax.quantumhall_operators import GetSpinlessDenIntTerms, GetSpinfulDenIntTerms, GetSpinfulPolTerms
from qhuantax.quantumhall_samplers import FermionTwoBodyDipoleCons, GetLzDenseProjector
from qhuantax.quantumhall_utils import adaptive_learning_rate, generate_spin_configs
from quspin.basis import spinful_fermion_basis_1d

from datetime import datetime
from pathlib import Path

S1 = np.array([[1,0],[0,0]])
S2 = np.array([[0,0],[0,1]])
SX = np.array([[0,1],[1,0]])

N = 24
L = N
lz = 0
z2 = 1
id = 6
cont_run = False
if cont_run:
    run_id = f"n_{N}_2s_{L-1}_lz_{lz}_z2_{z2}_id0{id}-1"
else:
    run_id = f"n_{N}_2s_{L-1}_lz_{lz}_z2_{z2}_id0{id}"
path = f"./data/"

meta_dict = {}
with open ("sample.txt","r",encoding="utf-8") as file:
    for line in file:
        line = line.strip().split(":")
        meta_dict.appent({"A":int(line[0]),
                  "B":int(line[1]),
                  "C":int(line[2])})


nsamples = 2048
nb = 4
nh = 4
d = 96

nsweeps = 3000
lr0 = 1e-3
delay = 500
decay = 2e-3
baseline = 1e-4
rw = 1.0
model_type = "DetBackflow"


lattice = qtx.sites.Chain(L=L, boundary=0, particle_type="spinful_fermion", Nparticles=N)
symm = qtx.symmetry.SpinInverse(eigval=z2)

# V1 quantum Hall Hamiltonian
H = GetSpinfulDenIntTerms(nm = L, ps_pot=2*np.array([4.75,1.]), mat_a = S1, mat_b = S2)
H -= 3.16 * GetSpinfulPolTerms(nm=L, mat = SX)
#print(H)

# Exact diagonalization
if do_ED:
    E, wf = H.diagonalize(k=10, symm=symm)
    print(E)

    proj_mask = GetLzDenseProjector(L, N, lz, 2)
    print(f"(Lz = {lz}) dim = ", proj_mask.size)
    print("GS norm = ",np.dot(wf[proj_mask,0],wf[proj_mask,0]))

startTime = datetime.now()

with open(f"{path}/meta_{run_id}.txt", "w") as f:
  f.write(f"continue run: {cont_run}\n")
  f.write(f"MF ansatz: MF_ansatz\n")

# MF pre training
if MF_ansatz and z2 == 1:
    path_MF = Path(f"{path}/data_MF_{run_id}.txt")
    if path_MF.exists() is False:
        energy_MF = qtx.utils.DataTracer()

        if do_ED:
            overlap_MF = qtx.utils.DataTracer()
            exact_energy_MF = qtx.utils.DataTracer()
            exact_variance_MF = qtx.utils.DataTracer()

        t = 1.0
        U = np.zeros((2, 2*L, N))
        U[0, :N,:N] = np.eye(N)*np.cos(t/2)
        U[0, N:2*N,:N] = np.eye(N)*np.sin(t/2)
        U[1, :N,:N] = np.eye(N)*np.cos(np.pi/2 - t/2)
        U[1, N:2*N,:N] = np.eye(N)*np.sin(np.pi/2 - t/2)

        model_MF = qtx.model.MultiDet(ndets = 2, U=U)
        state_MF = qtx.state.MultiDetState(model_MF)

        for i in range(nsweeps):
            step = state_MF.get_step(H)
            state_MF.update(step * lr_MF)
            energy_MF.append(state_MF.energy)

            if do_ED:
                np.savetxt(f"{path}/data_MF_{run_id}.txt", np.vstack((energy_MF.time, energy_MF.data, exact_energy_MF.data, exact_variance_MF.data, overlap_MF.data)).T)
            else:
                np.savetxt(f"{path}/data_MF_{run_id}.txt", np.vstack((energy_MF.time, energy_MF.data)).T)
            np.savetxt(f"{path}/state_MF_{run_id}.txt", np.vstack((state_MF.model.U_full[0,:,:], state_MF.model.U_full[1,:,:])))


MF_endTime = datetime.now()
print(MF_endTime - startTime)

if MF_ansatz and z2 == 1:
    with open(f"{path}/meta_{run_id}.txt", "a") as f:
      f.write(f"MF nbr. iter.: {nsweeps_MF}\n")
      f.write(f"MF learning rate: {lr_MF}\n")
      f.write(f"MF time: {MF_endTime - startTime}\n")


# start NN training
net = qtx.model.Transformer(nblocks=nb, d=d, heads=nh, final_sum=False)

if MF_ansatz:
    U = np.loadtxt(f"{path}/state_MF_n_{N}_2s_{L-1}_lz_{lz}_z2_1_id0{id}.txt")[:2*L,:]
    #U = state_MF.model.U_full[0,:,:]
else:
    U = np.zeros((2*L, N))
    U[:N,:N] = np.eye(N)
    if lz != 0:
        U[0,0] = 0
        U[N,lz] = 1

if model_type == "DetBackflow":
    model = qtx.model.DetBackflow(net, U0=U, d=d)
elif model_type == "PfBackflow":
    U_pf = jnp.zeros((2*L, 2*L))
    for i in range(N):
        U_pf = U_pf.at[:,2*i].add(U[:,i])
    model = qtx.model.PfBackflow(net, U0=U_pf, d=d)
else:
    print("Model not implemented")


if cont_run:
     state = qtx.state.Variational(model, symm=symm, max_parallel=16384, use_ref=True, param_file=f"{path}/state_{run_id[:-2]}.eqx")
else:
    state = qtx.state.Variational(model, symm=symm, max_parallel=16384, use_ref=True)


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
  f.write(f"exact data: {do_ED}\n")


init_configs = generate_spin_configs(L, N, lz, nsamples)

sampler = FermionTwoBodyDipoleCons(state, nsamples, n_neighbor=np.arange(1,3), initial_spins=init_configs, reweight = rw)
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