import quantax as qtx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from qhuantax.quantumhall_operators import GetSpinfulDenIntTerms, GetSpinfulPolTerms, GetLpTerms
from qhuantax.quantumhall_samplers import FermionTwoBodyDipoleCons, GetLzSymmetryProjector
from qhuantax.quantumhall_utils import adaptive_learning_rate, generate_spin_configs, diagonalize_lz_multiplet, read_meta_file
from qhuantax.quantumhall_symmetries import ParticleHoleQH, FlavourPermQH, IdentityQH

from datetime import datetime
from pathlib import Path

S1 = np.array([[1,0],[0,0]])
S2 = np.array([[0,0],[0,1]])
SX = np.array([[0,1],[1,0]])


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-n", action="store", required=True,
                    help="number of particles")
parser.add_argument("-s", action="store", required=True,
                    help="number of orbitals in the system (2s)")
parser.add_argument("--lz-sect", action="store", required=True,
                    help="Lz symmetry sector")
parser.add_argument("--z2-sect", action="store", default=0,
                    help="Z2 symmetry sector")
parser.add_argument("--ph-sect", action="store", default=0,
                    help="PH symmetry sector (without spin flip)")

parser.add_argument("--exact-diag", action="store_true", default=False,
                    help="perform exact diagonalization and track energy, energy variance and overlap with ground state")
parser.add_argument("--lmlp-coeff", action="store", default=0,
                    help="coefficient in front of L^- L^+ term")
parser.add_argument("--lmlp-freq", action="store", default=5,
                    help="measurement frequency of L^- L^+ term")

parser.add_argument("--initial-lz", action="store", default=0,
                    help="Lz symmetry sector of the initial state guess")
parser.add_argument("--initial-z2", action="store", default=0,
                    help="Z2 symmetry sector of the initial state guess")
parser.add_argument("--initial-ph", action="store", default=0,
                    help="PH symmetry sector of the initial state guess")
parser.add_argument("--initial-run-id", action="store", default=1,
                    help="run ID of the initial state guess")


parser.add_argument("--use-meta-init", action="store_true", default=False,
                    help="whether to use the metadata of the initial state NN")
parser.add_argument("--nbr-heads", action="store", default=4,
                    help="number of attention heads in the transformer")
parser.add_argument("--attn-dim", action="store", default=16,
                    help="attention dimension of the transformer")
parser.add_argument("--nbr-blocks", action="store", default=1,
		    help="number of layers in the transformer")
parser.add_argument("--nbr-sweeps", action="store", default=1000,
		    help="number of training iterations")
parser.add_argument("--nbr-samples", action="store", default=1024,
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
lz = int(args["lz_sect"])
z2 = int(args["z2_sect"])
ph = int(args["ph_sect"])
id = str(args["run_id"])
path = str(args["path"])
run_id = f"n_{N}_2s_{L-1}_lz_{lz}_z2_{z2}_ph_{ph}_id0{id}"

initial_lz = int(args["initial_lz"])
initial_z2 = int(args["initial_z2"])
initial_ph = int(args["initial_ph"])
initial_id = str(args["initial_run_id"])
initial_run_id = f"n_{N}_2s_{L-1}_lz_{initial_lz}_z2_{initial_z2}_ph_{initial_ph}_id0{initial_id}"

do_ED = bool(args["exact_diag"])
LmLp_coeff = float(args["lmlp_coeff"])
LmLp_freq = float(args["lmlp_freq"])


use_meta = bool(args["use_meta_init"])
if use_meta:
    meta_dict = read_meta_file(initial_run_id, path)

    nsweeps = meta_dict["nbr. iter."]
    nsamples = meta_dict["nbr. samples NN"]
    nb = meta_dict["nbr. blocks"]
    nh = meta_dict["nbr. heads"]
    d = meta_dict["attndim"]

    lr0 = meta_dict["learning rate"]
    baseline = meta_dict["baseline"]
    delay = meta_dict["delay"]
    decay = meta_dict["decay"]
    rw = meta_dict["reweight"]
else:
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


lattice = qtx.sites.Chain(L=L, boundary=0, particle_type="spinful_fermion", Nparticles=N)
if z2 != 0:
    symm = FlavourPermQH(eigval=z2)
    if ph != 0:
        symm = ParticleHoleQH(eigval=ph) @ symm
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


# Exact diagonalization
if do_ED:
    E, wf = tms_H.diagonalize(k=15, symm=symm) # need to fix the number of states being requested
    print("Exact spectrum: ", E)

    proj_mask = GetLzSymmetryProjector(L, N, lz, symm=symm, nflav=2)
    
    wf_array = np.zeros((np.size(wf[:,0]), 3))
    wf_array[:,0] = wf[:,0]

    inds = np.array([0,2,5])
    lz_target = np.array([1,2])
    for l in lz_target:
        mask = np.isclose(E, E[inds[l]])
        lz_vals, wf_lz = diagonalize_lz_multiplet(
                    wf[:, mask],
                    L=L,
                    nflav=2,
                    symm=symm,)
        wf_exact = wf_lz[:,2*l]
        proj_mask = GetLzSymmetryProjector(L, N, l, symm=symm, nflav=2)
        print("Target state norm = ",np.dot(wf_exact[proj_mask], wf_exact[proj_mask]))
        wf_array[:,l] = wf_exact


startTime = datetime.now()

tms_Lp = GetLpTerms(L, 2)
tms_Lm = tms_Lp.H
tms_LmLp = tms_Lm @ tms_Lp
if LmLp_coeff:
    tms_H = tms_H + LmLp_coeff * tms_LmLp


# start NN training
net = qtx.model.Transformer(nblocks=nb, d=d, heads=nh, final_sum=False)

U = np.zeros((2*L, N))
if model_type == "DetBackflow":
    model = qtx.model.DetBackflow(net, U0=U, d=d)
elif model_type == "PfBackflow":
    U_pf = jnp.zeros((2*L, 2*L))
    for i in range(N):
        U_pf = U_pf.at[:,2*i].add(U[:,i])
    model = qtx.model.PfBackflow(net, U0=U_pf, d=d)
else:
    print("Model not implemented")


state = qtx.state.Variational(model, symm=symm, max_parallel=16384, param_file=f"{path}/state_{initial_run_id}.eqx")


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
  f.write(f"model: {model_type}\n")
  f.write(f"net: Transformer\n")
  f.write(f"nbr. blocks: {nb}\n")
  f.write(f"nbr. heads: {nh}\n")
  f.write(f"attndim: {d}\n")
  f.write(f"nbr. params: {state.nparams}\n")


init_configs = generate_spin_configs(L, N, lz, nsamples)
sampler = FermionTwoBodyDipoleCons(state, nsamples, n_neighbor=np.arange(1,3), initial_spins=init_configs, reweight = rw)
optimizer = qtx.optimizer.AdamSR(state, tms_H)

energy = qtx.utils.DataTracer()
VarE = qtx.utils.DataTracer()
LmLp_tracer = qtx.utils.DataTracer()
LmLp_var_tracer = qtx.utils.DataTracer()
if do_ED:
    overlap = [qtx.utils.DataTracer() for l in range(3)]
    exact_energy = qtx.utils.DataTracer()
    exact_variance = qtx.utils.DataTracer()


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
    
    if do_ED:
        for l in range(3):
            proj_mask_lz  = GetLzSymmetryProjector(L, N, l, symm=symm, nflav=2)
            
            dense = state.todense(symm).normalize()
            new = dense.psi.value().at[:].set(0)
            norm = np.sqrt(dense.psi.value()[proj_mask_lz] @ dense.psi.value()[proj_mask_lz])
            new = qtx.state.DenseState(new.at[proj_mask_lz].set(dense.psi.value()[proj_mask_lz] / norm), symm)
            
            overlap[l].append(abs( (new @ wf_array[:,l])**2 ))

            if l == 0:
                exact_energy.append((new @ tms_H @ new))
                exact_variance.append(((new @ tms_H @ tms_H @ new) - (new @ tms_H @ new)**2))
                #exact_LmLp.append(new @ dense_LmLp @ new)
                #new_1 = new
                #for i in range(1,26):
                #    new_1 = OneMinusProjector(0, i) @ new_1
                #exact_L0proj.append(new_1 @ new_1)
                np.savetxt(f"{path}/data_energy_exact_{run_id}.txt", np.vstack((exact_energy.time, exact_energy.data, exact_variance.data )).T)

        np.savetxt(f"{path}/data_ovl_exact_{run_id}.txt", np.vstack((overlap[0].time, overlap[0].data, overlap[1].data, overlap[2].data)).T)

    np.savetxt(f"{path}/data_energy_{run_id}.txt", np.vstack((energy.time, energy.data, VarE.data)).T)
    state.save(f"{path}/state_{run_id}.eqx")


print("Training completed in: ",datetime.now() - startTime)

with open(f"{path}/meta_{run_id}.txt", "a") as f:
  f.write(f"time: {datetime.now() - startTime}\n")

