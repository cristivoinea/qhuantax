import quantax as qtx
import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import matplotlib.pyplot as plt
from qhuantax.quantumhall_operators import GetSpinlessDenIntTerms, GetSpinfulDenIntTerms, GetSpinfulPolTerms
from qhuantax.quantumhall_samplers import FermionTwoBodyDipoleCons, GetLzDenseProjector
from qhuantax.quantumhall_symmetries import ParticleHoleQH, FlavourPermQH, IdentityQH
from qhuantax.quantumhall_utils import adaptive_learning_rate, generate_spin_configs, read_meta_file
from quspin.basis import spinful_fermion_basis_1d

from datetime import datetime
from pathlib import Path

S1 = np.array([[1,0],[0,0]])
S2 = np.array([[0,0],[0,1]])
SX = np.array([[0,1],[1,0]])

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
parser.add_argument("--run-id", action="store", default=1,
                    help="")
parser.add_argument("--path", action="store", required=True,
                    help="path")
parser.add_argument("--meas-op", action="store", default="H",
                    help="measured operator")
parser.add_argument("--pf-backflow", action="store_true", default=False,
                    help="change the ansatz structure from PfBackflow to DetBackflow")

args = vars(parser.parse_args())

N = int(args["N"])
L = int(args["L"])
lz = int(args["lz_sect"])
z2 = int(args["z2_sect"])
ph = int(args["ph_sect"])
id = int(args["run_id"])
path = str(args["path"])
meas_op = str(args["meas_op"])
pf_backflow = bool(args["pf_backflow"])


run_id = f"n_{N}_2s_{L-1}_lz_{lz}_z2_{z2}_ph_{ph}_id0{id}"


nsamples = 2048
nmeas = 5000

meta_dict = read_meta_file(run_id, path)


lattice = qtx.sites.Chain(L=L, boundary=0, particle_type="spinful_fermion", Nparticles=N)
if z2 != 0:
    symm = FlavourPermQH(eigval=z2)
    if ph != 0:
        symm = ParticleHoleQH(eigval=ph) @ symm
else:
    symm = IdentityQH()
    if ph != 0:
        symm = ParticleHoleQH(eigval=ph) @ symm


# initialize NQS
nb = int(meta_dict["nbr. blocks"])
nh = int(meta_dict["nbr. heads"])
d = int(meta_dict["attndim"])
net = qtx.model.Transformer(nblocks=nb, d=d, heads=nh, final_sum=False)

U = np.zeros((2*L, N))

if pf_backflow:
    U_pf = jnp.zeros((2*L, 2*L))
    for i in range(N):
        U_pf = U_pf.at[:,2*i].add(U[:,i])
    model = qtx.model.PfBackflow(net, U0=U_pf, d=d)
else:
    model = qtx.model.DetBackflow(net, U0=U, d=d)


state = qtx.state.Variational(model, symm=symm, max_parallel=16384, param_file=f"{path}/state_{run_id}.eqx")


# get operator
if meas_op == "H":
    O = GetSpinfulDenIntTerms(nm = L, ps_pot=2*np.array([4.75,1.]), mat_a = S1, mat_b = S2)
    O -= 3.16 * GetSpinfulPolTerms(nm=L, mat = SX)



init_configs = generate_spin_configs(L, N, lz, nsamples)
sampler = FermionTwoBodyDipoleCons(state, nsamples, n_neighbor=np.arange(1,4), initial_spins=init_configs)

expval = np.zeros(nmeas)
var = np.zeros(nmeas)
for i in range(nmeas):
    samples = sampler.sweep()
    expval[i], var[i] = O.expectation(state, samples, return_var=True)
    np.savetxt(f"{path}/meastest_{meas_op}_nsamples_{nsamples}_{run_id}.txt", np.vstack((expval, var)).T)
