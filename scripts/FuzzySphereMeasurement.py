import quantax as qtx
from qhuantax.quantumhall_transformer import transformer_backflow_state
import equinox as eqx
import jax.scipy as jsp
import numpy as np
import matplotlib.pyplot as plt
from qhuantax.quantumhall_operators import GetSpinlessDenIntTerms, GetSpinfulDenIntTerms, GetSpinfulPolTerms
from qhuantax.quantumhall_samplers import FermionTwoBodyDipoleCons, GetLzDenseProjector
from qhuantax.quantumhall_symmetries import ParticleHoleQH, FlavourPermQH, IdentityQH
from qhuantax.quantumhall_utils import adaptive_learning_rate_exp, generate_spin_configs, read_meta_file
from quspin.basis import spinful_fermion_basis_1d

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

parser.add_argument("--run-id", action="store", default=1,
                    help="")
parser.add_argument("--path", action="store", required=True,
                    help="path")
parser.add_argument("--meas-op", action="store", default="H",
                    help="measured operator")
parser.add_argument("--pf-backflow", action="store_true", default=False,
                    help="change the ansatz structure from PfBackflow to DetBackflow")

args = vars(parser.parse_args())

N = int(args["n"])
L = int(args["s"])+1
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


# Rebuild the ansatz with the tree the checkpoint was written from. What fixes that tree
# comes from the run's own meta file rather than the command line, because a mismatch does
# not raise on load: Equinox reads leaves positionally, so a wrong tree quietly fills the
# model from the wrong slots. `transformer_backflow_state` checks it instead.
nterms = int(meta_dict["nterms"])
nb = int(meta_dict["nbr. blocks"])
nh = int(meta_dict["nbr. heads"])
d = int(meta_dict["attndim"])

meta_pf = str(meta_dict.get("model", "DetBackflow")) == "PfBackflow"
if pf_backflow and not meta_pf:
    raise ValueError(f"{run_id} was trained with {meta_dict['model']}; drop --pf-backflow")
pf_backflow = meta_pf

# Only the shape of `U` matters here: every leaf is about to come from the checkpoint.
U = np.zeros((2*L, N))
state = transformer_backflow_state(0, L, N, d, nb, nh, symm, pf_backflow, U,
                                   orbital_noise=0, nterms=nterms,
                                   param_file=f"{path}/state_{run_id}.eqx")


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
    np.savetxt(f"{path}/meas_{meas_op}_nsamples_{nsamples}_{run_id}.txt", np.vstack((expval, var)).T)
