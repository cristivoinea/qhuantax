import quantax as qtx
import jax.numpy as jnp
import numpy as np
import scipy as sp
from qhuantax.quantumhall_operators import GetSpinfulDenIntTerms, GetSpinfulPolTerms, GetLpTerms
from qhuantax.quantumhall_samplers import FermionTwoBodyDipoleCons
from qhuantax.quantumhall_symmetries import ParticleHoleQH, FlavourPermQH, IdentityQH
from qhuantax.quantumhall_utils import generate_spin_configs, read_meta_file
from qhuantax.quantumhall_transformer import transformer_backflow_state
from qhuantax.nes.subspace import dense_reduced_matrices
from qhuantax.nes.state_set import NaturalStateSet

from datetime import datetime

S1 = np.array([[1,0],[0,0]])
S2 = np.array([[0,0],[0,1]])
SX = np.array([[0,1],[1,0]])


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-n", action="store", required=True,
                    help="number of particles")
parser.add_argument("-s", action="store", required=True,
                    help="number of orbitals in the system (2s)")
parser.add_argument("--nbr-states", action="store", default=None,
                    help="how many of the run's states to span, defaults to all of them; "
                         "fewer is a smaller trial subspace and so a different spectrum")
parser.add_argument("--lz-sect", action="store", required=True,
                    help="Lz symmetry sector")
parser.add_argument("--z2-sect", action="store", default=0,
                    help="Z2 symmetry sector")
parser.add_argument("--ph-sect", action="store", default=0,
                    help="PH symmetry sector (without spin flip)")

parser.add_argument("--exact-diag", action="store_true", default=False,
                    help="calcualte the reduced matrix exactly using dense states")

parser.add_argument("--nbr-samples", action="store", default=2048,
                    help="number of samples for each measurement")
parser.add_argument("--nbr-meas", action="store", default=5000,
                    help="number of measurements")
parser.add_argument("--sampling-state-index", action="store", default=0,
                    help="state used as the sampling distribution")

parser.add_argument("--meas-op", action="store", default="H",
                    help="measured operator")
parser.add_argument("--pf-backflow", action="store_true", default=False,
                    help="assert the run used PfBackflow; the meta file settles it either way, "
                         "so this only ever agrees or errors")

parser.add_argument("--run-id", action="store", default=1,
                    help="")
parser.add_argument("--path", action="store", required=True,
                    help="path")

args = vars(parser.parse_args())

N = int(args["n"])
L = int(args["s"])+1
nstates = None if args["nbr_states"] is None else int(args["nbr_states"])
lz = int(args["lz_sect"])
z2 = int(args["z2_sect"])
ph = int(args["ph_sect"])
id = int(args["run_id"])
path = str(args["path"])
run_id = f"nes_n_{N}_2s_{L-1}_lz_{lz}_z2_{z2}_ph_{ph}_id0{id}"

meas_op = str(args["meas_op"])
do_ED = bool(args["exact_diag"])
meta_dict = read_meta_file(run_id, path)
pf_backflow = bool(args["pf_backflow"])

if not do_ED:
    nsamples = int(args["nbr_samples"])
    nmeas = int(args["nbr_meas"])
    sampling_state_index = int(args["sampling_state_index"])


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
if meas_op == "H":
    pspot_inter = np.array([4.75, 1])
    transverse_fld = -3.16
    tms_O = GetSpinfulDenIntTerms(nm = L, ps_pot=2*pspot_inter, mat_a = S1, mat_b = S2)
    tms_O += transverse_fld * GetSpinfulPolTerms(nm=L, mat = SX)


# Rebuild the ansatz with exactly the tree it was trained with. Everything that fixes
# that tree comes from the run's own meta file rather than the command line -- network
# geometry, determinants per member, and which backflow -- since a mismatch does not
# raise on load, it quietly fills the model from the wrong slots (see `check_param_file`).
nb = int(meta_dict["nbr. blocks"])
nh = int(meta_dict["nbr. heads"])
d = int(meta_dict["attndim"])

meta_nstates = int(meta_dict["nbr. states"])
if nstates is None:
    nstates = meta_nstates
elif nstates > meta_nstates:
    parser.error(f"{run_id} holds {meta_nstates} states, cannot span {nstates}")

meta_pf = str(meta_dict.get("model", "DetBackflow")) == "PfBackflow"
if pf_backflow and not meta_pf:
    parser.error(f"{run_id} was trained with {meta_dict['model']}; drop --pf-backflow")
pf_backflow = meta_pf

# Runs that predate the entry are single-determinant; `check_param_file` says so if not.
nterms = meta_dict.get("nterms per state", [1] * meta_nstates)
if isinstance(nterms, int):
    nterms = [nterms] * meta_nstates

U = np.zeros((2*L, N))
U[:N,:N] = np.eye(N)
if lz != 0:
    U[0,0] = 0
    U[N,lz] = 1

# `orbital_noise=0`: every leaf is about to be overwritten from the checkpoint, so only
# the shape of `U` matters here, never its value.
states = NaturalStateSet([
    transformer_backflow_state(index, L, N, d, nb, nh, symm, pf_backflow, U,
                               orbital_noise=0, nterms=nterms[index],
                               param_file=f"{path}/state{index}_{run_id}.eqx")
    for index in range(nstates)
])
print(f"{nstates} of {meta_nstates} states, "
      f"{'PfBackflow' if pf_backflow else 'DetBackflow'}, "
      f"{nterms[:nstates]} terms per state")

startTime = datetime.now()
if do_ED:
    print("Converting to dense states and calculating reduced matrices.")
    S, O = dense_reduced_matrices(states, tms_O, lz, z2)
    eigval, eigst = sp.linalg.eigh(O, S)
    print("Measured eigenvalues: ", np.sort(eigval))
else:
    print("Calculating reduced matrices via Monte Carlo.")
    sampling_state = states[sampling_state_index]

    init_configs = generate_spin_configs(L, N, lz, nsamples)
    sampler = FermionTwoBodyDipoleCons(
        sampling_state,
        nsamples,
        n_neighbor=np.arange(1,3),
        initial_spins=init_configs,
    )


    H = np.zeros((nmeas, nstates, nstates))
    H_var = np.zeros((nmeas, nstates, nstates))
    S = np.zeros((nmeas, nstates, nstates))
    S_var = np.zeros((nmeas, nstates, nstates))

    for i in range(nmeas):
        samples = sampler.sweep()
        spins = jnp.asarray(samples.spins)
        sampling_psi = samples.psi
        if sampling_psi is None:
            sampling_psi = sampling_state(spins)
        sampling_psi = jnp.asarray(sampling_psi)

        if samples.reweight_factor is None:
            reweight = jnp.ones(samples.nsamples)
        else:
            reweight = samples.reweight_factor

        denom = jnp.abs(sampling_psi) ** 2
        scale = jnp.where(denom > 0, reweight / denom, 0.0)
        psi = jnp.stack([jnp.asarray(state(spins)) for state in states], axis=-1)
        eloc = jnp.stack(
            [
                tms_O.Oloc(state, spins).astype(psi.dtype)
                for state in states
            ],
            axis=-1,
        )
        S_sample = scale[:,None,None] * psi.conj()[:,:,None] * psi[:,None,:]
        H_sample = scale[:,None,None] * psi.conj()[:,:,None] * psi[:,None,:] * eloc[:,None,:]

        for row in range(nstates):
            S_diag_sample = S_sample[:,row,row]
            H_diag_sample = H_sample[:,row,row]

            S[i,row,row] = np.asarray(jnp.real(jnp.mean(S_diag_sample)))
            S_var[i,row,row] = np.asarray(jnp.mean(jnp.abs(S_diag_sample - S[i,row,row]) ** 2))
            H[i,row,row] = np.asarray(jnp.real(jnp.mean(H_diag_sample)))
            H_var[i,row,row] = np.asarray(jnp.mean(jnp.abs(H_diag_sample - H[i,row,row]) ** 2))

            for col in range(row+1, nstates):
                S_pair_sample = S_sample[:,row,col]
                H_pair_sample = 0.5 * (H_sample[:,row,col] + H_sample[:,col,row])

                S[i,row,col] = np.asarray(jnp.real(jnp.mean(S_pair_sample)))
                S[i,col,row] = S[i,row,col]
                S_var[i,row,col] = np.asarray(jnp.mean(jnp.abs(S_pair_sample - S[i,row,col]) ** 2))
                S_var[i,col,row] = S_var[i,row,col]

                H[i,row,col] = np.asarray(jnp.real(jnp.mean(H_pair_sample)))
                H[i,col,row] = H[i,row,col]
                H_var[i,row,col] = np.asarray(jnp.mean(jnp.abs(H_pair_sample - H[i,row,col]) ** 2))
                H_var[i,col,row] = H_var[i,row,col]

        for row in range(nstates):
            for col in range(row, nstates):
                np.savetxt(
                    f"{path}/meastest_S_{row}_{col}_nsamples_{nsamples}_{run_id}.txt",
                    np.vstack((S[:i+1,row,col], S_var[:i+1,row,col])).T,
                )
                np.savetxt(
                    f"{path}/meastest_H_{row}_{col}_nsamples_{nsamples}_{run_id}.txt",
                    np.vstack((H[:i+1,row,col], H_var[:i+1,row,col])).T,
                )

print("Reduced matrix measurements completed in: ", datetime.now() - startTime)
