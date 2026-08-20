import quantax as qtx
from qhuantax.quantumhall_transformer import transformer_backflow_state
import equinox as eqx
import numpy as np
from qhuantax.quantumhall_operators import GetSpinfulDenIntTerms, GetSpinfulPolTerms, GetLpTerms
from qhuantax.quantumhall_samplers import FermionTwoBodyDipoleCons, GetLzSymmetryProjector
from qhuantax.quantumhall_optimizers import PenalizedAdamSR
from qhuantax.quantumhall_utils import (
    MF_BACKFLOW_SCALE,
    adaptive_learning_rate_inv,
    generate_spin_configs,
    diagonalize_lz_multiplet,
)
from qhuantax.quantumhall_symmetries import ParticleHoleQH, FlavourPermQH, IdentityQH
from qhuantax.quantumhall_userbasis import LzUserBasisSymmetry

import json
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

parser.add_argument("--mean-field", action="store", default=None,
                    help="path to a .npz written by FuzzySphereMeanField.py; its lowest state "
                         "becomes the reference of the ansatz, replacing the 0/1 Fock U0. Files "
                         "holding more than one reference must have been built with "
                         "--orthogonalize, since only then does the lowest state span them all")

parser.add_argument("--exact-diag", action="store_true", default=False,
                    help="perform exact diagonalization and track energy, energy variance and overlap with ground state")
parser.add_argument("--lmlp-coeff", action="store", default=0,
                    help="coefficient in front of the L^- L^+ term added to the Hamiltonian")
parser.add_argument("--lmlp-freq", action="store", default=None,
                    help="how often to measure <L^- L^+>, and with it the unpenalized <H>; "
                         "0 never does and costs nothing, 1 does it every sweep for the ~20% "
                         "overhead of a second Oloc call. Independent of --lmlp-coeff: 0 with a "
                         "coefficient penalizes without measuring, a frequency without a "
                         "coefficient labels the state by L(L+1) without biasing it. "
                         "Defaults to 1 with a coefficient and 0 without")

parser.add_argument("--multi", action="store", default=None,
                    help="number of terms of the ansatz, i.e. MultiDetBackflow (or MultiPfBackflow "
                         "if --pf-backflow) rather than a single determinant; defaults to 1. With "
                         "--mean-field the count comes from the file instead, and this is only "
                         "cross-checked against it")
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
lz = int(args["lz_sect"])
z2 = int(args["z2_sect"])
ph = int(args["ph_sect"])
id = int(args["run_id"])
path = str(args["path"])
run_id = f"n_{N}_2s_{L-1}_lz_{lz}_z2_{z2}_ph_{ph}_id0{id}"

mf_file = args["mean_field"]

do_ED = bool(args["exact_diag"])
LmLp_coeff = float(args["lmlp_coeff"])
# A penalized run needs the split to report a physical energy at all, so it is on by
# default there; an unpenalized one gets the plain single-operator cost it had before.
LmLp_freq = ((1 if LmLp_coeff else 0) if args["lmlp_freq"] is None
             else int(args["lmlp_freq"]))

# None means "unset", which `--mean-field` fills in from the file; the cold start defaults to 1.
# A list is accepted for symmetry with FuzzySphereNESTrain, but only one state is trained here.
nterms_arg = (None if args["multi"] is None
              else [int(v) for v in str(args["multi"]).replace(",", " ").split()])
if nterms_arg is not None and len(nterms_arg) != 1:
    parser.error(f"--multi takes a single value here, got {len(nterms_arg)}; this script trains "
                 "one state, so use FuzzySphereNESTrain for a per-state list")
nterms = 1 if nterms_arg is None else nterms_arg[0]
pf_backflow = bool(args["pf_backflow"])
nsweeps = int(args["nbr_sweeps"])
nsamples = int(args["nbr_samples"])
nb = int(args["nbr_blocks"])
nh = int(args["nbr_heads"])
d = int(args["attn_dim"])

lr0 = float(args["lr"])
t0 = N
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
    dense_symm = LzUserBasisSymmetry(lz, z2)
    E, wf = tms_H.diagonalize(k=15, symm=dense_symm) # need to fix the number of states being requested
    print("Exact spectrum: ", E)

    wf_exact = wf[:,0]
    print("Target state norm = ",np.vdot(wf_exact, wf_exact))


# Kept out of `tms_H`: the optimizer takes it as a separate operator, so <H> and
# <L^- L^+> come back separately at every step for the same cost as the sum, and the
# exact readout below stays on the unshifted Hamiltonian.
tms_Lp = GetLpTerms(L, 2)
tms_Lm = tms_Lp.H
tms_LmLp = tms_Lm @ tms_Lp


# Mean-field reference, or the cold-start Fock matrix
mf_coeffs = None
if mf_file is not None:
    with np.load(mf_file) as mf:
        U = np.asarray(mf["U0"])
        mf_coeffs = np.asarray(mf["coeffs"])
        mf_meta = json.loads(str(mf["meta"]))
    if (mf_meta["N"], mf_meta["nm"]) != (N, L):
        raise ValueError(f"{mf_file} was built for N={mf_meta['N']}, 2s+1={mf_meta['nm']}, "
                         f"but this run is N={N}, 2s+1={L}")
    if (mf_meta["lz"], mf_meta["z2"]) != (lz, z2):
        raise ValueError(f"{mf_file} is for sector (lz,z2)=({mf_meta['lz']},{mf_meta['z2']}), "
                         f"but this run is ({lz},{z2})")
    if mf_meta["nrefs"] > 1 and not mf_meta.get("orthogonalize", True):
        # Only NES cares about the span. Here the single member *is* the state, and without
        # --orthogonalize state 0 is reference 0 alone -- a valid state, but above the root the
        # file reports, which belongs to the whole span.
        raise ValueError(
            f"{mf_file} holds {mf_meta['nrefs']} references and was built without "
            f"--orthogonalize, so its lowest state is reference 0 on its own rather than the "
            f"{mf_meta['energies'][0]:.6f} it reports. Rerun FuzzySphereMeanField.py with "
            f"--orthogonalize, or with a single-reference mode spec"
        )
    mf_coeffs = mf_coeffs[0]
    # The term count is fixed by the mode spec of the mean-field solve, so --multi can only ever
    # agree with it or disagree; treat it as an assertion rather than an input.
    nterms = int(np.count_nonzero(mf_coeffs))
    if nterms_arg is not None and nterms_arg[0] != nterms:
        raise ValueError(f"{mf_file} gives its lowest state {nterms} determinants, but "
                         f"--multi asks for {nterms_arg[0]}; the mode spec sets this, so drop "
                         f"--multi")
    print(f"mean field: {nterms} of {U.shape[0]} determinants, modes {mf_meta['modes']}, "
          f"starting energy {mf_meta['energies'][0]:.6f}")
else:
    U = np.zeros((2*L, N))
    U[:N,:N] = np.eye(N)
    if lz != 0:
        U[0,0] = 0
        U[N,lz] = 1

startTime = datetime.now()


# start NN training
if mf_coeffs is not None:
    # `U` is the determinant stack and `mf_coeffs` this state's vector over it; the builder
    # keeps only the determinants the vector actually uses, and shrinks the backflow so the
    # mean-field reference is what the run opens from.
    state = transformer_backflow_state(0, L, N, d, nb, nh, symm, pf_backflow, U,
                                       coeffs=mf_coeffs, orbital_noise=0,
                                       backflow_scale=MF_BACKFLOW_SCALE)
else:
    # A sum gets its own 1e-2 perturbation per term, so that the terms differ; a single term
    # has nothing to be separated from and takes the same perturbation as `orbital_noise`.
    state = transformer_backflow_state(0, L, N, d, nb, nh, symm, pf_backflow, U,
                                       orbital_noise=(1e-2 if nterms == 1 else 0),
                                       nterms=nterms)


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
  f.write(f"lr schedule: inv\n")
  f.write(f"t0: {t0}\n")
  f.write(f"sampler: DipoleCons\n")
  f.write(f"nbr. samples NN: {nsamples}\n")
  f.write(f"reweight: {rw}\n")
  f.write(f"model: {"PfBackflow" if pf_backflow else "DetBackflow"}\n")
  f.write(f"nterms: {nterms}\n")
  f.write(f"net: Transformer\n")
  f.write(f"nbr. blocks: {nb}\n")
  f.write(f"nbr. heads: {nh}\n")
  f.write(f"attndim: {d}\n")
  f.write(f"nbr. params: {state.nparams}\n")
  f.write(f"mean field file: {mf_file}\n")
  if mf_coeffs is not None:
      f.write(f"backflow scale: {MF_BACKFLOW_SCALE}\n")


init_configs = generate_spin_configs(L, N, lz, nsamples)
sampler = FermionTwoBodyDipoleCons(state, nsamples, n_neighbor=np.arange(1,3), initial_spins=init_configs, reweight = rw)
optimizer = PenalizedAdamSR(state, tms_H, penalty=tms_LmLp, penalty_coeff=LmLp_coeff,
                            penalty_every=LmLp_freq)

# `energy`/`VarE` describe H + lambda L^- L^+, the quantity the optimizer minimizes;
# `energy_H` is the physical <H> to compare against ED, and reads nan on the sweeps
# --lmlp-freq skips. The three coincide when --lmlp-coeff is 0.
energy = qtx.utils.DataTracer()
VarE = qtx.utils.DataTracer()
energy_H = qtx.utils.DataTracer()
VarE_H = qtx.utils.DataTracer()
LmLp_tracer = qtx.utils.DataTracer()
LmLp_var_tracer = qtx.utils.DataTracer()
if do_ED:
    overlap = qtx.utils.DataTracer()
    exact_energy = qtx.utils.DataTracer()
    exact_variance = qtx.utils.DataTracer()


for i in range(nsweeps):
    samples = sampler.sweep()
    step = optimizer.get_step(samples)
    state.update(step * adaptive_learning_rate_inv(lr0, t0, i))

    energy.append(optimizer.energy)
    VarE.append(optimizer.VarE)
    # Same batch, no extra Oloc pass: the optimizer already split the two operators.
    energy_H.append(optimizer.energy_H)
    VarE_H.append(optimizer.VarE_H)
    LmLp_tracer.append(optimizer.penalty_value)
    LmLp_var_tracer.append(optimizer.VarPenalty)

    if do_ED:
        dense = state.todense(dense_symm).normalize()

        overlap.append(abs( (dense @ wf_exact)**2 ))
        exact_energy.append((dense @ tms_H @ dense))
        exact_variance.append(((dense @ tms_H @ tms_H @ dense) - (dense @ tms_H @ dense)**2))

        np.savetxt(f"{path}/data_energy_exact_{run_id}.txt", np.vstack((exact_energy.time, exact_energy.data, exact_variance.data, overlap.data)).T)

    # Columns 0-2 are unchanged, so the existing readers keep working; the rest are
    # appended. `VarE_H` is the variance that has to vanish at convergence -- `VarE`
    # cannot, since it also carries lambda^2 Var(L^- L^+) and the covariance.
    np.savetxt(
        f"{path}/data_energy_{run_id}.txt",
        np.vstack((energy.time, energy.data, VarE.data, energy_H.data, VarE_H.data,
                   LmLp_tracer.data, LmLp_var_tracer.data)).T,
        header="sweep E_total VarE E_H VarE_H LmLp VarLmLp",
    )
    state.save(f"{path}/state_{run_id}.eqx")


print("Training completed in: ",datetime.now() - startTime)

with open(f"{path}/meta_{run_id}.txt", "a") as f:
  f.write(f"time: {datetime.now() - startTime}\n")

