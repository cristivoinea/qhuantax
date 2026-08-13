r"""Build and diagonalise a mean-field reference set for one fuzzy-sphere symmetry sector.

The sector is ``(L, Lz, Z2)`` and the ansatz is set by ``--modes``, one ``(ell,mu[,kbeta])`` entry
per *reference*, i.e. per obtainable state. ``kbeta`` is the number of Gauss-Legendre nodes that
entry's L-projection uses, and hence the determinants it costs; it defaults to 1:

    --modes "(0,0),(0,0)"                 U_HF + one tuned N00 rotation      2 dets, 2 states
    --modes "(0,0),(0,0),(1,0,2)"         the same + an L-projected N10      4 dets, 3 states
    --modes "(0,0),(0,0),(1,0,2),(1,0,2)" two such projected references      6 dets, 4 states
    --modes "(1,1),(0,0),(0,0)"           stretched L=1 exciton + two N00    3 dets, 3 states

Example:
    python FuzzySphereMeanField.py -n 12 -s 11 --l-sect 0 --lz-sect 0 --z2-sect 1 \
        --nbr-states 3 --modes "(0,0),(0,0),(1,0,2)" --path .
"""
import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import quantax as qtx

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-n", required=True, help="number of particles")
parser.add_argument("-s", required=True, help="number of orbitals minus one (2s)")
parser.add_argument("--l-sect", required=True, help="total angular momentum symmetry sector")
parser.add_argument("--lz-sect", required=True, help="Lz symmetry sector")
parser.add_argument("--z2-sect", default=1, help="Z2 symmetry sector")
parser.add_argument("--nbr-states", default=1, help="number of states to solve for")
parser.add_argument("--modes", required=True,
                    help="one (L,Lz[,kbeta]) entry per state, kbeta being that reference's "
                         "L-projection nodes (number of determinants needed); ")
parser.add_argument("--ndets", default=None,
                    help="number of determinants of each returned state")
parser.add_argument("--ps-pot", default="4.75,1",
                    help="inter-layer Haldane pseudopotentials V_0,V_1,... ; the ")
parser.add_argument("--transverse-fld", default=3.16,
                    help="transverse field h")
parser.add_argument("--tol", default=1e-2, help="Nelder-Mead tolerance on the tuned weights")
parser.add_argument("--no-residual", action="store_true", default=False,
                    help="skip the <L^2> residual, which costs one extra reduced-matrix build")
parser.add_argument("--path", required=True, help="output directory")
args = vars(parser.parse_args())

N = int(args["n"])
L = int(args["s"]) + 1
l_target = int(args["l_sect"])
lz = int(args["lz_sect"])
z2 = int(args["z2_sect"])
nstates = int(args["nbr_states"])
spec = str(args["modes"])
tol = float(args["tol"])
path = Path(str(args["path"]))
pspot_inter = np.array([float(v) for v in str(args["ps_pot"]).replace(",", " ").split()])
transverse_fld = float(args["transverse_fld"])
if N != L:
    parser.error(f"this construction is nu=1: needs N == 2s+1, got N={N}, 2s+1={L}")
if z2 not in (1, -1):
    parser.error("--z2-sect must be +1 or -1")

# The gauge rotation inside the Lz projector makes the orbitals complex, so the mean-field
# expectation has to run in complex128.
qtx.set_default_dtype(jnp.complex128)
qtx.sites.Chain(L=L, boundary=0, particle_type="spinful_fermion", Nparticles=N)

from qhuantax.quantumhall_meanfield import (  # noqa: E402  (needs the lattice above)
    hf_vacuum,
    mode_references,
    solve_modes,
)
from qhuantax.quantumhall_operators import (  # noqa: E402
    GetL2Terms,
    GetSpinfulDenIntTerms,
    GetSpinfulPolTerms,
)

S1 = np.array([[1, 0], [0, 0]])
S2 = np.array([[0, 0], [0, 1]])
SX = np.array([[0, 1], [1, 0]])

tms_H = GetSpinfulDenIntTerms(nm=L, ps_pot=2 * pspot_inter, mat_a=S1, mat_b=S2)
tms_H -= transverse_fld * GetSpinfulPolTerms(nm=L, mat=SX)

# z2 makes this variation *after* projection: the sampler always projects, so the angle that
# extremises the projected energy is the right one, not the one extremising E_HF.
vac = hf_vacuum(L, 2 * pspot_inter, transverse_fld, z2=z2)
refs, _, ndets, nfree = mode_references(vac, spec, l_target)
if args["ndets"] is not None and int(args["ndets"]) != ndets:
    parser.error(f"--modes implies {ndets} determinants, but --ndets={args['ndets']}")


print(f"  -> {ndets} determinants, {len(refs)} references, {nfree} free weights, "
      f"theta*={np.degrees(vac.theta):.4f} deg")
for i, r in enumerate(refs):
    kind = "plain" if r["kind"] == "plain" else f"L-projected N{r['ell']}0, K_beta={r['kbeta']}"
    print(f"     ref {i}: {kind}  ({r['kbeta']} det, {r['free']} free)")

result = solve_modes(
    vac, spec, l_target, lz, z2, tms_H, nstates=nstates,
    l2_terms=None if args["no_residual"] else GetL2Terms(L, 2), tol=tol,
)

print(f"\n  cond(S)={result['cond']:.2e}  kphi={result['kphi']}  weights="
      + " ".join(f"{v:+.4f}" for v in result["params"]))
print(f"  {'state':>5}  {'energy':>14}  residual |<L^2>-L(L+1)|")
for k in range(nstates):
    res = "n/a" if result["residual"] is None else f"{result['residual'][k]:.2e}"
    print(f"  {k:>5}  {result['energies'][k]:14.6f}  {res}")

name = f"meanfield_n_{N}_2s_{L-1}_l_{l_target}_lz_{lz}_z2_{z2}.npz"
path.mkdir(parents=True, exist_ok=True)
meta = dict(N=N, nm=L, L=l_target, lz=lz, z2=z2, nstates=nstates, modes=spec,
            ndets=ndets, nrefs=result["nrefs"], kphi=result["kphi"], theta=float(vac.theta),
            ps_pot=[float(v) for v in pspot_inter], transverse_fld=transverse_fld,
            params=result["params"], cond=result["cond"],
            energies=[float(e) for e in result["energies"]],
            residual=None if result["residual"] is None else [float(r) for r in result["residual"]])
np.savez(path / name, U0=result["U0"], coeffs=result["coeffs"], meta=json.dumps(meta))