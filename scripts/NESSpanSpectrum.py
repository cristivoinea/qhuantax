# L-resolved spectrum of the span of an NES run (its meta file and checkpoints): with tuples X ~ |det A|^2,
# A_ij = psi_i(x_j), the K x K matrices M_O = E[B_O A^-1]^T = S^-1 O (B_O,ij = (O psi_i)(x_j)) for O = H and L^- L^+.
# In an exact span they commute: the eigenvectors of M_L sort it by L (eigenvalue L(L+1) - Lz(Lz+1)) and M_H, in that
# basis, gives the energies of each L; the weight of M_H between different L measures how far the span is from that.
# The run's symmetries are imposed and the error bars are a jackknife over the sweeps. A FuzzySphereRotatedVMC.py meta
# (K = 1) works too: M_O is then <O>, sampled with that script's sampler. --reweight r samples |det A|^r and weights
# each tuple by its reweight factor; 2 (the default) is plain |det A|^2, 1 keeps the local energies bounded near nodes.
#
#   python NESSpanSpectrum.py --nes-meta runs/rotnes/meta_rotnes_n_20_..._K_6_..._id01.json --nbr-samples 4096 --sweeps 20
#   python NESSpanSpectrum.py --nes-meta runs/rotvmc/meta_rotvmc_n_20_..._id02.json --reweight 1
import argparse
import json
import time
from pathlib import Path

import numpy as np
import quantax as qtx
from quantax.sampler import Samples
from quantax.utils import LogArray, to_distributed_array

from qhuantax.nes import NaturalLzDetFlavourFlipSampler, NaturalStateSet
from qhuantax.quantumhall_excitons import rotated_vacuum_configs, rotated_vacuum_tuples
from qhuantax.quantumhall_operators import GetLpTerms, GetSpinfulDenIntTerms, GetSpinfulPolTerms
from qhuantax.quantumhall_samplers import FermionDipoleConsFlavourFlip
from qhuantax.quantumhall_symmetries import FlavourPermQH, fuzzy_sphere_symmetry
from qhuantax.rotated import load_member


S1 = np.array([[1, 0], [0, 0]])
S2 = np.array([[0, 0], [0, 1]])
SX = np.array([[0, 1], [1, 0]])

parser = argparse.ArgumentParser()
parser.add_argument("--nes-meta", required=True,
                    help="meta file of a FuzzySphereRotatedNES.py run, or of a FuzzySphereRotatedVMC.py one")
parser.add_argument("--ps-pot", default="4.75,1", help="inter-flavour pseudopotentials")
parser.add_argument("--nbr-samples", type=int, default=4096, help="tuples per sweep")
parser.add_argument("--sweeps", type=int, default=20, help="number of sweeps (jackknife blocks)")
parser.add_argument("--sweep-steps", type=int, default=None, help="Metropolis steps between samples; the run's")
parser.add_argument("--reweight", type=float, default=2.0, help="sample |det A|^reweight, weighted back to |det A|^2")
parser.add_argument("--seed", type=int, default=0)
args = vars(parser.parse_args())

meta_file = Path(args["nes_meta"])
meta = json.load(open(meta_file))
N, h, K, lz = meta["N"], meta["h"], meta["K"], meta.get("lz", 0)
Ls = [m.get("L", lz) for m in meta["members"]]
symmetries = meta.get("symmetries", dict(z2=meta["z2"]))
qtx.sites.Chain(L=N, boundary=0, particle_type="spinful_fermion", Nparticles=N)
pspot = np.array([float(v) for v in args["ps_pot"].split(",")])
tms_H = GetSpinfulDenIntTerms(nm=N, ps_pot=2*pspot, mat_a=S1, mat_b=S2)
if h != 0:
    tms_H = tms_H - h * GetSpinfulPolTerms(nm=N, mat=SX)
tms_Lp = GetLpTerms(N, 2)
operators = dict(H=tms_H, L=tms_Lp.H @ tms_Lp)
symm = fuzzy_sphere_symmetry(**symmetries) if "symmetries" in meta else FlavourPermQH(eigval=meta["z2"])
states = [qtx.state.Variational(load_member(m["spec"], N, meta_file.parent / m["file"]), symm=symm,
                                max_parallel=16384, use_ref=False) for m in meta["members"]]
n = args["nbr_samples"]
both = symmetries.get("z2") is not None
rng = np.random.default_rng(args["seed"])
sweep_steps = args["sweep_steps"] or meta["args"].get("sweep_steps")
if K == 1:
    # a single member: FuzzySphereRotatedVMC.py's sampler (NaturalStateSet needs two)
    init = rotated_vacuum_configs(N, n, meta["theta"], lz, both=both, rng=rng)
    sampler = FermionDipoleConsFlavourFlip(states[0], n, n_neighbor=np.arange(1, 3), initial_spins=init,
                                           reweight=args["reweight"], flip_prob=meta["args"].get("flip_prob", 0.5),
                                           sweep_steps=sweep_steps)
else:
    init = rotated_vacuum_tuples(N, n, K, meta["theta"], lz, both=both, rng=rng)
    sampler = NaturalLzDetFlavourFlipSampler(NaturalStateSet(states), n, n_neighbor=np.arange(1, 3),
                                             initial_spins=init, reweight=args["reweight"], sweep_steps=sweep_steps)
print(f"N={N}, Lz={lz}, K={K}, members' L {Ls}, symmetries {symmetries}; {n} tuples x {args['sweeps']} sweeps, "
      f"reweight {args['reweight']:g}", flush=True)


def logval(p):
    p = p if isinstance(p, LogArray) else LogArray.from_value(p)
    return np.asarray(p.sign, float), np.asarray(p.logabs, float)


shift = None                    # per member, fixed over the sweeps: a constant similarity of M, unlike per-sample ones
blocks = {name: [] for name in operators}
t_start = time.time()
for sweep in range(args["sweeps"]):
    samples = sampler.sweep()
    spins = np.asarray(samples.spins)
    rw = np.ones(n) if samples.reweight_factor is None else np.asarray(samples.reweight_factor, float).reshape(n)
    flat = to_distributed_array(spins.reshape(n * K, 2 * N))
    sign, logabs, oloc = [], [], {name: [] for name in operators}
    for state in states:
        psi = state(flat)
        s, l = logval(psi)
        sign.append(s); logabs.append(l)
        for name, op in operators.items():
            oloc[name].append(np.real(np.asarray(op.Oloc(state, Samples(flat, psi)))))
    sign, logabs = np.array(sign), np.array(logabs)                 # (K members, n K configurations)
    if shift is None:
        shift = logabs.max(1, keepdims=True)
    la = logabs - shift
    la -= la.max(0, keepdims=True)                                  # per configuration: cancels in B A^-1
    A = (sign * np.exp(la)).reshape(K, n, K).transpose(1, 0, 2)     # (n, i member, j configuration)
    for name in operators:
        B = A * np.array(oloc[name]).reshape(K, n, K).transpose(1, 0, 2)
        G = np.linalg.solve(A.transpose(0, 2, 1), B.transpose(0, 2, 1)).transpose(0, 2, 1)     # B A^-1
        ok = np.all(np.isfinite(G), axis=(1, 2)) & np.isfinite(rw)
        blocks[name].append(np.einsum("t,tij->ij", rw[ok], G[ok]).T / rw[ok].sum())
    print(f"  sweep {sweep}: tr M_H {np.trace(blocks['H'][-1]):.4f}, tr M_L {np.trace(blocks['L'][-1]):.4f} "
          f"[{time.time() - t_start:.0f}s]", flush=True)


def analyse(MH, ML):
    """Eigenvalues of M_L -> L of each direction; M_H in that basis, its blocks of equal L -> energies; the share of
    M_H between different L."""
    w, V = np.linalg.eig(ML)
    w, V = np.real(w), np.real_if_close(V)
    L = (-1 + np.sqrt(1 + 4*np.clip(w + lz * (lz + 1), 0, None))) / 2
    T = np.real(np.linalg.solve(V, MH @ V))
    groups = {int(round(l)): np.flatnonzero(np.round(L) == round(l)) for l in L}
    energies = {l: np.sort(np.real(np.linalg.eigvals(T[np.ix_(g, g)]))) for l, g in sorted(groups.items())}
    same = np.zeros_like(T, bool)
    for g in groups.values():
        same[np.ix_(g, g)] = True
    return L, energies, float(np.linalg.norm(T[~same]) / np.linalg.norm(T))


MH, ML = np.mean(blocks["H"], 0), np.mean(blocks["L"], 0)
L, energies, mixing = analyse(MH, ML)
nb = len(blocks["H"])
jack = [analyse(np.mean(np.delete(blocks["H"], b, 0), 0), np.mean(np.delete(blocks["L"], b, 0), 0)) for b in range(nb)]
trH = np.array([np.trace(M) for M in blocks["H"]])
print(f"\ntr M_H = {trH.mean():.5f} +- {trH.std(ddof=1) / np.sqrt(nb):.5f} (the trace NES minimises)")
print(f"L of the span's directions: {np.round(np.sort(L), 3)} (members' L: {sorted(Ls)})")
print(f"M_H between different L: {mixing:.2e} of its norm")
result = dict(nes_meta=str(meta_file), nbr_samples=n, sweeps=nb, reweight=args["reweight"], trH=float(trH.mean()),
              trH_err=float(trH.std(ddof=1) / np.sqrt(nb)), L=np.sort(L).tolist(), mixing=mixing, energies={})
for l, e in energies.items():
    same = [j[1][l] for j in jack if l in j[1] and len(j[1][l]) == len(e)]
    err = np.sqrt((len(same) - 1) / len(same) * np.sum((np.array(same) - np.mean(same, 0))**2, 0)) if len(same) == nb \
        else np.full(len(e), np.nan)
    print(f"  L={l}: " + ", ".join(f"{v:.5f} +- {s:.5f}" for v, s in zip(e, err)))
    result["energies"][str(l)] = dict(E=e.tolist(), err=err.tolist())
out = meta_file.with_name(meta_file.name.replace("meta_", "spectrum_")
                          .replace(".json", ".json" if args["reweight"] == 2 else f"_rw{args['reweight']:g}.json"))
with open(out, "w") as f:
    json.dump(result, f, indent=1)
print(f"written {out}")
