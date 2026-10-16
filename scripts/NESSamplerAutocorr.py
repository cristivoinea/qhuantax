# Autocorrelation of the NES sampler per Monte Carlo step, on the members of an NES run (its meta file):
#   joint       the current move: every configuration of a tuple proposed at once, K^2 network evaluations per step
#   single-row  one configuration per step, K evaluations per step in an efficient implementation (here evaluated
#               with the full determinant; only its autocorrelation in steps is measured)
# Recorded per chain: each configuration's |k - N/2|, k its down-electron count (changed only by flavour flips; the
# Z2-invariant form, since Z2 maps k to N-k and hops between the two mirror modes are irrelevant), raw k for
# comparison, log|det A|, and the tuple's local energy tr(A^-1 B) every --eloc-every steps. Prints the integrated
# autocorrelation time of each, in steps and in network evaluations per chain, and the acceptance.
# --sweep-test S1,S2,...: whole sweeps of S steps on the frozen members, as in training: per sweep the energy and the
# raw SR step (plain updater, the run's lambda). Prints the time per sweep, the lag-1 correlation of the sweep
# energies, and the cosine between consecutive steps against steps 5-10 sweeps apart: the true gradient is fixed,
# so an excess at lag 1 is noise shared by consecutive sweeps.
#
#   python NESSamplerAutocorr.py --nes-meta runs/rotnes/meta_rotnes_..._id01.json --nbr-samples 512 --steps 400
#   python NESSamplerAutocorr.py --nes-meta runs/rotnes/meta_rotnes_..._id01.json --nbr-samples 2048 --steps 0 \
#       --sweep-test 224,112
import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import quantax as qtx
from quantax.global_defs import get_subkeys
from quantax.sampler import Samples
from quantax.utils import LogArray

from qhuantax.nes import NaturalExcitedSR, NaturalLzDetFlavourFlipSampler, NaturalStateSet
from qhuantax.nes.optimizer import NaturalTraceEnergyGrad
from qhuantax.quantumhall_excitons import KSectors
from qhuantax.quantumhall_operators import GetLpTerms, GetSpinfulDenIntTerms, GetSpinfulPolTerms
from qhuantax.quantumhall_symmetries import FlavourPermQH, fuzzy_sphere_symmetry
from qhuantax.rotated import load_member


S1 = np.array([[1, 0], [0, 0]])
S2 = np.array([[0, 0], [0, 1]])
SX = np.array([[0, 1], [1, 0]])


parser = argparse.ArgumentParser()
parser.add_argument("--nes-meta", required=True, help="meta file of a FuzzySphereRotatedNES.py run")
parser.add_argument("--ps-pot", default="4.75,1", help="inter-flavour pseudopotentials")
parser.add_argument("--nbr-samples", type=int, default=512, help="number of chains")
parser.add_argument("--steps", type=int, default=400, help="joint-move steps; the single-row run makes K times as many")
parser.add_argument("--eloc-every", type=int, default=8, help="joint steps between local energies (K times for rows)")
parser.add_argument("--reweight", type=float, default=2.0, help="sample from |det A|^reweight")
parser.add_argument("--flip-prob", type=float, default=0.5, help="fraction of flavour-flip proposals")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default=None, help="npz file for the recorded series")
parser.add_argument("--sweep-test", default="", help="comma-separated sweep lengths to compare on frozen members")
parser.add_argument("--test-sweeps", type=int, default=40, help="sweeps per sweep length in --sweep-test")
parser.add_argument("--lmlp-coeff", type=float, default=None, help="--sweep-test penalty; defaults to the run's")
args = vars(parser.parse_args())

meta = json.load(open(args["nes_meta"]))
N, lz, z2, h, K = meta["N"], meta.get("lz", 0), meta["z2"], meta["h"], meta["K"]
n = args["nbr_samples"]
pspot_inter = np.array([float(v) for v in args["ps_pot"].split(",")])
lattice = qtx.sites.Chain(L=N, boundary=0, particle_type="spinful_fermion", Nparticles=N)

tms_H = GetSpinfulDenIntTerms(nm=N, ps_pot=2*pspot_inter, mat_a=S1, mat_b=S2)
if h != 0:
    tms_H = tms_H - h * GetSpinfulPolTerms(nm=N, mat=SX)
# the run's symmetries; Z2 alone was always imposed before they became options
symm = fuzzy_sphere_symmetry(**meta["symmetries"]) if "symmetries" in meta else FlavourPermQH(eigval=z2)
D = Path(args["nes_meta"]).parent
members = [qtx.state.Variational(load_member(m["spec"], N, D / m["file"]), symm=symm, max_parallel=16384, use_ref=False)
           for m in meta["members"]]
state_set = NaturalStateSet(members)
grad = NaturalTraceEnergyGrad(tms_H)

ksectors = KSectors(N, lz)
rng = np.random.default_rng(args["seed"])
k0 = int(lz != 0)                                   # k = 0 is empty at Lz != 0
init = np.stack([ksectors.configs(k)[rng.integers(ksectors.dim(k), size=n)] for k in range(k0, k0 + K)], axis=1)
t = time.time()
sampler = NaturalLzDetFlavourFlipSampler(state_set, n, n_neighbor=np.arange(1, 3), initial_spins=init,
                                         reweight=args["reweight"], flip_prob=args["flip_prob"])
spins0 = sampler._spins
print(f"N={N}, K={K}, {n} chains, reweight {args['reweight']:g}: thermalised in {time.time() - t:.0f}s", flush=True)


def joint_step(keyp, keyu, samples):
    return sampler._single_sweep(keyp, keyu, samples)


def single_row_step(keyp, keyu, samples):
    """The joint proposal kept on one random row per chain: a symmetric single-configuration move."""
    proposal = sampler.propose(keyp, samples.spins)
    row = jr.randint(jr.fold_in(keyp, 1), (n,), 0, K)
    trial = jnp.where((jnp.arange(K)[None, :] == row[:, None])[..., None], proposal, samples.spins)
    return sampler._update(keyu, None, samples, Samples(trial, sampler.determinant(trial)))


def run(step, nsteps, eloc_every):
    samples = Samples(spins0, sampler.determinant(spins0))
    k, logdet, eloc, moved, valid = [], [], [], 0.0, 0.0
    keys = get_subkeys(2 * nsteps)
    for s in range(nsteps):
        samples = step(keys[2 * s], keys[2 * s + 1], samples)
        moved, valid = moved + float(sampler._step_moved), valid + float(sampler._step_valid)
        k.append(np.asarray(jnp.sum(samples.spins[..., N:] > 0, axis=-1)))
        logdet.append(np.asarray(LogArray.from_value(samples.psi).logabs))
        if s % eloc_every == 0:
            eloc.append(np.real(np.asarray(grad.local_energies(state_set, samples)[0])))
    k = np.array(k).reshape(nsteps, -1)
    return dict(kz=np.abs(k - N / 2), k=k, logdet=np.array(logdet), eloc=np.array(eloc),
                acceptance=moved / max(valid, 1e-12))


def tau_int(x, spacing=1):
    """Integrated autocorrelation time of the chains x (time, chain) in steps, Sokal's window (W >= 5 tau)."""
    x = x - np.nanmean(x)
    var = np.nanmean(x * x)
    tau = 0.5
    for lag in range(1, x.shape[0] // 2):
        tau += np.nanmean(x[:-lag] * x[lag:]) / var
        if lag >= 5 * tau:
            break
    return tau * spacing


if args["steps"] > 0:
    results, series = {}, {}
    for name, step, nsteps, every, evals in (("joint", joint_step, args["steps"], args["eloc_every"], K * K),
                                             ("single-row", single_row_step, K * args["steps"], K * args["eloc_every"], K)):
        t = time.time()
        r = run(step, nsteps, every)
        taus = dict(kz=tau_int(r["kz"]), logdet=tau_int(r["logdet"]), eloc=tau_int(r["eloc"], every),
                    k_raw=tau_int(r["k"]))
        series[name] = r
        results[name] = dict(acceptance=r["acceptance"], tau_steps=taus, evals_per_step=evals)
        print(f"{name:10s} ({nsteps} steps, {time.time() - t:.0f}s): acceptance {r['acceptance']:.3f}; tau_int in steps: "
              + ", ".join(f"{o} {v:.1f}" for o, v in taus.items())
              + f"; in network evaluations per chain ({evals}/step): "
              + ", ".join(f"{o} {v * evals:.0f}" for o, v in taus.items()), flush=True)

    if args["out"]:
        np.savez(args["out"], **{f"{name}_{o}": v for name, r in series.items() for o, v in r.items()})
    slowest = {name: max(v for o, v in r["tau_steps"].items() if o != "k_raw") for name, r in results.items()}
    print(f"slowest tau: joint {slowest['joint']:.1f} steps -> --sweep-steps ~{2 * slowest['joint']:.0f} (now "
          f"{2 * K * 2 * N}); cost per independent sample, single-row / joint: "
          f"{slowest['single-row'] * K / (slowest['joint'] * K * K):.2f}")


# --sweep-test: the per-sweep estimates of training, for each sweep length, on the same frozen members
lam = (meta.get("lam") if args["lmlp_coeff"] is None else args["lmlp_coeff"]) or 0.0
tms_Lp = GetLpTerms(N, 2)
for S in [int(v) for v in args["sweep_test"].split(",") if v.strip()]:
    sampler._spins = spins0
    optimizer = NaturalExcitedSR(state_set, tms_H, updater=qtx.optimizer.PlainUpdater(), penalty=tms_Lp.H @ tms_Lp,
                                 penalty_coeff=lam)
    energies, steps = [], []
    t = time.time()
    for _ in range(args["test_sweeps"]):
        step = optimizer.get_step(sampler.sweep(S))
        energies.append(float(optimizer.energy_H))
        steps.append(np.asarray(step))
    per_sweep = (time.time() - t) / args["test_sweeps"]
    u = np.array(steps)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    cos = lambda lag: np.mean(np.sum(u[:-lag] * u[lag:], axis=1))
    e = np.array(energies) - np.mean(energies)
    print(f"sweep-steps {S:4d}: {per_sweep:.1f} s per sweep; lag-1 correlation of tr E_H {np.mean(e[:-1] * e[1:]) / np.mean(e * e):+.2f}; "
          f"cos(SR steps) lag 1 {cos(1):.3f} vs lags 5-10 {np.mean([cos(l) for l in range(5, 11)]):.3f} "
          f"(lambda {lam:g}, {args['test_sweeps']} sweeps)", flush=True)
