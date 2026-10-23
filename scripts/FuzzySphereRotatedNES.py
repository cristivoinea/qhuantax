# NES at transverse field h from the rotated exciton members of FuzzySphereRotatedAnsatz.py, in its Lz sector (--lz),
# each member with its own L >= Lz (--L-sect there). The members are released before training: untied (the features may
# then see the flavour; no amplitude changes), the k=2 up-feature suppression lowered to exp(-gap), and the k=0,1
# determinants (W = 0, where the network has no gradient) given a small random backflow; the angle trains. The
# L^- L^+ penalty vanishes on L = Lz and lifts a level of L > Lz by lambda [L(L+1) - Lz(Lz+1)], so it suits members that
# all have L = Lz; with members of higher L (e.g. K=6 at Lz=0: four L=0 and the two L=2 levels below them) leave it off
# and separate the converged span by L with NESSpanSpectrum.py. --lmlp-coeff auto takes lambda from the ED levels at h_c
# (misc/iqh_gaps.py). Symmetries are imposed only when their quantum number is given: --z2-sect (Ising Z2), --ry-sect
# (pi rotation about y, (-1)^L, Lz = 0 only) and --ry-ph-sect (particle-hole x Ry, P (-1)^L at any Lz, P as in
# PRX 13, 021009). --exact-diag tracks the members against the lowest levels of their L's in the same sectors.
#
#   python FuzzySphereRotatedNES.py --ansatz-meta runs/rot/meta_n_8_2s_7_lz_0_z2_1_ryph_0_h_3.16_id04.json \
#       --lmlp-coeff 3.0 --exact-diag --nbr-sweeps 200 --path runs/rotnes
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import quantax as qtx
import scipy as sp
from quantax.global_defs import get_subkeys
from quantax.nn import lecun_normal
from quantax.utils import LogArray

from qhuantax.nes import NaturalExcitedSR, NaturalLzDetFlavourFlipSampler, NaturalStateSet
from qhuantax.nes.optimizer import NaturalTraceEnergyGrad
from qhuantax.quantumhall_excitons import KSectors, rotated_vacuum_tuples
from qhuantax.quantumhall_operators import GetL2Terms, GetLpTerms, GetSpinfulDenIntTerms, GetSpinfulPolTerms
from qhuantax.quantumhall_symmetries import PHRyQH, RyQH, fuzzy_sphere_symmetry
from qhuantax.quantumhall_userbasis import LzUserBasisSymmetry
from qhuantax.quantumhall_utils import adaptive_learning_rate_inv
from qhuantax.rotated import load_member, member_spec, set_gap, untie


S1 = np.array([[1, 0], [0, 0]])
S2 = np.array([[0, 0], [0, 1]])
SX = np.array([[0, 1], [1, 0]])


parser = argparse.ArgumentParser()
parser.add_argument("--ansatz-meta", required=True, help="meta file written by FuzzySphereRotatedAnsatz.py")
parser.add_argument("--h", type=float, default=None, help="transverse field; defaults to the ansatz's")
parser.add_argument("--z2-sect", type=int, default=None, choices=(1, -1),
                    help="impose the Ising Z2 with this eigenvalue")
parser.add_argument("--ry-sect", type=int, default=None, choices=(1, -1),
                    help="impose the pi rotation about y, (-1)^L; Lz = 0 only, where +1 removes every odd L")
parser.add_argument("--ry-ph-sect", type=int, default=None, choices=(1, -1),
                    help="impose particle-hole x Ry, P (-1)^L at any Lz (the Ising ground state is +1)")
parser.add_argument("--int-cutoff", type=float, default=1e-15,
                    help="train without the interaction hopping terms below this |U|; their summed |U| (printed) "
                         "bounds the shift of every energy. ED keeps the full H")
parser.add_argument("--no-untie", action="store_true", help="keep the features charge-only")
parser.add_argument("--det-backflow-scale", type=float, default=0.1,
                    help="k=0,1 members: W = this x the default random init; 0 keeps W = 0")
parser.add_argument("--gap", type=float, default=8.0,
                    help="k=2 members: up features suppressed by exp(-gap); <= 0 keeps the ansatz's")
parser.add_argument("--lmlp-coeff", default="0",
                    help="coefficient lambda of the L^- L^+ penalty, or 'auto': the smallest lambda that keeps the "
                         "lowest level of every L > Lz --lmlp-margin above the K-th L = Lz one, and at least the "
                         "--lmlp-floor-gap floor; from the ED levels at h = 3.16 of misc/iqh_gaps.py (N = 7-14, other "
                         "N extrapolated). Every member must have L = Lz")
parser.add_argument("--lmlp-margin", type=float, default=1.0,
                    help="auto: energy by which the L > Lz levels must clear the K targets; the a + b/sqrt(N) "
                         "extrapolation of the ED gaps misses by ~0.1-0.5 already from N = 9-12 to 14")
parser.add_argument("--lmlp-floor-gap", type=float, default=20.0,
                    help="auto: floor on lambda, kept when the levels need less, as the energy by which it lifts a "
                         "random state of the Lz sector, taken for the L > Lz admixture the members pick up in "
                         "training (mean L(L+1) - Lz(Lz+1) 241 at N=14, Lz=0, so 20 gives 0.083; ~N^3); 0 disables")
parser.add_argument("--lmlp-every", type=int, default=None,
                    help="measure <L^- L^+> every this many sweeps, also at lambda = 0 (the gradient is unchanged; "
                         "~10-20%% per measured sweep); 0 never. Defaults to 1 with a penalty, 0 without")
parser.add_argument("--exact-diag", action="store_true", help="track the members against ED (feasible up to N=14)")
parser.add_argument("--ed-every", type=int, default=5, help="ED readout every this many sweeps")
parser.add_argument("--ed-k", type=int, default=10, help="lowest levels of the Lz block to compute, all L")
parser.add_argument("--ed-dir", default=None, help="ED cache folder, searched first; defaults to --path")
parser.add_argument("--nbr-sweeps", type=int, default=200, help="number of training iterations")
parser.add_argument("--nbr-samples", type=int, default=512, help="number of sample tuples per iteration")
parser.add_argument("--lr", type=float, default=1e-2, help="starting value of the learning rate")
parser.add_argument("--lr-t0", type=float, default=None, help="lr / (1 + sweep / lr_t0), 0 for a constant lr; defaults to N")
parser.add_argument("--norm-clip", type=float, default=0.0,
                    help="cap on the step norm before the learning rate; <= 0 disables")
parser.add_argument("--rshift", type=float, default=None,
                    help="relative diagonal shift of the SR solve; defaults to quantax's")
parser.add_argument("--max-fs", type=float, default=0.0,
                    help="trust region: cap on 1 - F per sweep, F the fidelity of the spans before and after the "
                         "update, to second order dtheta^T S dtheta with S from the previous sweep; 0 disables")
parser.add_argument("--warmup-fs", type=float, default=0.0,
                    help="the trust region (in place of --max-fs) for the first --warmup-sweeps sweeps of a fresh run, "
                         "whose SR steps are mostly noise (their 1 - F 10-1000x larger on an independent batch than on "
                         "their own) and push the members out of their L, e.g. 1e-4; not applied with --cont-run; "
                         "0 (the default) disables")
parser.add_argument("--warmup-sweeps", type=int, default=None, help="sweeps under --warmup-fs; defaults to 2N")
parser.add_argument("--max-dL", type=float, default=0.0,
                    help="L trust region: cap on the L^- L^+ a step creates, the part of its change of "
                         "tr(S^-1 L^- L^+) (the trLmLp column's units) quadratic in the step, measured like fs on the "
                         "previous sweep's batch from the parameters before, after and before minus the step; "
                         "applied after --max-fs to the step that one left. Costs 3 evaluations of the L^- L^+ trace "
                         "per sweep; 0 disables")
parser.add_argument("--max-dL-samples", type=int, default=0,
                    help="--max-dL: measure on this many tuples of the previous batch (cost in proportion); 0 all")
parser.add_argument("--diagnostics", action="store_true",
                    help="record the SR matrix spectrum every 5 sweeps and the sampler's acceptance every sweep")
parser.add_argument("--profile", action="store_true",
                    help="time each part of a sweep into data_profile_<run_id>.txt")
parser.add_argument("--thermal-steps", type=int, default=None,
                    help="sampler thermalisation steps; defaults to the sampler's")
parser.add_argument("--sweep-steps", type=int, default=None,
                    help="Metropolis steps between samples; defaults to the sampler's 2 K 2N. Check it with "
                         "NESSamplerAutocorr.py --sweep-test")
parser.add_argument("--reweight", type=float, default=1.0, help="reweight factor for training sampling")
parser.add_argument("--flip-prob", type=float, default=0.5, help="fraction of flavour-flip proposals")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--run-id", type=int, default=1)
parser.add_argument("--cont-run", type=int, default=None,
                    help="run id under --path to continue from: its members as they are (no release); the sweep "
                         "counter, the lr schedule and Adam restart, so --lr is the rate this run opens with")
parser.add_argument("--path", required=True, help="output directory")
args = vars(parser.parse_args())

ansatz = json.load(open(args["ansatz_meta"]))
N, lz, z2, nstates = ansatz["N"], ansatz["lz"], ansatz["z2"], len(ansatz["members"])
Ls = [m.get("L", lz) for m in ansatz["members"]]        # older ansatz metas: every member has L = Lz
mixed_L = set(Ls) != {lz}
h = ansatz["h"] if args["h"] is None else args["h"]
pspot_inter = np.array([float(v) for v in ansatz["args"]["ps_pot"].split(",")])
nsamples, max_fs, max_dL, do_ED, path = (args[k] for k in ("nbr_samples", "max_fs", "max_dL", "exact_diag", "path"))
t0 = N if args["lr_t0"] is None else args["lr_t0"]


sectors = dict(z2=args["z2_sect"], ry=args["ry_sect"], phry=args["ry_ph_sect"])


def run_name(run):
    return f"rotnes_n_{N}_2s_{N-1}_lz_{lz}_z2_{z2}_ryph_{sectors['phry'] or 0}_K_{nstates}_h_{h:g}_id0{run}"


run_id = run_name(args["run_id"])
OPTION = dict(z2="--z2-sect", ry="--ry-sect", phry="--ry-ph-sect")
if sectors["ry"] is not None and (lz != 0 or sectors["ry"] == -1):
    parser.error("--ry-sect: Ry maps Lz to -Lz, so it exists at Lz = 0 only, where the L = 0 members are +1")
fit_ry_ph = ansatz.get("ry_ph")         # the ansatz fitted its members projected onto this PH x Ry sector
if fit_ry_ph is not None and sectors["phry"] != fit_ry_ph:
    parser.error(f"the ansatz fitted its members as PH x Ry {fit_ry_ph:+d} projected models, which only --ry-ph-sect "
                 f"{fit_ry_ph:+d} reproduces: unprojected, they carry an unfitted part in the other sector")
if sectors["z2"] is not None and sectors["z2"] != z2:
    print(f"warning: --z2-sect {sectors['z2']:+d}, but the ansatz was made for Z2 = {z2:+d} (older ansatz runs were "
          "rotated by that sector's vacuum angle, newer ones by the Z2 = +1 angle in every sector)", flush=True)

lmlp_auto = args["lmlp_coeff"].strip().lower() == "auto"
if lmlp_auto:
    if mixed_L:
        parser.error(f"--lmlp-coeff auto keeps the L > Lz levels out of the window, but the members have L = {Ls}")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "misc"))
    from iqh_gaps import H_FIELD, PS_POT, min_lambda, random_state_LL
    if not np.isclose(h, H_FIELD) or tuple(pspot_inter.tolist()) != PS_POT:
        parser.error(f"--lmlp-coeff auto reads the ED levels at h = {H_FIELD}, V = {PS_POT}; this run has h = {h}, "
                     f"V = {ansatz['args']['ps_pot']}")
    try:
        lam_need = min_lambda(N, lz, nstates, Z=sectors["z2"], ry_ph=sectors["phry"], ry=sectors["ry"],
                              margin=args["lmlp_margin"])
    except ValueError as err:
        parser.error(f"--lmlp-coeff auto: {err}")
    lam_floor = args["lmlp_floor_gap"] / random_state_LL(N, lz)
    lam = max(lam_need, lam_floor)
    print(f"--lmlp-coeff auto: lambda {lam:g} (the levels need {lam_need:g} at margin {args['lmlp_margin']:g}, "
          f"floor {lam_floor:g} from gap {args['lmlp_floor_gap']:g})", flush=True)
else:
    lam = float(args["lmlp_coeff"])
lmlp_every = int(lam > 0) if args["lmlp_every"] is None else args["lmlp_every"]
if lam <= 0 and not lmlp_auto and not mixed_L and (lz != 0 or nstates >= 3):
    print("warning: no --lmlp-coeff, so nothing keeps the L > Lz levels out of the window (at Lz = 0 and h_c, K >= 3 "
          "already reaches some)", flush=True)
if lam > 0 and mixed_L:
    print(f"warning: --lmlp-coeff also lifts the members of L > Lz (L = {Ls}); the targets become the lowest K levels "
          "of H + lambda L^- L^+", flush=True)
if max_dL > 0 and mixed_L:
    print(f"warning: --max-dL caps rises of tr(S^-1 L^- L^+), but with members of L > Lz (L = {Ls}) a leak of theirs "
          "to a lower L lowers it and passes", flush=True)
if args["max_dL_samples"] > 0:
    # whole tuples per JAX device, as the full batch has
    args["max_dL_samples"] = max(jax.device_count(), min(args["max_dL_samples"], nsamples)
                                 // jax.device_count() * jax.device_count())

cont_run_id = None if args["cont_run"] is None else run_name(args["cont_run"])
if cont_run_id is not None:
    if cont_run_id == run_id:
        parser.error(f"--cont-run {args['cont_run']} would overwrite the run it reads from; "
                     "give this one a new --run-id")
    cont_meta_file = Path(f"{path}/meta_{cont_run_id}.json")
    if not cont_meta_file.is_file():
        parser.error(f"--cont-run {args['cont_run']}: {cont_meta_file} is missing")
    cont_meta = json.load(open(cont_meta_file))

Path(path).mkdir(parents=True, exist_ok=True)

lattice = qtx.sites.Chain(L=N, boundary=0, particle_type="spinful_fermion", Nparticles=N)
symm = fuzzy_sphere_symmetry(**sectors)
symm_label = ", ".join(f"{OPTION[name]} {e:+d}" for name, e in sectors.items() if e is not None) or "none"
ksectors = KSectors(N, lz)


def member_eigenvalue(member, U, project=fit_ry_ph is not None):
    """<U> of an ansatz member at h=0, an exact sum over its sector k, which Ry and PH x Ry map onto itself: the
    eigenvalue for an exact member; the projection onto the other sector would keep (1 - <U>)/2 of its weight. With
    `project`, of the PH x Ry projected model the ansatz fitted and NES uses, psi_P(x) = [psi(x) + r sign psi(x')]/2."""
    model = load_member(member["spec"], N, Path(args["ansatz_meta"]).parent / member["file"]).rebuild(theta=0.0)
    s = jnp.asarray(ksectors.configs(member["k"]))
    t, sign = jax.vmap(U.images)(s)

    def terms(x):                       # (coefficient, configurations) whose amplitudes add up to the member's at x
        if not project:
            return [(1.0, x)]
        y, sign_y = jax.vmap(PHRyQH(1).images)(x)
        return [(0.5, x), (0.5 * fit_ry_ph * sign_y[:, 1], y[:, 1])]

    parts = [(c, jax.vmap(model)(x)) for x0 in (s, t[:, 1]) for c, x in terms(x0)]
    top = max(jnp.max(p.logabs) for _, p in parts)
    vals = [c * p.sign * jnp.exp(p.logabs - top) for c, p in parts]
    a, b = sum(vals[:len(vals) // 2]), sum(vals[len(vals) // 2:])
    return float(jnp.sum(a * sign[:, 1] * b) / jnp.sum(a * a))


# The members must lie in the imposed Ry and PH x Ry sectors, or the projection removes them (R(theta) commutes with
# both). Members the ansatz fitted projected are taken projected, so their PH x Ry needs no check; printed instead is
# the share of the network's weight the projection keeps (a small share is a cancellation that float32 resolves poorly)
member_eigvals = {}
if cont_run_id is None and fit_ry_ph is not None:
    kept = member_eigvals["phry_kept"] = [(1 + fit_ry_ph * member_eigenvalue(m, PHRyQH(1), project=False)) / 2
                                          for m in ansatz["members"]]
    print(f"members fitted as PH x Ry {fit_ry_ph:+d} projected models: the projection keeps {np.round(kept, 3)} of the "
          "networks' weight", flush=True)
for name, U in (("ry", RyQH(1)), ("phry", PHRyQH(1))):
    if cont_run_id is None and sectors[name] is not None and not (name == "phry" and fit_ry_ph is not None):
        ev = member_eigvals[name] = [member_eigenvalue(m, U) for m in ansatz["members"]]
        print(f"{OPTION[name]}: eigenvalue of the ansatz members at h=0 {np.round(ev, 4)}", flush=True)
        off = [i for i, e in enumerate(ev) if e * sectors[name] < 0.5]
        if off:
            raise SystemExit(f"{OPTION[name]} {sectors[name]:+d}: members {off} lie mostly in the other sector "
                             "and the projection would remove them")


def hamiltonian(cutoff=1e-15):
    tms, dropped = GetSpinfulDenIntTerms(nm=N, ps_pot=2*pspot_inter, mat_a=S1, mat_b=S2, cutoff=cutoff,
                                         return_dropped=True)
    return (tms if h == 0 else tms - h * GetSpinfulPolTerms(nm=N, mat=SX)), dropped


tms_H, int_dropped = hamiltonian(args["int_cutoff"])
if int_dropped > 0:
    print(f"interaction hopping terms below {args['int_cutoff']:g} dropped: every energy shifts by at most "
          f"{int_dropped:.1e}", flush=True)
# a separate operator for the optimizer, so <H> and <L^- L^+> come back separately
tms_Lp = GetLpTerms(N, 2)
tms_LmLp = tms_Lp.H @ tms_Lp


def release(member):
    model = load_member(member["spec"], N, Path(args["ansatz_meta"]).parent / member["file"])
    if not args["no_untie"]:
        model = untie(model)
    if member["k"] >= 2 and args["gap"] > 0:               # the block fits
        model = set_gap(model, args["gap"])
    if member["k"] < 2 and args["det_backflow_scale"] > 0 and not jnp.any(model.W):    # the exact determinant(s)
        # relative to U0, whose std is 1 unless the ansatz script scaled the member to unit norm
        W = lecun_normal(get_subkeys(), model.W.shape, dtype=model.W.dtype) / 10 * float(jnp.std(model.U0))
        model = eqx.tree_at(lambda m: m.W, model, args["det_backflow_scale"] * W)
    return model


if cont_run_id is None:
    models = [release(member) for member in ansatz["members"]]
    start = (f"theta {np.degrees(ansatz['theta']):.4f} deg, {'charge-only' if args['no_untie'] else 'untied'}, "
             f"k=2 gap {args['gap'] if args['gap'] > 0 else 'unchanged'}")
else:
    models = [load_member(m["spec"], N, f"{path}/{m['file']}") for m in cont_meta["members"]]
    cont_lam = cont_meta.get("lam_bump", {}).get("lam_to", cont_meta["lam"])     # older runs could double lambda
    start = f"continued from {cont_run_id} (lambda {cont_lam:g} at its end)"
    if lam < cont_lam:
        print(f"warning: --lmlp-coeff {lam:g} is below the lambda {cont_lam:g} that {cont_run_id} ended with",
              flush=True)
member_states = [qtx.state.Variational(model, symm=symm, max_parallel=16384, use_ref=False) for model in models]
state_set = NaturalStateSet(member_states)
print(f"N={N}, Lz={lz}, h={h}, K={nstates}, symmetries imposed: {symm_label}; members (k, L) = "
      f"{[(m['k'], L) for m, L in zip(ansatz['members'], Ls)]}, {start}; {state_set.nparams_per_state} parameters per "
      "member", flush=True)

# Chain seeds distributed like the rotated vacuum (and its Z2 image when Z2 is imposed), the K of a tuple distinct
init_configs = rotated_vacuum_tuples(N, nsamples, nstates, ansatz["theta"], lz, both=sectors["z2"] is not None,
                                     rng=np.random.default_rng(args["seed"]))
sampler = NaturalLzDetFlavourFlipSampler(state_set, nsamples, n_neighbor=np.arange(1, 3),
                                         initial_spins=init_configs, reweight=args["reweight"],
                                         flip_prob=args["flip_prob"], thermal_steps=args["thermal_steps"],
                                         sweep_steps=args["sweep_steps"])


def exact_levels():
    """The --ed-k lowest levels of the block (all L, L from <L^2>), sorted: read from ed_dir or computed and cached
    there. An older cache of L = lz levels only (..._nst_<n>.npz) serves when every member has L = lz."""
    base = f"ed_n_{N}_2s_{N-1}_lz_{lz}_z2_{sectors['z2'] or 0}_h_{h:g}_V_{ansatz['args']['ps_pot'].replace(',', '_')}"
    f = ed_dir / f"{base}_nev_{args['ed_k']}.npz"
    if f.is_file():
        with np.load(f) as cached:
            return cached["E"], cached["V"], cached["L"], f"read from {f}"
    if not mixed_L:
        for g in sorted(ed_dir.glob(f"{base}_nst_*.npz")):
            n = g.name[len(base) + 5:-len(".npz")]
            if n.isdigit() and int(n) >= nstates:
                with np.load(g) as cached:
                    if cached["V"].shape[0] == dim:
                        return cached["E"], cached["V"], np.full(len(cached["E"]), lz), f"read from {g}"

    E, V = tms_H_ED.diagonalize(symm=dense_symm, k=min(args["ed_k"], dim - 2))
    q = np.real(np.einsum("si,si->i", V.conj(), L2_sp @ V))
    L = np.round((-1 + np.sqrt(1 + 4*np.clip(q, 0, None))) / 2).astype(int)
    order = np.argsort(E)
    E, V, L, q = E[order], V[:, order], L[order], q[order]
    print("Lowest levels of the block, (L) from <L^2>: " + " ".join(f"{e:.6f}({l})" for e, l in zip(E, L)), flush=True)
    if np.any(np.abs(q - L*(L + 1)) > 0.05):         # float32 eigenvectors leave ~1e-2 (N=14); a mix is O(1)
        print(f"  warning: <L^2> = {np.round(q, 4)} not all L(L+1): degenerate levels of different L mix", flush=True)
    # written under a temporary name and renamed, so a run starting meanwhile never reads half a file
    part = ed_dir / f"{base}_nev_{args['ed_k']}.part{os.getpid()}.npz"
    np.savez(part, E=E, V=V, L=L)
    os.replace(part, f)
    return E, V, L, f"saved to {f}"


def ed_eigenvalue(V, U):
    """The eigenvalue of each ED vector (columns of V, dense basis) under U, from <s|U|v> = u v(s) on its largest
    entries (the levels are non-degenerate within the block)."""
    states = qtx.utils.ints_to_array(dense_symm.basis.states)
    out = []
    for v in V.T:
        ds = qtx.state.DenseState(np.asarray(v), dense_symm)
        s = states[np.argsort(-np.abs(v))[:8]]
        t, sign = jax.vmap(U.images)(jnp.asarray(s))
        out.append(float(np.median(np.asarray(sign[:, 1]) * ds(np.asarray(t[:, 1])) / ds(s))))
    return np.array(out)


def dense_members():
    """The members on the dense basis (columns), each divided by its largest amplitude, and the log of that amplitude.
    Taken in log space: todense overflows float32 beyond e^88, and nothing read from them depends on a member's
    scale."""
    basis_ints = dense_symm.basis.states
    symm_norm = np.real(dense_symm.basis.get_amp(basis_ints))
    P, top = [], []
    for state in state_set.states:
        psi = state[basis_ints]
        logabs, sign = np.asarray(psi.logabs, float), np.asarray(psi.sign, float)
        top.append(np.max(logabs[np.isfinite(logabs)]))
        P.append(sign * np.exp(logabs - top[-1]) / symm_norm)
    P = np.stack(P, 1)
    bad = ~np.isfinite(P)
    if bad.any():
        s = qtx.utils.ints_to_array(basis_ints[np.flatnonzero(bad.any(1))[:3]])
        raise SystemExit(f"dense readout: non-finite amplitudes of members {np.flatnonzero(bad.any(0)).tolist()} on "
                         f"{bad.any(1).sum()} of {len(P)} configurations; electrons per orbital of the first: "
                         f"{((s[:, :N] > 0).astype(int) + (s[:, N:] > 0).astype(int)).tolist()}")
    return P, np.array(top)


def dense_leak(P=None):
    """The leak tr(S^-1 L^2) - sum_i L_i(L_i+1) out of the members' L's (0 on the exact span), and S, L^2 reduced to
    the members."""
    P = dense_members()[0] if P is None else P
    S = (P.conj().T @ P + (P.conj().T @ P).conj().T) / 2
    L2_red = P.conj().T @ (L2_sp @ P)
    return np.real(np.trace(np.linalg.solve(S, L2_red))) - sum(L * (L + 1) for L in Ls), S, L2_red


def dense_readout():
    """tr(S^-1 H), subspace fidelity with the ED targets, Ritz values with the L of their vectors (from <L^2>), the
    leak (dense_leak), each member's L^2 outside the targets and log10 of its largest amplitude."""
    P, top = dense_members()
    leak, S, L2_red = dense_leak(P)
    H_red = P.conj().T @ (H_sp @ P)
    ritz, C = sp.linalg.eigh((H_red + H_red.conj().T) / 2, S)
    q = np.real(np.einsum("ir,ij,jr->r", C.conj(), L2_red, C))
    fidelity = np.linalg.norm(V_ED.conj().T @ sp.linalg.orth(P))**2 / nstates
    # each member's L^2 outside the ED targets (exact L^2 eigenstates), per unit norm: blind to the members mixing among
    # the targets, which the trace allows and which moves <L^2> between members of different L
    A = V_ED.conj().T @ P
    member_leak = np.real(np.diag(L2_red) - np.einsum("ta,t->a", np.abs(A)**2, L_ED * (L_ED + 1))) / np.real(np.diag(S))
    return (float(np.sum(ritz)), float(fidelity), ritz, (-1 + np.sqrt(1 + 4*np.clip(q, 0, None))) / 2, float(leak),
            member_leak, top / np.log(10))


# ED in the Lz user basis, Z2-resolved with --z2-sect (~7e5 states at N=14, Lz=0). The targets: for every L among the
# members the lowest levels of that L, as many as there are members of it (with --ry-ph-sect those of that eigenvalue)
if do_ED:
    tms_H_ED = tms_H if int_dropped == 0 else hamiltonian()[0]
    dense_symm = LzUserBasisSymmetry(lz, sectors["z2"] or 0)
    L2_sp = GetL2Terms(N, 2).get_quspin_op(dense_symm).tocsr()
    H_sp = tms_H_ED.get_quspin_op(dense_symm).tocsr()
    dim = H_sp.shape[0]
    ed_dir = Path(path if args["ed_dir"] is None else args["ed_dir"])
    ed_dir.mkdir(parents=True, exist_ok=True)
    E_all, V_all, L_all, source = exact_levels()
    keep = []
    for L in sorted(set(Ls)):
        found = np.flatnonzero(L_all == L)
        if sectors["phry"] is not None:
            ph = ed_eigenvalue(V_all[:, found], PHRyQH(1))
            print(f"PH x Ry eigenvalue of the exact L={L} levels {np.round(E_all[found], 6)}: {np.round(ph, 4)}",
                  flush=True)
            found = found[ph * sectors["phry"] > 0.5]
        if len(found) < Ls.count(L):
            raise SystemExit(f"only {len(found)} exact L={L} levels in the sector among the lowest {len(E_all)}, "
                             f"{Ls.count(L)} members: raise --ed-k")
        keep += list(found[:Ls.count(L)])
    keep = sorted(keep, key=lambda j: E_all[j])
    E_ED, V_ED, L_ED = E_all[keep], V_all[:, keep], L_all[keep]
    print("ED targets: " + " ".join(f"{e:.6f}(L{l})" for e, l in zip(E_ED, L_ED)) + f", sum {np.sum(E_ED):.6f}; "
          f"block dim {dim}, {source}", flush=True)


meta = dict(args=args, ansatz_meta=str(args["ansatz_meta"]), N=N, lz=lz, z2=z2, h=h, K=nstates, lam=lam,
            theta=ansatz["theta"], continued_from=cont_run_id, int_dropped=int_dropped, symmetries=sectors,
            member_eigenvalues=member_eigvals,
            members=[dict(k=m["k"], L=L, root=m["root"], file=f"state{i}_{run_id}.eqx",
                          spec=member_spec(model, m["spec"]["d"], m["spec"]["nb"], m["spec"]["nh"]))
                     for i, (m, L, model) in enumerate(zip(ansatz["members"], Ls, models))],
            nparams_per_state=[int(n) for n in state_set.nparams_per_state])
if do_ED:
    meta["E_ED"], meta["L_ED"] = E_ED.tolist(), L_ED.tolist()


def save_meta():
    with open(f"{path}/meta_{run_id}.json", "w") as f:
        json.dump(meta, f, indent=1, default=float)


save_meta()
exact_rows = []
exact_dL = (np.nan, np.nan)      # --max-dL with --exact-diag: dL1, dL2 of the dense members at the readout sweeps


def record_exact(sweep):
    """Dense readout at this sweep, appended to data_energy_exact; returns the printed summary."""
    trE, fidelity, ritz, ritz_L, leak, member_leak, top = dense_readout()
    exact_rows.append((sweep, trE, fidelity, *ritz, leak, *ritz_L, *member_leak, *exact_dL))
    np.savetxt(f"{path}/data_energy_exact_{run_id}.txt", np.array(exact_rows),
               header="sweep trE fidelity " + " ".join(f"ritz{k}" for k in range(nstates)) + " trL2 "
                      + " ".join(f"ritzL{k}" for k in range(nstates)) + " "
                      + " ".join(f"outL2_{k}" for k in range(nstates)) + " dL1 dL2")
    # the members start at norm 1, so a larger amplitude is a blow-up on configurations their fits never saw
    return (f"dense trE = {trE:.6f}, fidelity = {fidelity:.6f}, Ritz = {np.round(ritz, 4)} "
            f"(L {np.round(ritz_L, 2)}), L^2 leak = {leak:.2e}"
            + (f", log10 of each member's largest amplitude {np.round(top, 1)}" if sweep < 0 else ""))


if do_ED:
    print(f"  start (sweep -1): {record_exact(-1)}; ED trE = {np.sum(E_ED[:nstates]):.6f}", flush=True)


class LoggedAdam(qtx.optimizer.Adam):
    """Adam that also returns the norm of the raw SR solve g in its buffers."""

    def init(self, nparams):
        return {**super().init(nparams), "g_norm": jnp.zeros(())}

    def update(self, core_solve, Obar, Ebar, buffers):
        solves = []

        def solve(*a, **k):
            solves.append(core_solve(*a, **k))
            return solves[-1]

        step, buffers = super().update(solve, Obar, Ebar, buffers)
        buffers["g_norm"] = jnp.linalg.norm(solves[0])
        return step, buffers


class ObarKeepingSR(NaturalExcitedSR):
    def get_Obar(self, samples):
        self.last_Obar = super().get_Obar(samples)
        return self.last_Obar


solver = None if args["rshift"] is None else qtx.optimizer.auto_shift_eig(rshift=args["rshift"])
optimizer = ObarKeepingSR(state_set, tms_H, updater=LoggedAdam(), solver=solver, penalty=tms_LmLp, penalty_coeff=lam,
                          penalty_every=lmlp_every, diagnostics=args["diagnostics"])

# --max-dL: the step's change of tr(S^-1 L^- L^+) on the previous batch by correlated sampling (the same tuples at the
# parameters before and after it). From a clean span the mean change is quadratic in the step, but every sample's is
# linear: those linear parts average out and leave noise ~ |step|/sqrt(n) over a signal ~ |step|^2 (N=8: correlation
# 0.3 with the exact change). The even part [dL(+step) + dL(-step)]/2 cancels them sample by sample and leaves the
# quadratic part, the L^- L^+ the step creates; the odd part, the linear term (the penalty's restoring force, or H's
# pull on junk already there), is logged with its noise. Its own operator object, so that --profile's eloc_LmLp
# times the training's alone
leak_trace = NaturalTraceEnergyGrad(tms_Lp.H @ tms_Lp)


def batch_LmLp(samples):
    """tr(S^-1 L^- L^+) of the members as they stand, on `samples`: weights |det A|^2 over the tuples' sampling
    weights, which makes it exact for tuples drawn at other parameters (the previous sweep's)."""
    t = jnp.real(leak_trace.local_energies(state_set, samples, resolve=False)[0])
    logw = jnp.log(samples.reweight_factor) + 2 * (sampler.determinant(samples.spins).logabs
                                                   - LogArray.from_value(samples.psi).logabs)
    ok = jnp.isfinite(t) & jnp.isfinite(logw)
    w = jnp.where(ok, jnp.exp(logw - jnp.max(jnp.where(ok, logw, -jnp.inf))), 0.0)
    return float(jnp.sum(w * jnp.where(ok, t, 0.0)) / jnp.sum(w))


def after_step(dtheta, fn, *args):
    """fn(*args) after the update dtheta, the members restored afterwards: their models are put back rather than
    stepped back, so no rounding remains (the jitted forwards take the model as an argument, nothing recompiles)."""
    models = [state._model for state in state_set.states]
    state_set.update(state_set.split_step(dtheta))
    try:
        return fn(*args)
    finally:
        for state, model in zip(state_set.states, models):
            state._model = model


DIAGNOSTICS_COLUMNS = ("sigma_max", "sigma_min", "rank_eff", "obar_fro2")
rows, member_rows = [], []

# --profile: wall time of each part of a sweep. Every timed call is waited on (JAX dispatches asynchronously), so the
# parts add up to the sweep; the first sweeps also contain compilation
PROFILE_COLUMNS = ("sample", "eloc_H", "eloc_LmLp", "ebar_rest", "obar", "solve", "update", "dL", "readout", "output")
timings, profile_rows = {}, []


def timed(key, fn):
    def run(*a, **k):
        t = perf_counter()
        out = jax.block_until_ready(fn(*a, **k))
        timings[key] = timings.get(key, 0.0) + perf_counter() - t
        return out
    return run


def profile_summary(first=2):
    p = np.array(profile_rows)[first:]
    if len(p) == 0:
        return "no sweeps after compilation yet"
    mean = p.mean(0)
    return (f"{mean[1]:.2f} s per sweep (sweeps {first}-{int(p[-1, 0])}): "
            + ", ".join(f"{c} {m / mean[1]:.0%}" for c, m in zip(PROFILE_COLUMNS + ("rest",), mean[2:])))


if args["profile"]:
    sampler.sweep = timed("sample", sampler.sweep)
    tms_H.Oloc = timed("eloc_H", tms_H.Oloc)
    tms_LmLp.Oloc = timed("eloc_LmLp", tms_LmLp.Oloc)
    optimizer.get_Ebar = timed("ebar", optimizer.get_Ebar)
    optimizer.get_Obar = timed("obar", optimizer.get_Obar)
    optimizer.solve = timed("solve", optimizer.solve)
# The size of a step is measured on an independent batch, the previous sweep's (an extra one before sweep 0 with a
# trust region): on the batch the step was solved from, |Obar dtheta|^2 is biased low. --warmup-fs holds the first
# sweeps of a fresh run, whose steps are mostly noise, to a smaller 1 - F. --max-dL measures on the same batch
warmup = 0 if cont_run_id is not None or args["warmup_fs"] <= 0 else \
    (2 * N if args["warmup_sweeps"] is None else args["warmup_sweeps"])
prev_samples = sampler.sweep() if max_fs > 0 or warmup > 0 or max_dL > 0 else None
prev_Obar = None if prev_samples is None else optimizer.get_Obar(prev_samples)
start_time = datetime.now()
for i in range(args["nbr_sweeps"]):
    timings.clear()
    t_sweep = perf_counter()
    samples = sampler.sweep()
    step = optimizer.get_step(samples)

    # checked before `update`, so the saved checkpoints remain the last good state
    if not (jnp.all(jnp.isfinite(step)) and np.isfinite(optimizer.energy)):
        raise RuntimeError(f"Non-finite update at sweep {i}: energy={optimizer.energy}, "
                           f"{optimizer.nonfinite} of {nsamples} samples dropped for Eloc, "
                           f"{optimizer.nonfinite_grad} for the log-derivative")

    # cap the step norm, then (--max-fs) the step's 1 - F: `fs` on the previous sweep's batch, `fs_same` on this one
    t = perf_counter()
    step_norm = float(jnp.linalg.norm(step))
    if 0 < args["norm_clip"] < step_norm:
        step = step * (args["norm_clip"] / step_norm)
    dtheta = step * adaptive_learning_rate_inv(args["lr"], t0, i)
    fs_same = float(jnp.sum(jnp.abs(optimizer.last_Obar @ dtheta)**2))
    fs = fs_applied = np.nan if prev_Obar is None else float(jnp.sum(jnp.abs(prev_Obar @ dtheta)**2))
    cap = args["warmup_fs"] if i < warmup else max_fs
    if 0 < cap < fs:
        dtheta, fs_applied = dtheta * np.sqrt(cap / fs), cap
    # then (--max-dL) the L^- L^+ the step creates, dL2 (scaling the step by c scales it by c^2), on the previous
    # batch too; dL1 is the linear part of the change
    dL1 = dL2 = dL_applied = np.nan
    exact_dL = (np.nan, np.nan)
    if max_dL > 0:
        t_dL = perf_counter()
        probe = prev_samples if args["max_dL_samples"] <= 0 else prev_samples[:args["max_dL_samples"]]
        L0 = batch_LmLp(probe)
        up, down = after_step(dtheta, batch_LmLp, probe) - L0, after_step(-dtheta, batch_LmLp, probe) - L0
        dL1, dL2 = (up - down) / 2, (up + down) / 2
        if do_ED and (i % args["ed_every"] == 0 or i == args["nbr_sweeps"] - 1):
            # the same two parts from the dense members, for the readout: what sampling misses
            L0 = dense_leak()[0]
            up, down = (after_step(d, lambda: dense_leak()[0]) - L0 for d in (dtheta, -dtheta))
            exact_dL = ((up - down) / 2, (up + down) / 2)
        dL_applied = dL2
        if dL2 > max_dL:
            c = np.sqrt(max_dL / dL2)
            dtheta, fs_applied, dL_applied = dtheta * c, fs_applied * c**2, max_dL
        timings["dL"] = perf_counter() - t_dL
    if args["diagnostics"] and prev_Obar is not None:
        # each member's part of the applied step: |Obar_a dtheta_a|^2 on the previous batch and on its own (a ratio far
        # above 1 is noise the step's batch fitted); its norm; cross terms between members are left out
        edges = np.cumsum([0, *state_set.nparams_per_state])
        blocks = list(zip(edges[:-1], edges[1:]))
        member_rows.append((i, *(float(jnp.sum(jnp.abs(O[:, a:b] @ dtheta[a:b])**2))
                                 for O in (prev_Obar, optimizer.last_Obar) for a, b in blocks),
                            *(float(jnp.linalg.norm(dtheta[a:b])) for a, b in blocks)))
        np.savetxt(f"{path}/data_members_{run_id}.txt", np.array(member_rows),
                   header="sweep " + " ".join(f"{q}{k}" for q in ("fs", "fs_same", "norm") for k in range(nstates)))
    state_set.update(state_set.split_step(dtheta))
    prev_Obar, prev_samples = optimizer.last_Obar, samples
    if args["profile"]:
        jax.block_until_ready([st.model for st in state_set.states])
        timings["update"] = perf_counter() - t - timings.get("dL", 0.0)

    t = perf_counter()
    g_norm = optimizer._buffers["g_norm"]
    diag = optimizer.diagnostics if args["diagnostics"] else {}
    acceptance = sampler.acceptance if args["diagnostics"] else {}
    rows.append((i, optimizer.energy, optimizer.VarE, optimizer.energy_H, optimizer.VarE_H, optimizer.penalty_value,
                 optimizer.VarPenalty, step_norm, g_norm, *(diag.get(key) for key in DIAGNOSTICS_COLUMNS),
                 fs, fs_applied, fs_same, optimizer.penalty_coeff, acceptance.get("moved"), acceptance.get("valid"),
                 dL1, dL2, dL_applied))
    np.savetxt(f"{path}/data_energy_{run_id}.txt",
               np.array([[np.nan if v is None else float(np.real(v)) for v in row] for row in rows]),
               header="sweep trE VarE trE_H VarE_H trLmLp VarLmLp step_norm g_norm "
                      + " ".join(DIAGNOSTICS_COLUMNS) + " fs fs_applied fs_same lambda moved valid dL1 dL2 dL_applied")
    for index, state in enumerate(state_set.states):
        state.save(f"{path}/state{index}_{run_id}.eqx")
    timings["output"] = perf_counter() - t

    line = (f"  sweep {i:5d}  trE = {float(optimizer.energy):.6f}  VarE = {float(optimizer.VarE):.3e}  "
            f"|step| = {step_norm:.3e}  |g| = {float(g_norm):.3e}  fs = {fs:.2e}"
            + (f"  dL2 = {dL2:.2e}" if max_dL > 0 else "")
            + (f"  acceptance = {acceptance['moved'] / max(acceptance['valid'], 1e-12):.3f}" if acceptance else ""))
    if do_ED and (i % args["ed_every"] == 0 or i == args["nbr_sweeps"] - 1):
        t = perf_counter()
        line += "  | " + record_exact(i)
        timings["readout"] = perf_counter() - t
    if i < 5 or i % 10 == 0 or i == args["nbr_sweeps"] - 1:
        print(line, flush=True)

    if args["profile"]:
        total = perf_counter() - t_sweep
        timings["ebar_rest"] = timings.get("ebar", 0.0) - timings.get("eloc_H", 0.0) - timings.get("eloc_LmLp", 0.0)
        parts = [timings.get(c, 0.0) for c in PROFILE_COLUMNS]
        profile_rows.append([i, total, *parts, total - sum(parts)])
        np.savetxt(f"{path}/data_profile_{run_id}.txt", np.array(profile_rows),
                   header="seconds: sweep total " + " ".join(PROFILE_COLUMNS) + " rest")
        if i in (10, 50) or i == args["nbr_sweeps"] - 1:
            print("  profile: " + profile_summary(), flush=True)

print("NES training completed in: ", datetime.now() - start_time)
meta["time"] = str(datetime.now() - start_time)
save_meta()
