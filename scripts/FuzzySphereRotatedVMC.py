# Single-state counterpart of FuzzySphereRotatedNES.py: SR on one rotated exciton member of FuzzySphereRotatedAnsatz.py
# (--member, its index in the ansatz), at transverse field h in the ansatz's Lz sector, where NES needs K >= 2. The member
# is released as there (untied, the k=2 up-feature suppression lowered to exp(-gap), the k=0,1 determinants given a
# small random backflow) and trained by energy minimisation against H + lambda L^- L^+, sampled from |psi|^reweight with
# pair hops mixed with single-electron flavour flips. Minimisation finds the lowest level of the imposed sectors, so
# the target is the member's only if it is that level: a k=2 root above the vacuum, or a member of L > Lz without a
# symmetry that excludes the lower L, falls to the vacuum. Symmetries, the penalty, the trust region, --cont-run and the
# output layout (one member, state0_<run_id>.eqx) are those of FuzzySphereRotatedNES.py; --exact-diag tracks the member
# against the lowest level of its L in the same sectors.
#
# As in NES, the first SR steps are 1e2-1e4 in norm and wreck the member (fidelity 0.66 -> 0.08 at N=8) whatever the
# trust region says, which cannot see the network directions of a W ~ 0 determinant; a tight --norm-clip holds them.
# N=8, k=0, 400 sweeps of the example: E -13.40230 vs ED -13.40266, fidelity 0.99999.
#
#   python FuzzySphereRotatedVMC.py --ansatz-meta runs/rot/meta_n_8_2s_7_lz_0_z2_1_ryph_0_h_3.16_id01.json \
#       --norm-clip 5 --lr 5e-2 --lr-t0 160 --nbr-samples 2048 --nbr-sweeps 400 --exact-diag --path runs/rotvmc
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import quantax as qtx
from quantax.global_defs import get_subkeys
from quantax.nn import lecun_normal
from quantax.utils import LogArray

from qhuantax.quantumhall_excitons import KSectors, rotated_vacuum_configs
from qhuantax.quantumhall_operators import GetL2Terms, GetLpTerms, GetSpinfulDenIntTerms, GetSpinfulPolTerms
from qhuantax.quantumhall_optimizers import PenalizedAdamSR, PenalizedEnergyGrad
from qhuantax.quantumhall_samplers import FermionDipoleConsFlavourFlip
from qhuantax.quantumhall_symmetries import PHRyQH, RyQH, fuzzy_sphere_symmetry
from qhuantax.quantumhall_userbasis import LzUserBasisSymmetry
from qhuantax.quantumhall_utils import adaptive_learning_rate_inv
from qhuantax.rotated import load_member, member_spec, set_gap, untie


S1 = np.array([[1, 0], [0, 0]])
S2 = np.array([[0, 0], [0, 1]])
SX = np.array([[0, 1], [1, 0]])


parser = argparse.ArgumentParser()
parser.add_argument("--ansatz-meta", required=True, help="meta file written by FuzzySphereRotatedAnsatz.py")
parser.add_argument("--member", type=int, default=None,
                    help="index of the ansatz member to train; may be omitted when the ansatz has one")
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
                    help="k=0,1 member: W = this x the default random init; 0 keeps W = 0")
parser.add_argument("--gap", type=float, default=8.0,
                    help="k=2 member, and a k=1 one with a live backflow: up features suppressed by exp(-gap); the "
                         "live k=0 one: down features; <= 0 keeps the ansatz's")
parser.add_argument("--fs-exact-sweeps", type=int, default=10,
                    help="in this run's first this many sweeps (continuations too: Adam restarts) --max-fs and "
                         "--warmup-fs act on the step's actual 1 - F, estimated from the amplitude ratios psi_new/psi "
                         "on the previous sweep's batch, not on the linear |Obar dtheta|^2: the step is shrunk until "
                         "that is below the cap (one forward pass per try). The linear estimate misses the steps that "
                         "move small parameters by many times their size, as AdamSR's first step does (N=8 vacuum: fs "
                         "1e-6 for an exact 1 - F of 0.99). 0 disables, -1 throughout")
parser.add_argument("--open-fill", type=float, default=0.0,
                    help="members with a live backflow: fill the reference's structural zeros (which keep it in its "
                         "exciton sector) with this x std(U0) x N(0,1), opening the neighbouring sectors linearly; "
                         "use with --gap 20, which keeps the suppressed flavour shut; 0 disables")
parser.add_argument("--lmlp-coeff", default="0",
                    help="coefficient lambda of the L^- L^+ penalty, or 'auto': the smallest lambda that keeps the "
                         "lowest level of every L > Lz --lmlp-margin above the lowest L = Lz one, and at least the "
                         "--lmlp-floor-gap floor; from the ED levels at h = 3.16 of misc/iqh_gaps.py (N = 7-14, other "
                         "N extrapolated). The member must have L = Lz")
parser.add_argument("--lmlp-margin", type=float, default=1.0,
                    help="auto: energy by which the L > Lz levels must clear the target")
parser.add_argument("--lmlp-floor-gap", type=float, default=20.0,
                    help="auto: floor on lambda, kept when the levels need less, as the energy by which it lifts a "
                         "random state of the Lz sector (0.083 at N=14, ~N^3); 0 disables")
parser.add_argument("--lmlp-every", type=int, default=None,
                    help="measure <L^- L^+> every this many sweeps, also at lambda = 0 (the gradient is unchanged); "
                         "0 never. Defaults to 1 with a penalty, 0 without")
parser.add_argument("--exact-diag", action="store_true", help="track the member against ED (feasible up to N=14)")
parser.add_argument("--ed-every", type=int, default=5, help="ED readout every this many sweeps")
parser.add_argument("--ed-k", type=int, default=10, help="lowest levels of the Lz block to compute, all L")
parser.add_argument("--ed-dir", default=None, help="ED cache folder, searched first; defaults to --path")
parser.add_argument("--nbr-sweeps", type=int, default=200, help="number of training iterations")
parser.add_argument("--nbr-samples", type=int, default=512, help="number of samples per iteration")
parser.add_argument("--lr", type=float, default=1e-2, help="starting value of the learning rate")
parser.add_argument("--lr-t0", type=float, default=None, help="lr / (1 + sweep / lr_t0), 0 for a constant lr; defaults to N")
parser.add_argument("--norm-clip", type=float, default=0.0,
                    help="cap on the step norm before the learning rate; <= 0 disables")
parser.add_argument("--rshift", type=float, default=None,
                    help="relative diagonal shift of the SR solve; defaults to quantax's")
parser.add_argument("--max-fs", type=float, default=0.0,
                    help="trust region: cap on 1 - F per sweep, F the fidelity of the state before and after the "
                         "update, to second order dtheta^T S dtheta with S from the previous sweep; 0 disables")
parser.add_argument("--warmup-fs", type=float, default=1e-4,
                    help="the trust region (in place of --max-fs) for the first --warmup-sweeps sweeps of a fresh run; "
                         "not applied with --cont-run; 0 disables")
parser.add_argument("--warmup-sweeps", type=int, default=None, help="sweeps under --warmup-fs; defaults to 2N")
parser.add_argument("--thermal-steps", type=int, default=None,
                    help="sampler thermalisation steps; defaults to the sampler's")
parser.add_argument("--sweep-steps", type=int, default=None,
                    help="Metropolis steps between samples; defaults to the sampler's")
parser.add_argument("--reweight", type=float, default=1.0,
                    help="sample from |psi|^reweight. 2 traps the k=0 member (N=8): the charge-fluctuating configurations "
                         "keep ~1e-4 of the weight but ~98%% of the exact variance, |psi|^2 never samples them, and "
                         "training stalls at fidelity 0.991 with a sampled VarE of 1e-6")
parser.add_argument("--flip-prob", type=float, default=0.5, help="fraction of flavour-flip proposals")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--run-id", type=int, default=1)
parser.add_argument("--cont-run", type=int, default=None,
                    help="run id under --path to continue from: its member as it is (no release); the sweep counter, "
                         "the lr schedule and Adam restart, so --lr is the rate this run opens with")
parser.add_argument("--path", required=True, help="output directory")
args = vars(parser.parse_args())

ansatz = json.load(open(args["ansatz_meta"]))
N, lz, z2 = ansatz["N"], ansatz["lz"], ansatz["z2"]
if args["member"] is None:
    if len(ansatz["members"]) != 1:
        parser.error(f"the ansatz has {len(ansatz['members'])} members: choose one with --member")
    args["member"] = 0
if not 0 <= args["member"] < len(ansatz["members"]):
    parser.error(f"--member {args['member']}: the ansatz has {len(ansatz['members'])} members")
member = ansatz["members"][args["member"]]
k, L = member["k"], member.get("L", lz)          # older ansatz metas: every member has L = Lz
h = ansatz["h"] if args["h"] is None else args["h"]
pspot_inter = np.array([float(v) for v in ansatz["args"]["ps_pot"].split(",")])
nsamples, max_fs, do_ED, path = (args[key] for key in ("nbr_samples", "max_fs", "exact_diag", "path"))
t0 = N if args["lr_t0"] is None else args["lr_t0"]


sectors = dict(z2=args["z2_sect"], ry=args["ry_sect"], phry=args["ry_ph_sect"])


def run_name(run):
    return (f"rotvmc_n_{N}_2s_{N-1}_lz_{lz}_z2_{z2}_ryph_{sectors['phry'] or 0}_k_{k}_L_{L}_root_{member['root']}"
            f"_h_{h:g}_id0{run}")


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
if k > 0 or member["root"] > 0 or L != lz:
    print(f"warning: the member (k={k}, L={L}, root {member['root']}) need not be the lowest level of the imposed "
          "sectors, which energy minimisation finds instead", flush=True)

lmlp_auto = args["lmlp_coeff"].strip().lower() == "auto"
if lmlp_auto:
    if L != lz:
        parser.error(f"--lmlp-coeff auto keeps the L > Lz levels out of the window, but the member has L = {L}")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "misc"))
    from iqh_gaps import H_FIELD, PS_POT, min_lambda, random_state_LL
    if not np.isclose(h, H_FIELD) or tuple(pspot_inter.tolist()) != PS_POT:
        parser.error(f"--lmlp-coeff auto reads the ED levels at h = {H_FIELD}, V = {PS_POT}; this run has h = {h}, "
                     f"V = {ansatz['args']['ps_pot']}")
    try:
        lam_need = min_lambda(N, lz, 1, Z=sectors["z2"], ry_ph=sectors["phry"], ry=sectors["ry"],
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
if lam > 0 and L != lz:
    print(f"warning: --lmlp-coeff also lifts the member's L = {L} > Lz; the target becomes the lowest level of "
          "H + lambda L^- L^+", flush=True)

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


def member_eigenvalue(U, project=fit_ry_ph is not None):
    """<U> of the ansatz member at h=0, an exact sum over its sector k, which Ry and PH x Ry map onto itself: the
    eigenvalue for an exact member; the projection onto the other sector would keep (1 - <U>)/2 of its weight. With
    `project`, of the PH x Ry projected model the ansatz fitted, psi_P(x) = [psi(x) + r sign psi(x')]/2."""
    model = load_member(member["spec"], N, Path(args["ansatz_meta"]).parent / member["file"]).rebuild(theta=0.0)
    s = jnp.asarray(ksectors.configs(k))
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


# The member must lie in the imposed Ry and PH x Ry sectors, or the projection removes it (R(theta) commutes with both)
member_eigvals = {}
if cont_run_id is None and fit_ry_ph is not None:
    kept = member_eigvals["phry_kept"] = (1 + fit_ry_ph * member_eigenvalue(PHRyQH(1), project=False)) / 2
    print(f"member fitted as a PH x Ry {fit_ry_ph:+d} projected model: the projection keeps {kept:.3f} of the "
          "network's weight", flush=True)
for name, U in (("ry", RyQH(1)), ("phry", PHRyQH(1))):
    if cont_run_id is None and sectors[name] is not None and not (name == "phry" and fit_ry_ph is not None):
        ev = member_eigvals[name] = member_eigenvalue(U)
        print(f"{OPTION[name]}: eigenvalue of the ansatz member at h=0 {ev:.4f}", flush=True)
        if ev * sectors[name] < 0.5:
            raise SystemExit(f"{OPTION[name]} {sectors[name]:+d}: the member lies mostly in the other sector "
                             "and the projection would remove it")


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
    live = member.get("backflow") == "live"            # k = 0, 1 with a live backflow (older ansatz runs: W = 0)
    if (member["k"] >= 2 or live) and args["gap"] > 0:     # the fits; the live vacuum suppresses its down features
        model = set_gap(model, args["gap"], flavour="down" if member["k"] == 0 else "up")
    if live and args["open_fill"] > 0:
        # the neighbouring sectors opened through the reference's structural zeros (the down block of the vacuum, the
        # off-diagonal blocks of a block member): their amplitudes are linear in these entries, so the trust region,
        # a linear estimate, sees the steps that open them -- the gap's exponential features it does not (N=8 vacuum:
        # fs 2e-6 for an exact 1 - F of 0.09 at gap 8)
        zero = model.U0 == 0
        noise = args["open_fill"] * float(jnp.std(model.U0)) * jax.random.normal(get_subkeys(), model.U0.shape,
                                                                                 model.U0.dtype)
        model = eqx.tree_at(lambda m: m.U0, model, jnp.where(zero, noise, model.U0))
    if member["k"] < 2 and args["det_backflow_scale"] > 0 and not jnp.any(model.W):    # the exact determinant(s)
        # relative to U0, whose std is 1 unless the ansatz script scaled the member to unit norm
        W = lecun_normal(get_subkeys(), model.W.shape, dtype=model.W.dtype) / 10 * float(jnp.std(model.U0))
        model = eqx.tree_at(lambda m: m.W, model, args["det_backflow_scale"] * W)
    return model


if cont_run_id is None:
    model = release(member)
    start = (f"theta {np.degrees(ansatz['theta']):.4f} deg, {'charge-only' if args['no_untie'] else 'untied'}, "
             f"k=2 gap {args['gap'] if args['gap'] > 0 else 'unchanged'}")
else:
    model = load_member(cont_meta["members"][0]["spec"], N, f"{path}/{cont_meta['members'][0]['file']}")
    start = f"continued from {cont_run_id} (lambda {cont_meta['lam']:g} at its end)"
    if lam < cont_meta["lam"]:
        print(f"warning: --lmlp-coeff {lam:g} is below the lambda {cont_meta['lam']:g} that {cont_run_id} ended with",
              flush=True)
state = qtx.state.Variational(model, symm=symm, max_parallel=16384, use_ref=False)
print(f"N={N}, Lz={lz}, h={h}, symmetries imposed: {symm_label}; member {args['member']} (k={k}, L={L}, root "
      f"{member['root']}), {start}; {state.nparams} parameters", flush=True)

# Chain seeds distributed like the rotated vacuum (and its Z2 image when Z2 is imposed)
init_configs = rotated_vacuum_configs(N, nsamples, ansatz["theta"], lz, both=sectors["z2"] is not None,
                                      rng=np.random.default_rng(args["seed"]))
sampler = FermionDipoleConsFlavourFlip(state, nsamples, n_neighbor=np.arange(1, 3), initial_spins=init_configs,
                                       reweight=args["reweight"], flip_prob=args["flip_prob"],
                                       thermal_steps=args["thermal_steps"], sweep_steps=args["sweep_steps"])


def exact_levels():
    """The --ed-k lowest levels of the block (all L, L from <L^2>), sorted: read from ed_dir or computed and cached
    there, in the cache FuzzySphereRotatedNES.py shares. An older cache of L = lz levels only (..._nst_<n>.npz) serves
    when the member has L = lz."""
    base = f"ed_n_{N}_2s_{N-1}_lz_{lz}_z2_{sectors['z2'] or 0}_h_{h:g}_V_{ansatz['args']['ps_pot'].replace(',', '_')}"
    f = ed_dir / f"{base}_nev_{args['ed_k']}.npz"
    if f.is_file():
        with np.load(f) as cached:
            return cached["E"], cached["V"], cached["L"], f"read from {f}"
    if L == lz:
        for g in sorted(ed_dir.glob(f"{base}_nst_*.npz")):
            with np.load(g) as cached:
                if cached["V"].shape[0] == dim:
                    return cached["E"], cached["V"], np.full(len(cached["E"]), lz), f"read from {g}"

    E, V = tms_H_ED.diagonalize(symm=dense_symm, k=min(args["ed_k"], dim - 2))
    q = np.real(np.einsum("si,si->i", V.conj(), L2_sp @ V))
    Ls = np.round((-1 + np.sqrt(1 + 4*np.clip(q, 0, None))) / 2).astype(int)
    order = np.argsort(E)
    E, V, Ls, q = E[order], V[:, order], Ls[order], q[order]
    print("Lowest levels of the block, (L) from <L^2>: " + " ".join(f"{e:.6f}({l})" for e, l in zip(E, Ls)), flush=True)
    if np.any(np.abs(q - Ls*(Ls + 1)) > 0.05):       # float32 eigenvectors leave ~1e-2 (N=14); a mix is O(1)
        print(f"  warning: <L^2> = {np.round(q, 4)} not all L(L+1): degenerate levels of different L mix", flush=True)
    # written under a temporary name and renamed, so a run starting meanwhile never reads half a file
    part = ed_dir / f"{base}_nev_{args['ed_k']}.part{os.getpid()}.npz"
    np.savez(part, E=E, V=V, L=Ls)
    os.replace(part, f)
    return E, V, Ls, f"saved to {f}"


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


def dense_readout():
    """<H>, fidelity with the ED target, the L of <L^2>, the leak <L^2> - L(L+1) out of the member's L (0 on an exact
    eigenstate of it) and log10 of the largest amplitude. The state enters divided by that amplitude, taken in log
    space: todense overflows float32 beyond e^88, and nothing here depends on its scale."""
    basis_ints = dense_symm.basis.states
    symm_norm = np.real(dense_symm.basis.get_amp(basis_ints))
    psi = state[basis_ints]
    logabs, sign = np.asarray(psi.logabs, float), np.asarray(psi.sign, float)
    top = np.max(logabs[np.isfinite(logabs)])
    p = sign * np.exp(logabs - top) / symm_norm
    bad = ~np.isfinite(p)
    if bad.any():
        s = qtx.utils.ints_to_array(basis_ints[np.flatnonzero(bad)[:3]])
        raise SystemExit(f"dense readout: non-finite amplitudes on {bad.sum()} of {len(p)} configurations; electrons "
                         f"per orbital of the first: {((s[:, :N] > 0).astype(int) + (s[:, N:] > 0).astype(int)).tolist()}")
    norm = np.real(np.vdot(p, p))
    energy = np.real(np.vdot(p, H_sp @ p)) / norm
    q = np.real(np.vdot(p, L2_sp @ p)) / norm
    fidelity = np.abs(np.vdot(V_ED, p))**2 / norm
    return (float(energy), float(fidelity), (-1 + np.sqrt(1 + 4*max(q, 0))) / 2, float(q - L*(L + 1)), top / np.log(10),
            p / np.sqrt(norm))


# ED in the Lz user basis, Z2-resolved with --z2-sect (~7e5 states at N=14, Lz=0). The target: the lowest level of the
# member's L (with --ry-ph-sect of that eigenvalue)
if do_ED:
    tms_H_ED = tms_H if int_dropped == 0 else hamiltonian()[0]
    dense_symm = LzUserBasisSymmetry(lz, sectors["z2"] or 0)
    L2_sp = GetL2Terms(N, 2).get_quspin_op(dense_symm).tocsr()
    H_sp = tms_H_ED.get_quspin_op(dense_symm).tocsr()
    dim = H_sp.shape[0]
    ed_dir = Path(path if args["ed_dir"] is None else args["ed_dir"])
    ed_dir.mkdir(parents=True, exist_ok=True)
    E_all, V_all, L_all, source = exact_levels()
    found = np.flatnonzero(L_all == L)
    if sectors["phry"] is not None:
        ph = ed_eigenvalue(V_all[:, found], PHRyQH(1))
        print(f"PH x Ry eigenvalue of the exact L={L} levels {np.round(E_all[found], 6)}: {np.round(ph, 4)}",
              flush=True)
        found = found[ph * sectors["phry"] > 0.5]
    if len(found) == 0:
        raise SystemExit(f"no exact L={L} level in the sector among the lowest {len(E_all)}: raise --ed-k")
    E_ED, V_ED = E_all[found[0]], V_all[:, found[0]]
    print(f"ED target: {E_ED:.6f} (L{L}), sector ground state {E_all[0]:.6f} (L{L_all[0]}); block dim {dim}, {source}",
          flush=True)


meta = dict(args=args, ansatz_meta=str(args["ansatz_meta"]), N=N, lz=lz, z2=z2, h=h, K=1, lam=lam,
            theta=ansatz["theta"], continued_from=cont_run_id, int_dropped=int_dropped, symmetries=sectors,
            member_eigenvalues=member_eigvals,
            members=[dict(k=k, L=L, root=member["root"], file=f"state0_{run_id}.eqx",
                          spec=member_spec(model, member["spec"]["d"], member["spec"]["nb"], member["spec"]["nh"]))],
            nparams=int(state.nparams))
if do_ED:
    meta["E_ED"], meta["L_ED"] = float(E_ED), L


def save_meta():
    with open(f"{path}/meta_{run_id}.json", "w") as f:
        json.dump(meta, f, indent=1, default=float)


save_meta()
exact_rows = []
last_dense = None


def record_exact(sweep):
    """Dense readout at this sweep, appended to data_energy_exact; returns the printed summary. dF is the exact 1 - F
    between this readout and the previous one: with --ed-every 1 the step's own, against the fs the trust region
    allowed it (fs_applied)."""
    global last_dense
    energy, fidelity, L_q, leak, top, p = dense_readout()
    dF = np.nan if last_dense is None else 1 - float(np.abs(np.vdot(last_dense, p))**2)
    last_dense = p
    exact_rows.append((sweep, energy, fidelity, L_q, leak, dF))
    np.savetxt(f"{path}/data_energy_exact_{run_id}.txt", np.array(exact_rows), header="sweep E fidelity L L2_leak dF")
    # the member starts at norm 1, so a larger amplitude is a blow-up on configurations its fit never saw
    return (f"dense E = {energy:.6f}, fidelity = {fidelity:.6f}, L {L_q:.2f}, L^2 leak = {leak:.2e}"
            + (f", dF = {dF:.2e}" if np.isfinite(dF) else "")
            + (f", log10 of the largest amplitude {top:.1f}" if sweep < 0 else ""))


if do_ED:
    print(f"  start (sweep -1): {record_exact(-1)}; ED E = {E_ED:.6f}", flush=True)


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


class ObarKeepingSR(PenalizedAdamSR):
    """`PenalizedAdamSR` with `LoggedAdam`, keeping the last Obar for the trust region."""

    def __init__(self, state, hamiltonian, penalty, penalty_coeff, penalty_every, solver=None):
        grad = PenalizedEnergyGrad(hamiltonian, penalty, penalty_coeff, penalty_every)
        qtx.optimizer.StochasticQNGD.__init__(self, state, grad, solver=solver, updater=LoggedAdam())

    def get_Obar(self, samples):
        self.last_Obar = super().get_Obar(samples)
        return self.last_Obar


solver = None if args["rshift"] is None else qtx.optimizer.auto_shift_eig(rshift=args["rshift"])
optimizer = ObarKeepingSR(state, tms_H, tms_LmLp, lam, lmlp_every, solver=solver)

rows = []
# The size of a step is measured on an independent batch, the previous sweep's (an extra one before sweep 0 with a
# trust region): on the batch the step was solved from, |Obar dtheta|^2 is biased low. --warmup-fs holds the first
# sweeps of a fresh run to a smaller 1 - F
warmup = 0 if cont_run_id is not None or args["warmup_fs"] <= 0 else \
    (2 * N if args["warmup_sweeps"] is None else args["warmup_sweeps"])
def keep_batch(samples):
    """A host copy of a batch for --fs-exact (the sampler reuses its buffers): configurations, log|psi| and sign at
    sampling time, reweight factors."""
    psi = LogArray.from_value(samples.psi)
    rw = np.ones(len(psi.logabs)) if samples.reweight_factor is None else np.asarray(samples.reweight_factor)
    return np.array(samples.spins), np.array(psi.logabs, float), np.real(np.array(psi.sign, float)), rw


prev_samples = sampler.sweep() if max_fs > 0 or warmup > 0 else None
prev_Obar = None if prev_samples is None else optimizer.get_Obar(prev_samples)
prev_batch = None if prev_samples is None else keep_batch(prev_samples)


def log_amplitudes(spins):
    v = LogArray.from_value(state(jnp.asarray(spins)))
    return np.array(v.logabs, float), np.real(np.array(v.sign, float))


def step_infidelity(dtheta, batch, current):
    """1 - F between the state and the state after dtheta, from r = psi_new/psi on `batch`, drawn from an earlier
    state (the previous sweep's) and reweighted to this one by w = rw |psi/psi_sampled|^2: F = |<w r>|^2 /
    (<w> <w |r|^2>). `current` is (log|psi|, sign) of the state on the batch."""
    spins, log_s, _, rw = batch
    log_c, sign_c = current
    model = state._model
    state.update(dtheta)
    try:
        log_n, sign_n = log_amplitudes(spins)
    finally:
        state._model = model
    logw = np.log(rw) + 2 * (log_c - log_s)
    logr = log_n - log_c
    ok = np.isfinite(logw) & np.isfinite(logr)
    w = np.where(ok, np.exp(logw - np.max(logw[ok])), 0.0)
    r = np.where(ok, sign_n * sign_c * np.exp(np.where(ok, logr, 0.0) - np.max(logr[ok])), 0.0)
    return float(1 - np.sum(w * r)**2 / (np.sum(w) * np.sum(w * r**2)))


start_time = datetime.now()
for i in range(args["nbr_sweeps"]):
    samples = sampler.sweep()
    step = optimizer.get_step(samples)

    # checked before `update`, so the saved checkpoint remains the last good state
    if not (jnp.all(jnp.isfinite(step)) and np.isfinite(optimizer.energy)):
        raise RuntimeError(f"Non-finite update at sweep {i}: energy={optimizer.energy}, "
                           f"{optimizer.nonfinite} of {nsamples} samples dropped for Eloc")

    # cap the step norm, then (--max-fs) the step's 1 - F: `fs` on the previous sweep's batch, `fs_same` on this one
    step_norm = float(jnp.linalg.norm(step))
    if 0 < args["norm_clip"] < step_norm:
        step = step * (args["norm_clip"] / step_norm)
    dtheta = step * adaptive_learning_rate_inv(args["lr"], t0, i)
    fs_same = float(jnp.sum(jnp.abs(optimizer.last_Obar @ dtheta)**2))
    fs = fs_applied = np.nan if prev_Obar is None else float(jnp.sum(jnp.abs(prev_Obar @ dtheta)**2))
    cap = args["warmup_fs"] if i < warmup else max_fs
    if 0 < cap < fs:
        dtheta, fs_applied = dtheta * np.sqrt(cap / fs), cap
    fs_exact = np.nan
    exact_now = args["fs_exact_sweeps"] < 0 or i < args["fs_exact_sweeps"]
    if exact_now and cap > 0 and prev_batch is not None:
        # shrink until the actual 1 - F is within the cap: the scale 0.9 x the quadratic guess per try
        current = log_amplitudes(prev_batch[0])
        for _ in range(12):
            fs_exact = step_infidelity(dtheta, prev_batch, current)
            if not fs_exact > cap:
                break
            c = 0.9 * np.sqrt(cap / fs_exact) if np.isfinite(fs_exact) else 0.25
            dtheta, fs_applied = dtheta * c, fs_applied * c**2
    state.update(dtheta)
    prev_Obar = optimizer.last_Obar
    if exact_now:
        prev_batch = keep_batch(samples)

    g_norm = optimizer._buffers["g_norm"]
    rows.append((i, optimizer.energy, optimizer.VarE, optimizer.energy_H, optimizer.VarE_H, optimizer.penalty_value,
                 optimizer.VarPenalty, step_norm, g_norm, fs, fs_applied, fs_same, lam, fs_exact))
    np.savetxt(f"{path}/data_energy_{run_id}.txt",
               np.array([[np.nan if v is None else float(np.real(v)) for v in row] for row in rows]),
               header="sweep E VarE E_H VarE_H LmLp VarLmLp step_norm g_norm fs fs_applied fs_same lambda fs_exact")
    state.save(f"{path}/state0_{run_id}.eqx")

    line = (f"  sweep {i:5d}  E = {float(optimizer.energy):.6f}  VarE = {float(optimizer.VarE):.3e}  "
            f"|step| = {step_norm:.3e}  |g| = {float(g_norm):.3e}  fs = {fs:.2e}"
            + (f"  fs_exact = {fs_exact:.2e}" if np.isfinite(fs_exact) else ""))
    if do_ED and (i % args["ed_every"] == 0 or i == args["nbr_sweeps"] - 1):
        line += "  | " + record_exact(i)
    if i < 5 or i % 10 == 0 or i == args["nbr_sweeps"] - 1:
        print(line, flush=True)

print("VMC training completed in: ", datetime.now() - start_time)
meta["time"] = str(datetime.now() - start_time)
save_meta()
