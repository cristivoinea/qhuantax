# NES at transverse field h from the rotated exciton members of FuzzySphereRotatedAnsatz.py, which all have L = Lz (its
# --lz). The members are released before training: untied (the features may then see the flavour; no amplitude
# changes), the k=2 up-feature suppression lowered to exp(-gap), and the k=0,1 determinants (W = 0, where the network
# has no gradient) given a small random backflow; the angle trains. The L^- L^+ penalty vanishes on L = Lz and lifts a
# level of L > Lz by lambda [L(L+1) - Lz(Lz+1)]; --lmlp-bump-var doubles lambda once its sampled variance has come down.
# Symmetries are imposed only when their quantum number is given: --z2-sect (Ising Z2), --ry-sect (pi rotation about y,
# (-1)^L, Lz = 0 only) and --ry-ph-sect (particle-hole x Ry, P (-1)^L at any Lz, P as in PRX 13, 021009).
# --exact-diag tracks the members against the lowest L = Lz levels of the same sectors, cached in --ed-dir.
#
#   python FuzzySphereRotatedNES.py --ansatz-meta runs/rot/meta_n_8_2s_7_lz_0_z2_1_h_3.16_id04.json \
#       --lmlp-coeff 3.0 --exact-diag --nbr-sweeps 200 --path runs/rotnes
import argparse
import json
import os
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

from qhuantax.nes import NaturalExcitedSR, NaturalLzDetFlavourFlipSampler, NaturalStateSet
from qhuantax.quantumhall_excitons import KSectors
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
parser.add_argument("--lmlp-coeff", type=float, default=0.0, help="coefficient lambda of the L^- L^+ penalty")
parser.add_argument("--lmlp-bump-var", type=float, default=0.0,
                    help="double lambda once, when the sampled Var(L^- L^+) averaged over the last --lmlp-bump-window "
                         "sweeps drops below this; 0 disables")
parser.add_argument("--lmlp-bump-window", type=int, default=50, help="sweeps averaged for --lmlp-bump-var")
parser.add_argument("--exact-diag", action="store_true", help="track the members against ED (feasible up to N=14)")
parser.add_argument("--ed-every", type=int, default=5, help="ED readout every this many sweeps")
parser.add_argument("--ed-k", type=int, default=10, help="lowest levels of the Lz block to compute, all L")
parser.add_argument("--ed-nstates", type=int, default=4, help="number of L = Lz levels kept")
parser.add_argument("--ed-dir", default=None, help="ED cache folder, searched first; defaults to --path")
parser.add_argument("--nbr-sweeps", type=int, default=200, help="number of training iterations")
parser.add_argument("--nbr-samples", type=int, default=512, help="number of sample tuples per iteration")
parser.add_argument("--lr", type=float, default=1e-2, help="starting value of the learning rate")
parser.add_argument("--lr-t0", type=float, default=None, help="lr / (1 + sweep / lr_t0); defaults to N")
parser.add_argument("--norm-clip", type=float, default=0.0,
                    help="cap on the step norm before the learning rate; <= 0 disables")
parser.add_argument("--rshift", type=float, default=None,
                    help="relative diagonal shift of the SR solve; defaults to quantax's")
parser.add_argument("--max-fs", type=float, default=0.0,
                    help="trust region: cap on 1 - F per sweep, F the fidelity of the spans before and after the "
                         "update, to second order dtheta^T S dtheta with S from the previous sweep; 0 disables")
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
h = ansatz["h"] if args["h"] is None else args["h"]
pspot_inter = np.array([float(v) for v in ansatz["args"]["ps_pot"].split(",")])
lam, nsamples, max_fs, do_ED, path = (args[k] for k in ("lmlp_coeff", "nbr_samples", "max_fs", "exact_diag", "path"))
t0 = N if args["lr_t0"] is None else args["lr_t0"]


def run_name(run):
    return f"rotnes_n_{N}_2s_{N-1}_lz_{lz}_z2_{z2}_K_{nstates}_h_{h:g}_id0{run}"


run_id = run_name(args["run_id"])
if lam <= 0 and lz == 0 and nstates >= 3:
    parser.error("with K >= 3 an L >= 1 level lies inside the window at h_c: give --lmlp-coeff")
if lam <= 0 and lz != 0:
    print("warning: no --lmlp-coeff, so nothing keeps the L > Lz levels out of the window", flush=True)
if args["lmlp_bump_var"] > 0 and lam <= 0:
    parser.error("--lmlp-bump-var needs --lmlp-coeff")
sectors = dict(z2=args["z2_sect"], ry=args["ry_sect"], phry=args["ry_ph_sect"])
OPTION = dict(z2="--z2-sect", ry="--ry-sect", phry="--ry-ph-sect")
if sectors["ry"] is not None and (lz != 0 or sectors["ry"] == -1):
    parser.error("--ry-sect: Ry maps Lz to -Lz, so it exists at Lz = 0 only, where the L = 0 members are +1")
if ansatz.get("ry_ph") is not None and sectors["phry"] != ansatz["ry_ph"]:
    print(f"warning: the ansatz members were built in PH x Ry {ansatz['ry_ph']:+d}, but --ry-ph-sect is "
          f"{'not given' if sectors['phry'] is None else format(sectors['phry'], '+d')}", flush=True)
if sectors["z2"] is not None and sectors["z2"] != z2:
    print(f"warning: --z2-sect {sectors['z2']:+d}, but the ansatz was rotated by the angle of sector {z2:+d}",
          flush=True)

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


def member_eigenvalue(member, U):
    """<U> of an ansatz member at h=0, an exact sum over its sector k, which Ry and PH x Ry map onto itself: the
    eigenvalue for an exact member; the projection onto the other sector would keep (1 - <U>)/2 of its weight."""
    model = load_member(member["spec"], N, Path(args["ansatz_meta"]).parent / member["file"]).rebuild(theta=0.0)
    s = jnp.asarray(ksectors.configs(member["k"]))
    t, sign = jax.vmap(U.images)(s)
    psi_s, psi_t = jax.vmap(model)(s), jax.vmap(model)(t[:, 1])
    top = jnp.maximum(jnp.max(psi_s.logabs), jnp.max(psi_t.logabs))
    a, b = (p.sign * jnp.exp(p.logabs - top) for p in (psi_s, psi_t))
    return float(jnp.sum(a * sign[:, 1] * b) / jnp.sum(a * a))


# The members must lie in the imposed Ry and PH x Ry sectors, or the projection removes them (R(theta) commutes with
# both)
member_eigvals = {}
for name, U in (("ry", RyQH(1)), ("phry", PHRyQH(1))):
    if cont_run_id is None and sectors[name] is not None:
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
    if member["spec"]["kind"] == "multidet" and args["gap"] > 0:
        model = set_gap(model, args["gap"])
    if member["spec"]["kind"] == "det" and args["det_backflow_scale"] > 0 and not jnp.any(model.W):
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
    cont_lam = cont_meta.get("lam_bump", {}).get("lam_to", cont_meta["lam"])     # meta["lam"] is the opening one
    start = f"continued from {cont_run_id} (lambda {cont_lam:g} at its end)"
    if lam < cont_lam:
        print(f"warning: --lmlp-coeff {lam:g} is below the lambda {cont_lam:g} that {cont_run_id} ended with",
              flush=True)
member_states = [qtx.state.Variational(model, symm=symm, max_parallel=16384, use_ref=False) for model in models]
state_set = NaturalStateSet(member_states)
print(f"N={N}, Lz={lz}, h={h}, K={nstates}, symmetries imposed: {symm_label}; members k = "
      f"{[m['k'] for m in ansatz['members']]}, {start}; {state_set.nparams_per_state} parameters per member",
      flush=True)

# Chain j opens in its own exciton sector (k = j, from k = 1 at Lz != 0, where k = 0 is empty): every rotated member
# reaches it, and distinct sectors keep the tuple's det A away from zero
rng = np.random.default_rng(args["seed"])
k0 = int(lz != 0)
init_configs = np.stack([ksectors.configs(k)[rng.integers(ksectors.dim(k), size=nsamples)]
                         for k in range(k0, k0 + nstates)], axis=1)
sampler = NaturalLzDetFlavourFlipSampler(state_set, nsamples, n_neighbor=np.arange(1, 3),
                                         initial_spins=init_configs, reweight=args["reweight"],
                                         flip_prob=args["flip_prob"], thermal_steps=args["thermal_steps"],
                                         sweep_steps=args["sweep_steps"])


def exact_levels():
    """The lowest L = lz levels and vectors: from the first usable file in ed_dir, or the --ed-k lowest levels of the
    block (all L), kept by <L^2>."""
    stem = (f"ed_n_{N}_2s_{N-1}_lz_{lz}_z2_{sectors['z2'] or 0}_h_{h:g}_"
            f"V_{ansatz['args']['ps_pot'].replace(',', '_')}_nst_")

    def nst(f):                                       # -1 for the .part file of a save in progress
        n = f.name[len(stem):-len(".npz")]
        return int(n) if n.isdigit() else -1

    for f in sorted((f for f in ed_dir.glob(stem + "*.npz") if nst(f) >= args["ed_nstates"]), key=nst):
        with np.load(f) as cached:
            if cached["V"].shape[0] == dim:
                return cached["E"][:args["ed_nstates"]], cached["V"][:, :args["ed_nstates"]], f"read from {f}"
            print(f"skipping {f.name}: vectors of length {cached['V'].shape[0]}, block dim {dim}", flush=True)

    E, V = tms_H_ED.diagonalize(symm=dense_symm, k=min(args["ed_k"], dim - 2))
    q = np.real(np.einsum("si,si->i", V.conj(), L2_sp @ V))
    L = np.round((-1 + np.sqrt(1 + 4*np.clip(q, 0, None))) / 2).astype(int)
    order = np.argsort(E)
    print("Lowest levels of the block, (L) from <L^2>: " + " ".join(f"{E[j]:.6f}({L[j]})" for j in order), flush=True)
    if np.any(np.abs(q - L*(L + 1)) > 1e-3):
        print(f"  warning: <L^2> = {np.round(q[order], 4)} not all L(L+1): degenerate levels of different L mix",
              flush=True)
    keep = order[L[order] == lz][:args["ed_nstates"]]
    if len(keep) < nstates:
        raise SystemExit(f"only {len(keep)} L={lz} levels among the lowest {args['ed_k']}, {nstates} members: "
                         "raise --ed-k")
    if len(keep) < args["ed_nstates"]:
        print(f"  warning: only {len(keep)} L={lz} levels among the lowest {args['ed_k']}", flush=True)
    # written under a temporary name and renamed, so a run starting meanwhile never reads half a file
    f = ed_dir / f"{stem}{len(keep)}.npz"
    part = ed_dir / f"{stem}{len(keep)}.part{os.getpid()}.npz"
    np.savez(part, E=E[keep], V=V[:, keep])
    os.replace(part, f)
    return E[keep], V[:, keep], f"saved to {f}"


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
    """tr(S^-1 H), subspace fidelity with ED, Ritz values and tr(S^-1 L^2) - K lz(lz+1), the leak out of L = lz."""
    P = np.stack([np.asarray(state.todense(dense_symm).psi, dtype=complex) for state in state_set.states], 1)
    S = P.conj().T @ P
    H_red = P.conj().T @ (H_sp @ P)
    ritz = sp.linalg.eigh((H_red + H_red.conj().T) / 2, (S + S.conj().T) / 2, eigvals_only=True)
    fidelity = np.linalg.norm(V_ED[:, :nstates].conj().T @ sp.linalg.orth(P))**2 / nstates
    trL2 = np.real(np.trace(np.linalg.solve(S, P.conj().T @ (L2_sp @ P)))) - nstates * lz * (lz + 1)
    return float(np.sum(ritz)), float(fidelity), ritz, float(trL2)


# ED in the Lz user basis, Z2-resolved with --z2-sect (~7e5 states at N=14, Lz=0); with --ry-ph-sect only the levels of
# that eigenvalue are targets
if do_ED:
    tms_H_ED = tms_H if int_dropped == 0 else hamiltonian()[0]
    dense_symm = LzUserBasisSymmetry(lz, sectors["z2"] or 0)
    L2_sp = GetL2Terms(N, 2).get_quspin_op(dense_symm).tocsr()
    H_sp = tms_H_ED.get_quspin_op(dense_symm).tocsr()
    dim = H_sp.shape[0]
    ed_dir = Path(path if args["ed_dir"] is None else args["ed_dir"])
    ed_dir.mkdir(parents=True, exist_ok=True)
    E_ED, V_ED, source = exact_levels()
    if sectors["phry"] is not None:
        ph_ED = ed_eigenvalue(V_ED, PHRyQH(1))
        print(f"PH x Ry eigenvalue of the exact L={lz} levels {np.round(E_ED, 6)}: {np.round(ph_ED, 4)}", flush=True)
        keep = np.flatnonzero(ph_ED * sectors["phry"] > 0.5)
        if len(keep) < nstates:
            raise SystemExit(f"only {len(keep)} of the {len(E_ED)} exact L={lz} levels have PH x Ry "
                             f"{sectors['phry']:+d}, {nstates} members: raise --ed-nstates (and --ed-k)")
        E_ED, V_ED = E_ED[keep], V_ED[:, keep]
    print(f"Exact L={lz} levels: {np.round(E_ED, 6)} (lowest {nstates} sum to {np.sum(E_ED[:nstates]):.6f}); "
          f"block dim {dim}, {source}", flush=True)


meta = dict(args=args, ansatz_meta=str(args["ansatz_meta"]), N=N, lz=lz, z2=z2, h=h, K=nstates, lam=lam,
            theta=ansatz["theta"], continued_from=cont_run_id, int_dropped=int_dropped, symmetries=sectors,
            member_eigenvalues=member_eigvals,
            members=[dict(k=m["k"], root=m["root"], file=f"state{i}_{run_id}.eqx",
                          spec=member_spec(model, m["spec"]["d"], m["spec"]["nb"], m["spec"]["nh"]))
                     for i, (m, model) in enumerate(zip(ansatz["members"], models))],
            nparams_per_state=[int(n) for n in state_set.nparams_per_state])
if do_ED:
    meta["E_ED"] = E_ED.tolist()


def save_meta():
    with open(f"{path}/meta_{run_id}.json", "w") as f:
        json.dump(meta, f, indent=1, default=float)


save_meta()
exact_rows = []


def record_exact(sweep):
    """Dense readout at this sweep, appended to data_energy_exact; returns the printed summary."""
    trE, fidelity, ritz, trL2 = dense_readout()
    exact_rows.append((sweep, trE, fidelity, *ritz, trL2))
    np.savetxt(f"{path}/data_energy_exact_{run_id}.txt", np.array(exact_rows),
               header="sweep trE fidelity " + " ".join(f"ritz{k}" for k in range(nstates)) + " trL2")
    return (f"dense trE = {trE:.6f}, fidelity = {fidelity:.6f}, Ritz = {np.round(ritz, 4)}, "
            f"L^2 leak = {trL2:.2e}")


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
                          penalty_every=1 if lam > 0 else 0, diagnostics=args["diagnostics"])

DIAGNOSTICS_COLUMNS = ("sigma_max", "sigma_min", "rank_eff", "obar_fro2")
rows = []

# --profile: wall time of each part of a sweep. Every timed call is waited on (JAX dispatches asynchronously), so the
# parts add up to the sweep; the first sweeps also contain compilation
PROFILE_COLUMNS = ("sample", "eloc_H", "eloc_LmLp", "ebar_rest", "obar", "solve", "update", "readout", "output")
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
# The size of a step is measured on an independent batch, the previous sweep's (an extra one before sweep 0 with
# --max-fs): on the batch the step was solved from, |Obar dtheta|^2 is biased low
prev_Obar = optimizer.get_Obar(sampler.sweep()) if max_fs > 0 else None
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
    if 0 < max_fs < fs:
        dtheta, fs_applied = dtheta * np.sqrt(max_fs / fs), max_fs
    state_set.update(state_set.split_step(dtheta))
    prev_Obar = optimizer.last_Obar
    if args["profile"]:
        jax.block_until_ready([st.model for st in state_set.states])
        timings["update"] = perf_counter() - t

    t = perf_counter()
    g_norm = optimizer._buffers["g_norm"]
    diag = optimizer.diagnostics if args["diagnostics"] else {}
    acceptance = sampler.acceptance if args["diagnostics"] else {}
    rows.append((i, optimizer.energy, optimizer.VarE, optimizer.energy_H, optimizer.VarE_H, optimizer.penalty_value,
                 optimizer.VarPenalty, step_norm, g_norm, *(diag.get(key) for key in DIAGNOSTICS_COLUMNS),
                 fs, fs_applied, fs_same, optimizer.penalty_coeff, acceptance.get("moved"), acceptance.get("valid")))
    np.savetxt(f"{path}/data_energy_{run_id}.txt",
               np.array([[np.nan if v is None else float(np.real(v)) for v in row] for row in rows]),
               header="sweep trE VarE trE_H VarE_H trLmLp VarLmLp step_norm g_norm "
                      + " ".join(DIAGNOSTICS_COLUMNS) + " fs fs_applied fs_same lambda moved valid")

    # --lmlp-bump-var: lambda doubles once, from the next sweep, when the penalty's variance has come down
    window = args["lmlp_bump_window"]
    if args["lmlp_bump_var"] > 0 and "lam_bump" not in meta and len(rows) >= window:
        var_penalty = np.mean([float(np.real(row[6])) for row in rows[-window:]])
        if var_penalty < args["lmlp_bump_var"]:
            meta["lam_bump"] = dict(sweep=i, var_penalty=var_penalty, lam_from=optimizer.penalty_coeff,
                                    lam_to=2 * optimizer.penalty_coeff)
            optimizer.penalty_coeff = 2 * optimizer.penalty_coeff
            save_meta()
            print(f"  sweep {i:5d}: Var(L^- L^+) over the last {window} sweeps {var_penalty:.3g} < "
                  f"{args['lmlp_bump_var']:g}, lambda {meta['lam_bump']['lam_from']:g} -> {optimizer.penalty_coeff:g}",
                  flush=True)
    for index, state in enumerate(state_set.states):
        state.save(f"{path}/state{index}_{run_id}.eqx")
    timings["output"] = perf_counter() - t

    line = (f"  sweep {i:5d}  trE = {float(optimizer.energy):.6f}  VarE = {float(optimizer.VarE):.3e}  "
            f"|step| = {step_norm:.3e}  |g| = {float(g_norm):.3e}  fs = {fs:.2e}"
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
