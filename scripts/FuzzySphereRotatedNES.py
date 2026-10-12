# NES at transverse field h, starting from the rotated exciton members of FuzzySphereRotatedAnsatz.py.
# The members are released before training: untied (the features may then depend on the flavour, no amplitude
# changes), the k=2 up-feature suppression lowered to exp(-gap), and the k=0,1 determinants (W = 0, where the
# network has no gradient and the first SR steps blow up) given a small random backflow. The angle stays
# trainable. With K >= 3 an L >= 1 level lies inside the window at h_c, so --lmlp-coeff is needed.
# --exact-diag tracks the members against the lowest L=0 levels of the (Lz=0, z2) block, cached in --ed-dir.
#
#   python FuzzySphereRotatedNES.py --ansatz-meta runs/rot/meta_n_8_2s_7_lz_0_z2_1_h_3.16_id04.json \
#       --lmlp-coeff 3.0 --exact-diag --nbr-sweeps 200 --path runs/rotnes
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import quantax as qtx
import scipy as sp
from quantax.global_defs import get_subkeys
from quantax.nn import lecun_normal

from qhuantax.nes import NaturalExcitedSR, NaturalLzDetFlavourFlipSampler, NaturalStateSet
from qhuantax.quantumhall_excitons import KSectors
from qhuantax.quantumhall_operators import GetL2Terms, GetLpTerms, GetSpinfulDenIntTerms, GetSpinfulPolTerms
from qhuantax.quantumhall_symmetries import FlavourPermQH
from qhuantax.quantumhall_userbasis import LzUserBasisSymmetry
from qhuantax.quantumhall_utils import adaptive_learning_rate_inv
from qhuantax.rotated import load_member, member_spec, set_gap, untie


S1 = np.array([[1, 0], [0, 0]])
S2 = np.array([[0, 0], [0, 1]])
SX = np.array([[0, 1], [1, 0]])


parser = argparse.ArgumentParser()
parser.add_argument("--ansatz-meta", required=True, help="meta file written by FuzzySphereRotatedAnsatz.py")
parser.add_argument("--h", type=float, default=None, help="transverse field; defaults to the ansatz's")
parser.add_argument("--no-untie", action="store_true", help="keep the features charge-only")
parser.add_argument("--det-backflow-scale", type=float, default=0.1,
                    help="k=0,1 members: W = this x the default random init; 0 keeps W = 0")
parser.add_argument("--gap", type=float, default=8.0,
                    help="k=2 members: up features suppressed by exp(-gap); <= 0 keeps the ansatz's")
parser.add_argument("--lmlp-coeff", default="0",
                    help="coefficient of the L^- L^+ penalty, or 'auto' (needs --exact-diag, small N only)")
parser.add_argument("--lmlp-margin", type=float, default=0.1,
                    help="auto: every L >= 1 level ends this far above the K-th L=0 level")
parser.add_argument("--lmlp-freq", type=int, default=None,
                    help="how often to measure <H> and <L^- L^+> separately; defaults to 1 with a penalty")
parser.add_argument("--exact-diag", action="store_true", help="track the members against ED (feasible up to N=14)")
parser.add_argument("--ed-every", type=int, default=5, help="ED readout every this many sweeps")
parser.add_argument("--ed-k", type=int, default=10,
                    help="lowest levels of the Lz=0 block to compute, all L; 10 holds 4 L=0 levels at N=8-14")
parser.add_argument("--ed-nstates", type=int, default=4, help="number of L=0 levels kept")
parser.add_argument("--ed-dir", default=None, help="ED cache folder, searched first; defaults to --path")
parser.add_argument("--nbr-sweeps", type=int, default=200, help="number of training iterations")
parser.add_argument("--nbr-samples", type=int, default=512, help="number of sample tuples per iteration")
parser.add_argument("--updater", default="adam", choices=("plain", "adam"))
parser.add_argument("--lr", type=float, default=1e-2, help="starting value of the learning rate")
parser.add_argument("--lr-t0", type=float, default=None, help="lr / (1 + sweep / lr_t0); defaults to N")
parser.add_argument("--norm-clip", type=float, default=5.0,
                    help="cap on the step norm before the learning rate; <= 0 disables")
parser.add_argument("--adam-clip", type=float, default=0.0, help="cap on the raw SR solve inside Adam; 0 disables")
parser.add_argument("--rshift", type=float, default=None,
                    help="relative diagonal shift of the SR solve; defaults to quantax's")
parser.add_argument("--max-fs", type=float, default=0.0,
                    help="trust region: cap on the sampled Fubini-Study size |Obar dtheta|^2 of the update; "
                         "0 disables")
parser.add_argument("--diagnostics", action="store_true", help="record the SR matrix spectrum every 5 sweeps")
parser.add_argument("--thermal-steps", type=int, default=None,
                    help="sampler thermalisation steps; defaults to the sampler's")
parser.add_argument("--reweight", type=float, default=1.0, help="reweight factor for training sampling")
parser.add_argument("--flip-prob", type=float, default=0.5, help="fraction of flavour-flip proposals")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--run-id", type=int, default=1)
parser.add_argument("--path", required=True, help="output directory")
args = vars(parser.parse_args())

ansatz = json.load(open(args["ansatz_meta"]))
N, lz, z2 = ansatz["N"], ansatz["lz"], ansatz["z2"]
nstates = len(ansatz["members"])
h = ansatz["h"] if args["h"] is None else args["h"]
pspot_inter = np.array([float(v) for v in ansatz["args"]["ps_pot"].split(",")])
lmlp_auto = args["lmlp_coeff"].strip().lower() == "auto"
LmLp_coeff = None if lmlp_auto else float(args["lmlp_coeff"])
penalized = lmlp_auto or LmLp_coeff > 0
LmLp_freq = (1 if penalized else 0) if args["lmlp_freq"] is None else args["lmlp_freq"]
nsweeps = args["nbr_sweeps"]
nsamples = args["nbr_samples"]
lr0 = args["lr"]
t0 = N if args["lr_t0"] is None else args["lr_t0"]
norm_clip = args["norm_clip"]
max_fs = args["max_fs"]
do_ED = args["exact_diag"]
path = args["path"]
run_id = f"rotnes_n_{N}_2s_{N-1}_lz_0_z2_{z2}_K_{nstates}_h_{h:g}_id0{args['run_id']}"

if lz != 0:
    parser.error("L = 0 members only for now")
if lmlp_auto and not do_ED:
    parser.error("--lmlp-coeff auto reads the L-resolved spectrum from ED; add --exact-diag")
if nstates >= 3 and not penalized:
    parser.error("with K >= 3 an L >= 1 level lies inside the window at h_c: give --lmlp-coeff")

Path(path).mkdir(parents=True, exist_ok=True)

lattice = qtx.sites.Chain(L=N, boundary=0, particle_type="spinful_fermion", Nparticles=N)
symm = FlavourPermQH(eigval=z2)


# quantum Hall Hamiltonian
tms_H = GetSpinfulDenIntTerms(nm=N, ps_pot=2*pspot_inter, mat_a=S1, mat_b=S2)
if h != 0:
    tms_H = tms_H - h * GetSpinfulPolTerms(nm=N, mat=SX)

# Kept out of `tms_H`: the optimizer takes it as a separate operator, so <H> and <L^- L^+> come back separately
tms_Lp = GetLpTerms(N, 2)
tms_LmLp = tms_Lp.H @ tms_Lp


# The members, released for training at h
def release(member):
    model = load_member(member["spec"], N, Path(args["ansatz_meta"]).parent / member["file"])
    if not args["no_untie"]:
        model = untie(model)
    if member["spec"]["kind"] == "multidet" and args["gap"] > 0:
        model = set_gap(model, args["gap"])
    if member["spec"]["kind"] == "det" and args["det_backflow_scale"] > 0 and not jnp.any(model.W):
        W = lecun_normal(get_subkeys(), model.W.shape, dtype=model.W.dtype) / 10
        model = eqx.tree_at(lambda m: m.W, model, args["det_backflow_scale"] * W)
    return model


models = [release(member) for member in ansatz["members"]]
member_states = [qtx.state.Variational(model, symm=symm, max_parallel=16384, use_ref=False) for model in models]
state_set = NaturalStateSet(member_states)
print(f"N={N}, h={h}, z2={z2:+d}, K={nstates}: members k = {[m['k'] for m in ansatz['members']]}, theta "
      f"{np.degrees(ansatz['theta']):.4f} deg, {'charge-only' if args['no_untie'] else 'untied'}, "
      f"k=2 gap {args['gap'] if args['gap'] > 0 else 'unchanged'}; {state_set.nparams_per_state} parameters "
      f"per member", flush=True)

# Chain j opens in exciton sector j: every rotated member reaches it, and distinct sectors keep the tuple's
# det A away from zero
ksectors = KSectors(N, 0)
rng = np.random.default_rng(args["seed"])
init_configs = np.stack([ksectors.configs(j)[rng.integers(ksectors.dim(j), size=nsamples)]
                         for j in range(nstates)], axis=1)
sampler = NaturalLzDetFlavourFlipSampler(state_set, nsamples, n_neighbor=np.arange(1, 3),
                                         initial_spins=init_configs, reweight=args["reweight"],
                                         flip_prob=args["flip_prob"], thermal_steps=args["thermal_steps"])


def L_from_L2(q):
    return np.round((-1 + np.sqrt(1 + 4*np.clip(q, 0, None))) / 2).astype(int)


def exact_L0_levels():
    """The lowest L=0 levels and vectors: read from the first usable file in ed_dir, or computed as in
    FuzzySphereNESTrain (the --ed-k lowest levels of the block, all L), keeping the L=0 ones by <L^2>."""
    stem = f"ed_n_{N}_2s_{N-1}_lz_0_z2_{z2}_h_{h:g}_V_{ansatz['args']['ps_pot'].replace(',', '_')}_nst_"

    def nst(f):                                       # -1 for the .part file of a save in progress
        n = f.name[len(stem):-len(".npz")]
        return int(n) if n.isdigit() else -1

    for f in sorted((f for f in ed_dir.glob(stem + "*.npz") if nst(f) >= args["ed_nstates"]), key=nst):
        with np.load(f) as cached:
            if cached["V"].shape[0] == dim:
                return cached["E"][:args["ed_nstates"]], cached["V"][:, :args["ed_nstates"]], f"read from {f}"
            print(f"skipping {f.name}: vectors of length {cached['V'].shape[0]}, block dim {dim}", flush=True)

    E, V = tms_H.diagonalize(symm=dense_symm, k=min(args["ed_k"], dim - 2))
    q = np.real(np.einsum("si,si->i", V.conj(), L2_sp @ V))
    L, order = L_from_L2(q), np.argsort(E)
    print("Lowest levels of the block, (L) from <L^2>: " + " ".join(f"{E[j]:.6f}({L[j]})" for j in order),
          flush=True)
    if np.any(np.abs(q - L*(L + 1)) > 1e-3):
        print(f"  warning: <L^2> = {np.round(q[order], 4)} not all L(L+1): degenerate levels of different L mix",
              flush=True)
    keep = order[L[order] == 0][:args["ed_nstates"]]
    if len(keep) < nstates:
        raise SystemExit(f"only {len(keep)} L=0 levels among the lowest {args['ed_k']}, {nstates} members: "
                         "raise --ed-k")
    if len(keep) < args["ed_nstates"]:
        print(f"  warning: only {len(keep)} L=0 levels among the lowest {args['ed_k']}", flush=True)
    # written under a temporary name and renamed, so a run starting meanwhile never reads half a file
    f = ed_dir / f"{stem}{len(keep)}.npz"
    part = ed_dir / f"{stem}{len(keep)}.part{os.getpid()}.npz"
    np.savez(part, E=E[keep], V=V[:, keep])
    os.replace(part, f)
    return E[keep], V[:, keep], f"saved to {f}"


def safe_lambda():
    """Smallest L^- L^+ coefficient that lifts the lowest level of every L >= 1 this margin above the K-th L=0
    level: at Lz = 0 an L level moves up by lambda L(L+1). Dense, so small blocks only."""
    q, W = np.linalg.eigh(L2_sp.toarray().real)
    L, H_dense = L_from_L2(q), H_sp.toarray().real
    E_L = {l: np.linalg.eigvalsh(W[:, L == l].T @ H_dense @ W[:, L == l]) for l in np.unique(L)}
    need, L_bind = max(((E_L[0][nstates - 1] + args["lmlp_margin"] - E_L[l][0]) / (l*(l + 1)), l)
                       for l in E_L if l >= 1)
    print(f"lambda = {max(0.0, need):.4f} (auto): lowest L={L_bind} level {E_L[L_bind][0]:.4f} vs L=0 level "
          f"#{nstates} {E_L[0][nstates - 1]:.4f}, margin {args['lmlp_margin']}", flush=True)
    return max(0.0, float(need))


def dense_readout():
    """tr(S^-1 H), subspace fidelity with ED, Ritz values and tr(S^-1 L^2) (the leak out of L = 0) of the members."""
    P = np.stack([np.asarray(state.todense(dense_symm).psi, dtype=complex) for state in state_set.states], 1)
    S = P.conj().T @ P
    H_red = P.conj().T @ (H_sp @ P)
    ritz = sp.linalg.eigh((H_red + H_red.conj().T) / 2, (S + S.conj().T) / 2, eigvals_only=True)
    fidelity = np.linalg.norm(V_ED[:, :nstates].conj().T @ sp.linalg.orth(P))**2 / nstates
    trL2 = np.real(np.trace(np.linalg.solve(S, P.conj().T @ (L2_sp @ P))))
    return float(np.sum(ritz)), float(fidelity), ritz, float(trL2)


# Exact diagonalization in the (Lz=0, z2) user basis, ~7e5 states at N=14
if do_ED:
    dense_symm = LzUserBasisSymmetry(0, z2)
    L2_sp = GetL2Terms(N, 2).get_quspin_op(dense_symm).tocsr()
    H_sp = tms_H.get_quspin_op(dense_symm).tocsr()
    dim = H_sp.shape[0]
    if lmlp_auto and dim > 20000:
        raise SystemExit(f"--lmlp-coeff auto diagonalizes the block densely and dim {dim} is too large; "
                         "pass a value (e.g. from the FuzzifiED spectra)")
    ed_dir = Path(path if args["ed_dir"] is None else args["ed_dir"])
    ed_dir.mkdir(parents=True, exist_ok=True)
    E_ED, V_ED, source = exact_L0_levels()
    print(f"Exact L=0 levels: {np.round(E_ED, 6)} (lowest {nstates} sum to {np.sum(E_ED[:nstates]):.6f}); "
          f"block dim {dim}, {source}", flush=True)
    if lmlp_auto:
        LmLp_coeff = safe_lambda()


meta = dict(args=args, ansatz_meta=str(args["ansatz_meta"]), N=N, z2=z2, h=h, K=nstates, lam=LmLp_coeff,
            theta=ansatz["theta"],
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
    return f"dense trE = {trE:.6f}, fidelity = {fidelity:.6f}, Ritz = {np.round(ritz, 4)}, tr L^2 = {trL2:.2e}"


if do_ED:
    print(f"  start (sweep -1): {record_exact(-1)}; ED trE = {np.sum(E_ED[:nstates]):.6f}", flush=True)


# The optimizer: NES with a log of the raw SR solve norm and the last Obar, for the Fubini-Study size of a step
class LoggedAdam(qtx.optimizer.Adam):
    """Adam that also returns the norm of the raw SR solve g (before its own clip) in its buffers."""

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


if args["updater"] == "adam":
    updater = LoggedAdam(norm_clip=args["adam_clip"] if args["adam_clip"] > 0 else None)
else:
    updater = qtx.optimizer.PlainUpdater()
solver = None if args["rshift"] is None else qtx.optimizer.auto_shift_eig(rshift=args["rshift"])
optimizer = ObarKeepingSR(state_set, tms_H, updater=updater, solver=solver, penalty=tms_LmLp,
                          penalty_coeff=LmLp_coeff, penalty_every=LmLp_freq, diagnostics=args["diagnostics"])

DIAGNOSTICS_COLUMNS = ("sigma_max", "sigma_min", "rank_eff", "obar_fro2")
rows = []
start_time = datetime.now()
for i in range(nsweeps):
    samples = sampler.sweep()
    step = optimizer.get_step(samples)

    # Checked before `update` so the saved checkpoints remain the last good state
    if not (jnp.all(jnp.isfinite(step)) and np.isfinite(optimizer.energy)):
        raise RuntimeError(f"Non-finite update at sweep {i}: energy={optimizer.energy}, "
                           f"{optimizer.nonfinite} of {nsamples} samples dropped for Eloc, "
                           f"{optimizer.nonfinite_grad} for the log-derivative")

    # Cap the step norm, then (--max-fs) the update's sampled Fubini-Study size, ~ 1 - fidelity of the step
    step_norm = float(jnp.linalg.norm(step))
    if 0 < norm_clip < step_norm:
        step = step * (norm_clip / step_norm)
    dtheta = step * adaptive_learning_rate_inv(lr0, t0, i)
    fs = fs_applied = float(jnp.sum(jnp.abs(optimizer.last_Obar @ dtheta)**2))
    if 0 < max_fs < fs:
        dtheta, fs_applied = dtheta * np.sqrt(max_fs / fs), max_fs
    state_set.update(state_set.split_step(dtheta))

    g_norm = optimizer._buffers.get("g_norm")
    diag = optimizer.diagnostics if args["diagnostics"] else {}
    rows.append((i, optimizer.energy, optimizer.VarE, optimizer.energy_H, optimizer.VarE_H, optimizer.penalty_value,
                 optimizer.VarPenalty, step_norm, g_norm, *(diag.get(key) for key in DIAGNOSTICS_COLUMNS),
                 fs, fs_applied))
    np.savetxt(f"{path}/data_energy_{run_id}.txt",
               np.array([[np.nan if v is None else float(np.real(v)) for v in row] for row in rows]),
               header="sweep trE VarE trE_H VarE_H trLmLp VarLmLp step_norm g_norm "
                      + " ".join(DIAGNOSTICS_COLUMNS) + " fs fs_applied")
    for index, state in enumerate(state_set.states):
        state.save(f"{path}/state{index}_{run_id}.eqx")

    line = (f"  sweep {i:5d}  trE = {float(optimizer.energy):.6f}  VarE = {float(optimizer.VarE):.3e}  "
            f"|step| = {step_norm:.3e}" + ("" if g_norm is None else f"  |g| = {float(g_norm):.3e}")
            + f"  fs = {fs:.2e}")
    if do_ED and (i % args["ed_every"] == 0 or i == nsweeps - 1):
        line += "  | " + record_exact(i)
    if i < 5 or i % 10 == 0 or i == nsweeps - 1:
        print(line, flush=True)

print("NES training completed in: ", datetime.now() - start_time)
meta["time"] = str(datetime.now() - start_time)
save_meta()
