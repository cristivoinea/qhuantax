r"""NES training at transverse field h, starting from the rotation-closed exciton members of
FuzzySphereRotatedAnsatz.py (charge-only backflow, rotated exactly to the field).

The members are read from the ansatz's meta file (rebuilt by ``qhuantax.rotated.load_member``) and
released for training at h:

* untied (unless --no-untie): the network's down-only embedding row becomes a copy of the up-only one
  (``qhuantax.rotated.untie``). Not a single amplitude changes, but from here on the features may
  depend on the flavour, so the members are ordinary flavour-aware backflows in the rotated frame.
* the k=2 block members suppress their up features by exp(-gap) at h = 0 (exactly zero support
  outside k=2 there); ``--gap`` lowers that (default 8, as in the h-transfer), so the up-feature
  channels get gradients again. The members change by O(exp(-gap)); the dense readout of sweep -1
  shows the set right after untying and release.

The rotation angle stays a trainable parameter of every member. Training is NES (NaturalExcitedSR on
tr(S^-1 (H + lambda L^-L^+))) at h, sampling K-tuples with NaturalLzDetFlavourFlipSampler (reweight 1,
single-electron flavour flips), lr / (1 + sweep / t0). L = 0 only for now. With K >= 3 an L >= 1
level lies inside the window at h_c, so --lmlp-coeff is needed ('auto' takes the smallest safe value
from ED, small blocks only). --exact-diag diagonalises the (Lz=0, z2) block in the Lz user basis with the
flavour Z2 and every --ed-every sweeps tracks tr E, the Ritz values, the subspace fidelity and tr L^2 of the
members against it. The ED result is looked up in --ed-dir (default the output directory) first and only
computed, then saved there, when missing, so the K=3 and K=4 runs at one N share a single diagonalisation.

Written, with run_id = rotnes_n_<N>_2s_<N-1>_lz_0_z2_<z2>_K_<K>_h_<h>_id0<run_id>:
    state<i>_<run_id>.eqx            member i (rewritten every sweep); its spec is in the meta file
    data_energy_<run_id>.txt         sweep, tr E, VarE, tr E_H, VarE_H, tr L^-L^+, Var, step norm
    data_energy_exact_<run_id>.txt   sweep, tr E_dense, subspace fidelity, Ritz values, tr L^2 (--exact-diag)
    ed_n_<N>_..._nst_<n>.npz         the ED levels and vectors (--exact-diag), in --ed-dir; reused when present
    meta_<run_id>.json

Example:
    python FuzzySphereRotatedNES.py --ansatz-meta runs/rot/meta_n_8_2s_7_lz_0_z2_1_h_3.16_id04.json \
        --lmlp-coeff auto --exact-diag --nbr-sweeps 200 --path runs/rotnes
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import quantax as qtx
import scipy.linalg
from scipy.sparse.linalg import LinearOperator, eigsh

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--ansatz-meta", required=True, help="meta_<tag>.json of FuzzySphereRotatedAnsatz.py")
parser.add_argument("--h", type=float, default=None, help="field to train at; default: the ansatz's h")
parser.add_argument("--no-untie", action="store_true", help="keep the features charge-only")
parser.add_argument("--det-backflow-scale", type=float, default=0.1,
                    help="the k=0,1 members start with W = 0 (exact determinants), where their network "
                         "has no gradient and the first SR steps blow up through it (step norms 1e3-1e4 "
                         "at N=8); they get this times the default random W instead, as the trained "
                         "members of FuzzySphereNESHTransfer (--backflow-scale); 0 keeps W = 0")
parser.add_argument("--gap", type=float, default=8.0,
                    help="new up-feature suppression exp(-gap) of the k=2 block members; <= 0 keeps "
                         "the ansatz's value")

parser.add_argument("--lmlp-coeff", default="0",
                    help="coefficient lambda of the L^- L^+ penalty, or 'auto' (needs --exact-diag)")
parser.add_argument("--lmlp-margin", type=float, default=0.1,
                    help="auto: every penalised L >= 1 level clears the K-th L=0 level by this")
parser.add_argument("--lmlp-freq", type=int, default=None,
                    help="how often to measure <H> and <L^- L^+> separately; default 1 with a penalty")
parser.add_argument("--exact-diag", action="store_true",
                    help="track the members against ED in the (Lz=0, z2) user basis (feasible to N=14)")
parser.add_argument("--ed-every", type=int, default=5, help="dense readout every this many sweeps")
parser.add_argument("--ed-nstates", type=int, default=4, help="number of L=0 ED levels")
parser.add_argument("--ed-dir", default=None,
                    help="folder where the ED results are looked up first and, if none is found, saved; "
                         "default --path. Point every run at one folder to diagonalise each (N, z2, h, V) once")

parser.add_argument("--nbr-sweeps", type=int, default=200)
parser.add_argument("--nbr-samples", type=int, default=512, help="sample tuples per sweep")
parser.add_argument("--updater", default="adam", choices=("plain", "adam"))
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--lr-t0", type=float, default=None, help="lr / (1 + sweep / lr_t0); default N")
parser.add_argument("--norm-clip", type=float, default=5.0,
                    help="cap on the norm of the step before the learning rate. The raw NES step is "
                         "O(1e2-1e3) at N=8 even near convergence, and unclipped at lr 1e-2 it wrecks the "
                         "rotated start in a few sweeps; <= 0 disables")
parser.add_argument("--adam-clip", type=float, default=0.0,
                    help="clip the raw SR solve g to this norm inside the Adam updater, before it enters "
                         "the momentum m and second moment v, so one noisy batch cannot steer the next "
                         "~1/(1-mu) steps or damp ~1/(1-beta); 0 disables. --norm-clip still caps the "
                         "final step (Adam's second solve is not covered by this)")
parser.add_argument("--rshift", type=float, default=None,
                    help="relative diagonal shift of the SR solve (quantax auto_shift_eig: eps = "
                         "Tr(T) rshift / sqrt(n) + 1e-6); default quantax's, 1e-6 in single precision")
parser.add_argument("--max-fs", type=float, default=0.0,
                    help="trust region on the change of the state rather than of the parameters: the "
                         "applied update dtheta (after lr) is scaled down so that its sampled "
                         "Fubini-Study size |Obar dtheta|^2 (~ 1 - fidelity of the step) stays <= this; "
                         "0 disables. The size is logged either way (columns fs, fs_applied)")
parser.add_argument("--diagnostics", action="store_true",
                    help="record the spectrum of the SR matrix (sigma_max, sigma_min, rank_eff, |Obar|^2) "
                         "every 5 steps")
parser.add_argument("--thermal-steps", type=int, default=None,
                    help="sampler thermalisation steps before the first sweep; default the sampler's "
                         "20 * K * 2N")
parser.add_argument("--reweight", type=float, default=1.0, help="sample from |det A|^reweight")
parser.add_argument("--flip-prob", type=float, default=0.5,
                    help="fraction of proposals that are single-electron flavour flips")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--run-id", type=int, default=1)
parser.add_argument("--path", required=True, help="output directory")
args = parser.parse_args()

ansatz = json.load(open(args.ansatz_meta))
ansatz_dir = Path(args.ansatz_meta).parent
N, lz, z2 = ansatz["N"], ansatz["lz"], ansatz["z2"]
h = ansatz["h"] if args.h is None else args.h
V = np.array([float(v) for v in ansatz["args"]["ps_pot"].split(",")])
K = len(ansatz["members"])
lmlp_auto = args.lmlp_coeff.strip().lower() == "auto"
lam = None if lmlp_auto else float(args.lmlp_coeff)
LmLp_freq = (1 if (lmlp_auto or lam > 0) else 0) if args.lmlp_freq is None else args.lmlp_freq
t0 = N if args.lr_t0 is None else args.lr_t0
if lz != 0:
    parser.error("L = 0 members only for now")
if lmlp_auto and not args.exact_diag:
    parser.error("--lmlp-coeff auto reads the L-resolved spectrum from ED; add --exact-diag")
if K >= 3 and not (lmlp_auto or lam > 0):
    parser.error("with K >= 3 an L >= 1 level lies inside the window at h_c: give --lmlp-coeff")
path = Path(args.path)
path.mkdir(parents=True, exist_ok=True)
run_id = f"rotnes_n_{N}_2s_{N-1}_lz_0_z2_{z2}_K_{K}_h_{h:g}_id0{args.run_id}"

lattice = qtx.sites.Chain(L=N, boundary=0, particle_type="spinful_fermion", Nparticles=N)

import equinox as eqx  # noqa: E402
from quantax.global_defs import get_subkeys  # noqa: E402
from quantax.nn import lecun_normal  # noqa: E402

from qhuantax.nes import NaturalExcitedSR, NaturalLzDetFlavourFlipSampler, NaturalStateSet  # noqa: E402
from qhuantax.quantumhall_excitons import KSectors  # noqa: E402
from qhuantax.quantumhall_operators import (GetL2Terms, GetLpTerms, GetSpinfulDenIntTerms,  # noqa: E402
                                            GetSpinfulPolTerms)
from qhuantax.quantumhall_symmetries import FlavourPermQH  # noqa: E402
from qhuantax.quantumhall_userbasis import LzUserBasisSymmetry  # noqa: E402
from qhuantax.quantumhall_utils import adaptive_learning_rate_inv  # noqa: E402
from qhuantax.rotated import load_member, member_spec, set_gap, untie  # noqa: E402

symm = FlavourPermQH(eigval=z2)
H_int = GetSpinfulDenIntTerms(nm=N, ps_pot=2 * V, mat_a=np.diag([1, 0]), mat_b=np.diag([0, 1]))
H = H_int if h == 0 else H_int - h * GetSpinfulPolTerms(nm=N, mat=np.array([[0, 1], [1, 0]]))
tms_Lp = GetLpTerms(N, 2)
tms_LmLp = tms_Lp.H @ tms_Lp

# ------------------------------------------------------------------ the members, released
models, specs = [], []
for m in ansatz["members"]:
    model = load_member(m["spec"], N, ansatz_dir / m["file"])
    if not args.no_untie:
        model = untie(model)
    if m["spec"]["kind"] == "multidet" and args.gap > 0:
        model = set_gap(model, args.gap)
    if m["spec"]["kind"] == "det" and args.det_backflow_scale > 0 and not jnp.any(model.W):
        W = lecun_normal(get_subkeys(), model.W.shape, dtype=model.W.dtype) / 10
        model = eqx.tree_at(lambda mm: mm.W, model, args.det_backflow_scale * W)
    models.append(model)
    specs.append(member_spec(model, m["spec"]["d"], m["spec"]["nb"], m["spec"]["nh"]))
members = [qtx.state.Variational(m, symm=symm, max_parallel=16384, use_ref=False) for m in models]
state_set = NaturalStateSet(members)
print(f"N={N}, h={h}, z2={z2:+d}, K={K}: members k = {[m['k'] for m in ansatz['members']]}, theta "
      f"{np.degrees(ansatz['theta']):.4f} deg, {'untied' if not args.no_untie else 'charge-only'}, "
      f"k=2 gap {args.gap if args.gap > 0 else 'unchanged'}; {state_set.nparams_per_state} parameters "
      f"per member", flush=True)

# chain j opens on sector j, which every rotated member reaches and which keeps the tuple's det A away
# from zero (repeated k would otherwise draw the same configuration twice)
ks = KSectors(N, 0)
rng = np.random.default_rng(args.seed)
init_configs = np.stack([ks.configs(j)[rng.integers(ks.dim(j), size=args.nbr_samples)]
                         for j in range(K)], axis=1)
sampler = NaturalLzDetFlavourFlipSampler(state_set, args.nbr_samples, n_neighbor=np.arange(1, 3),
                                         initial_spins=init_configs, reweight=args.reweight,
                                         flip_prob=args.flip_prob, thermal_steps=args.thermal_steps)

# ------------------------------------------------------------------ ED (optional)
if args.exact_diag:
    # the (Lz=0, z2) block in the Lz user basis with the flavour Z2 (~7e5 states at N=14)
    dense_symm = LzUserBasisSymmetry(0, z2)
    Q_dense = GetL2Terms(N, 2).get_quspin_op(dense_symm).tocsr()
    H_dense = H.get_quspin_op(dense_symm).tocsr()
    dim = H_dense.shape[0]
    if lmlp_auto and dim > 20000:
        raise SystemExit(f"--lmlp-coeff auto diagonalises the (Lz=0, z2) block densely; dim {dim} is too large, "
                         "pass an explicit value (e.g. from the FuzzifiED spectra)")
    # the L=0 levels are the same for every run at this (N, z2, h, V), and the diagonalisation is the slow part
    # at large N: look in --ed-dir first (any file with at least --ed-nstates levels, smallest first), and only
    # diagonalise if nothing usable is there
    ed_dir = path if args.ed_dir is None else Path(args.ed_dir)
    ed_dir.mkdir(parents=True, exist_ok=True)
    ed_stem = f"ed_n_{N}_2s_{N-1}_lz_0_z2_{z2}_h_{h:g}_V_{ansatz['args']['ps_pot'].replace(',', '_')}_nst_"

    def ed_nst(f):
        n = f.name[len(ed_stem):-len(".npz")]
        return int(n) if n.isdigit() else -1        # -1 also skips the .part files of a save in progress

    E_ed = None
    for f in sorted((f for f in ed_dir.glob(ed_stem + "*.npz") if ed_nst(f) >= args.ed_nstates), key=ed_nst):
        with np.load(f) as cached:
            if cached["V"].shape[0] != dim:
                print(f"skipping {f.name}: vectors of length {cached['V'].shape[0]}, block dim {dim}", flush=True)
                continue
            E_ed, V_ed = cached["E"][:args.ed_nstates], cached["V"][:, :args.ed_nstates]
        ed_file = f
        break
    from_cache = E_ed is not None
    if not from_cache:
        # matrix-free H + 30 L^2 + shift (which lifts every L > 0 level), so no third sparse matrix of the
        # block's size is built; the shift keeps |Up>-like zero rows from being skipped by eigsh "SA"
        shift = abs(h) * N + 10.0
        # float64 Lanczos even though the stored matrices are float32 (x64 is off in NES runs): float32
        # arithmetic cost ~1e-5 on the levels at N=8
        A = LinearOperator((dim, dim), dtype=np.result_type(H_dense.dtype, np.float64),
                           matvec=lambda x: H_dense @ x + 30.0 * (Q_dense @ x) + shift * x)
        _, Vv = eigsh(A, k=min(args.ed_nstates + 4, dim - 2), which="SA", tol=1e-12)
        q = np.real(np.einsum("si,si->i", Vv.conj(), Q_dense @ Vv))
        Vv = Vv[:, np.abs(q) < 1e-6]
        E_all = np.real(np.einsum("si,si->i", Vv.conj(), H_dense @ Vv))
        o = np.argsort(E_all)
        E_ed, V_ed = E_all[o][:args.ed_nstates], Vv[:, o][:, :args.ed_nstates]
        # written under a temporary name and renamed in one step, so a run starting meanwhile never reads a
        # half-written file
        ed_file = ed_dir / f"{ed_stem}{args.ed_nstates}.npz"
        part = ed_dir / f"{ed_stem}{args.ed_nstates}.part{os.getpid()}.npz"
        np.savez(part, E=E_ed, V=V_ed)
        os.replace(part, ed_file)
    print(f"Exact L=0 levels: {np.round(E_ed, 6)} (lowest {K} sum to {np.sum(E_ed[:K]):.6f}); block dim {dim}, "
          f"{'read from' if from_cache else 'saved to'} {ed_file}", flush=True)

    if lmlp_auto:
        # at Lz = 0 an L level moves up by lambda L(L+1): the smallest lambda that keeps the lowest level
        # of every L >= 1 above the K-th L=0 level by the margin
        qq, Wq = np.linalg.eigh(Q_dense.toarray().real)
        Ls = np.round((-1 + np.sqrt(1 + 4 * np.clip(qq, 0, None))) / 2).astype(int)
        M = H_dense.toarray().real
        E_L = {L: np.linalg.eigvalsh(Wq[:, Ls == L].T @ M @ Wq[:, Ls == L]) for L in np.unique(Ls)}
        need, L_b = max(((E_L[0][K - 1] + args.lmlp_margin - E_L[L][0]) / (L * (L + 1)), L)
                        for L in E_L if L >= 1)
        lam = max(0.0, float(need))
        print(f"lambda = {lam:.4f} (auto): lowest L={L_b} level {E_L[L_b][0]:.4f} vs L=0 level #{K} "
              f"{E_L[0][K - 1]:.4f}, margin {args.lmlp_margin}", flush=True)

    def dense_readout():
        """tr(S^-1 H), subspace fidelity, Ritz values and tr(S^-1 L^2) of the members, from their dense
        vectors; tr L^2 > 0 is the members' leak out of L = 0."""
        P = np.stack([np.asarray(st.todense(dense_symm).psi, dtype=complex) for st in state_set.states], 1)
        S = P.conj().T @ P
        Hm = P.conj().T @ (H_dense @ P)
        ritz = scipy.linalg.eigh((Hm + Hm.conj().T) / 2, (S + S.conj().T) / 2, eigvals_only=True)
        fid = np.linalg.norm(V_ed[:, :K].conj().T @ scipy.linalg.orth(P)) ** 2 / K
        trL2 = float(np.real(np.trace(np.linalg.solve(S, P.conj().T @ (Q_dense @ P)))))
        return float(np.sum(ritz)), float(fid), ritz, trL2

meta = dict(args=vars(args), ansatz_meta=str(args.ansatz_meta), N=N, z2=z2, h=h, K=K, lam=lam,
            theta=ansatz["theta"], members=[dict(k=m["k"], root=m["root"], spec=s,
                                                 file=f"state{i}_{run_id}.eqx")
                                            for i, (m, s) in enumerate(zip(ansatz["members"], specs))],
            nparams_per_state=[int(n) for n in state_set.nparams_per_state])
if args.exact_diag:
    meta["E_ED"] = E_ed.tolist()
json.dump(meta, open(path / f"meta_{run_id}.json", "w"), indent=1, default=float)

# ------------------------------------------------------------------ training
exact_rows, rows = [], []


def track_exact(i):
    trE_d, fid, ritz, trL2 = dense_readout()
    exact_rows.append((i, trE_d, fid, *ritz, trL2))
    np.savetxt(path / f"data_energy_exact_{run_id}.txt", np.array(exact_rows),
               header="sweep trE fidelity " + " ".join(f"ritz{k}" for k in range(K)) + " trL2")
    return trE_d, fid, ritz, trL2


if args.exact_diag:
    trE_d, fid, ritz, trL2 = track_exact(-1)
    print(f"  start (sweep -1): dense trE = {trE_d:.6f} (ED {np.sum(E_ed[:K]):.6f}), fidelity = {fid:.6f}, "
          f"Ritz = {np.round(ritz, 4)}, tr L^2 = {trL2:.2e}", flush=True)

class LoggedAdam(qtx.optimizer.Adam):
    """AdamSR that also keeps the norm of the raw SR solve g (before its own clip) in its buffers,
    which are what comes back out of the jitted solve."""

    def init(self, nparams):
        return {**super().init(nparams), "g_norm": jnp.zeros(())}

    def update(self, core_solve, Obar, Ebar, buffers):
        first = {}

        def solve(*a, **k):
            x = core_solve(*a, **k)
            first.setdefault("g", x)
            return x

        step, buffers = super().update(solve, Obar, Ebar, buffers)
        buffers["g_norm"] = jnp.linalg.norm(first["g"])
        return step, buffers


if args.updater == "adam":
    updater = LoggedAdam(norm_clip=args.adam_clip if args.adam_clip > 0 else None)
else:
    updater = qtx.optimizer.PlainUpdater()
class ObarKeepingSR(NaturalExcitedSR):
    """NaturalExcitedSR that keeps the last Obar (centred, reweighted, NES log-derivative of the
    member determinant), so the Fubini-Study size of a step can be measured outside the solve."""

    def get_Obar(self, samples):
        self.last_Obar = super().get_Obar(samples)
        return self.last_Obar


solver = None if args.rshift is None else qtx.optimizer.auto_shift_eig(rshift=args.rshift)
optimizer = ObarKeepingSR(state_set, H, updater=updater, solver=solver, penalty=tms_LmLp,
                          penalty_coeff=lam, penalty_every=LmLp_freq, diagnostics=args.diagnostics)
diag_keys = ("sigma_max", "sigma_min", "rank_eff", "obar_fro2")
start = datetime.now()
for i in range(args.nbr_sweeps):
    samples = sampler.sweep()
    step = optimizer.get_step(samples)
    # checked before `update`, so the saved checkpoints stay the last good members
    if not (jnp.all(jnp.isfinite(step)) and np.isfinite(optimizer.energy)):
        raise RuntimeError(f"Non-finite update at sweep {i}: energy={optimizer.energy}, "
                           f"{optimizer.nonfinite} of {args.nbr_samples} samples dropped for Eloc, "
                           f"{optimizer.nonfinite_grad} for the log-derivative")
    step_norm = float(jnp.linalg.norm(step))
    if args.norm_clip > 0 and step_norm > args.norm_clip:
        step = step * (args.norm_clip / step_norm)
    dtheta = step * adaptive_learning_rate_inv(args.lr, t0, i)
    # sampled Fubini-Study size of the update (1 - fidelity of the step, to leading order)
    fs = float(jnp.sum(jnp.abs(optimizer.last_Obar @ dtheta) ** 2))
    fs_applied = fs
    if args.max_fs > 0 and fs > args.max_fs:
        dtheta = dtheta * np.sqrt(args.max_fs / fs)
        fs_applied = args.max_fs
    state_set.update(state_set.split_step(dtheta))
    g_norm = optimizer._buffers.get("g_norm")
    diag = optimizer.diagnostics if args.diagnostics else {}
    rows.append((i, optimizer.energy, optimizer.VarE, optimizer.energy_H, optimizer.VarE_H,
                 optimizer.penalty_value, optimizer.VarPenalty, step_norm, g_norm,
                 *(diag.get(key) for key in diag_keys), fs, fs_applied))
    np.savetxt(path / f"data_energy_{run_id}.txt",
               np.array([[np.nan if v is None else float(np.real(v)) for v in r] for r in rows]),
               header="sweep trE VarE trE_H VarE_H trLmLp VarLmLp step_norm g_norm " + " ".join(diag_keys)
               + " fs fs_applied")
    for k, st in enumerate(state_set.states):
        st.save(str(path / f"state{k}_{run_id}.eqx"))
    line = (f"  sweep {i:5d}  trE = {float(optimizer.energy):.6f}  VarE = {float(optimizer.VarE):.3e}  "
            f"|step| = {step_norm:.3e}" + ("" if g_norm is None else f"  |g| = {float(g_norm):.3e}")
            + f"  fs = {fs:.2e}")
    if args.exact_diag and (i % args.ed_every == 0 or i == args.nbr_sweeps - 1):
        trE_d, fid, ritz, trL2 = track_exact(i)
        line += f"  | dense trE = {trE_d:.6f}, fidelity = {fid:.6f}, Ritz = {np.round(ritz, 4)}, tr L^2 = {trL2:.2e}"
    if i < 5 or i % 10 == 0 or i == args.nbr_sweeps - 1:
        print(line, flush=True)

print(f"Training completed in {datetime.now() - start}", flush=True)
meta["time"] = str(datetime.now() - start)
json.dump(meta, open(path / f"meta_{run_id}.json", "w"), indent=1, default=float)
