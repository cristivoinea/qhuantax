r"""Rotation-closed exciton starting states at transverse field h (charge-only backflow,
``qhuantax.rotated``).

Each member is named by its exciton number k (``--members``, e.g. "0,1,2,2"; repeated k take the
next root of that sector). L = Lz = ``--lz``, one Z2 sector. Every member is built at h = 0:

    k=0  |Up>, one determinant (W = 0)
    k=1  B_{L,L}|Up>, one determinant (W = 0)
    k=2  ``--nbr-dets`` charge-only block determinants sharing one network
         (``multidet_block_reference``), fitted once to the exact two-exciton state of ``KSectors``
         by an exact sum over sector k=2. There are no restarts: the fidelity reached is reported. A
         stuck fit shows as a plateau at 1-F ~ 1e-3 - 1e-2 (more determinants, or another --seed).

and then rotated to the field *exactly*: the features of a charge-only model depend only on the
orbital occupations, which R(theta) preserves, so R(theta) acts on the orbital rows, and setting the
model's ``theta`` is the rotation, with no refit. theta is the angle that minimises the exact
Z2-projected energy of the rotated vacuum (``vacuum_angle``) unless ``--theta`` is given; h = 0 gives
theta = 0. ``--check`` (N <= 12) compares every member and the whole set with ED at h.

Written, with tag = n_<N>_2s_<N-1>_lz_<lz>_z2_<z2>_h_<h>_id0<run_id>:
    state<i>_<tag>.eqx    member i, in --members order
    meta_<tag>.json       angle, the spec of every member (rebuilt by ``qhuantax.rotated.load_member``),
                          h = 0 fidelities, and with --check the comparison with ED

Example:
    python FuzzySphereRotatedAnsatz.py -n 8 --h 3.16 --members 0,1,2,2 --nbr-dets 4 --check --path runs/rot
"""
import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import quantax as qtx

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-n", type=int, required=True, help="number of particles = orbitals (nu=1)")
parser.add_argument("--lz", type=int, default=0, help="Lz sector; every member has L = Lz")
parser.add_argument("--z2-sect", type=int, default=1, choices=(1, -1), help="Z2 sector")
parser.add_argument("--h", type=float, required=True, help="transverse field (sets the angle)")
parser.add_argument("--members", required=True,
                    help="exciton number k of each member, comma separated; repeated k take the "
                         "next root of that sector (k <= 2)")
parser.add_argument("--theta", type=float, default=None,
                    help="rotation angle in degrees; default: the optimal vacuum angle at this h")
parser.add_argument("--ps-pot", default="4.75,1",
                    help="inter-flavour pseudopotentials V_0,V_1,... (the factor 2 is applied here)")

parser.add_argument("--attn-dim", type=int, default=16, help="transformer width d")
parser.add_argument("--nbr-blocks", type=int, default=4, help="transformer blocks")
parser.add_argument("--nbr-heads", type=int, default=4, help="attention heads")
parser.add_argument("--nbr-dets", type=int, default=6,
                    help="k=2: number of charge-only block determinants (one shared network)")

parser.add_argument("--gap", type=float, default=20.0,
                    help="k=2 block reference: up-feature suppression, x_up ~ exp(-gap)")
parser.add_argument("--pretrain-steps", type=int, default=4000, help="k=2 fit steps (one fit)")
parser.add_argument("--pretrain-lr", type=float, default=1e-2)
parser.add_argument("--pretrain-log-every", type=int, default=500,
                    help="record (and print at the end) the k=2 loss every this many steps")

parser.add_argument("--check", action="store_true", help="compare with ED at h (N <= 12)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--run-id", type=int, default=1)
parser.add_argument("--path", required=True, help="output directory")
args = parser.parse_args()

N, lz, z2, h = args.n, args.lz, args.z2_sect, args.h
d, nb, nh = args.attn_dim, args.nbr_blocks, args.nbr_heads
members = [int(v) for v in args.members.replace(" ", "").split(",") if v != ""]
if any(k not in (0, 1, 2) for k in members):
    parser.error("--members: only k = 0, 1, 2 are implemented")
if 0 in members and lz != 0:
    parser.error("k=0 (the vacuum) exists only at Lz = 0")
if members.count(1) > 1:
    parser.error("k=1 has one state per L; it cannot be repeated")
if args.check and N > 12:
    parser.error("--check needs ED; use N <= 12")
V = np.array([float(v) for v in args.ps_pot.split(",")])
path = Path(args.path)
path.mkdir(parents=True, exist_ok=True)
tag = f"n_{N}_2s_{N-1}_lz_{lz}_z2_{z2}_h_{h:g}_id0{args.run_id}"

qtx.set_default_dtype(jnp.float64)
lattice = qtx.sites.Chain(L=N, boundary=0, particle_type="spinful_fermion", Nparticles=N)

from qhuantax.quantumhall_excitons import H0Excitons, KSectors, vacuum_angle  # noqa: E402
from qhuantax.quantumhall_operators import (GetL2Terms, GetSpinfulDenIntTerms,  # noqa: E402
                                            GetSpinfulPolTerms)
from qhuantax.quantumhall_symmetries import FlavourPermQH  # noqa: E402
from qhuantax.quantumhall_transformer import Transformer  # noqa: E402
from qhuantax.rotated import (RotatedDetBackflow, member_spec, multidet_block_reference,  # noqa: E402
                              pretrain_exact)

symm = FlavourPermQH(eigval=z2)
H_int = GetSpinfulDenIntTerms(nm=N, ps_pot=2 * V, mat_a=np.diag([1, 0]), mat_b=np.diag([0, 1]))
H_op = H_int if h == 0 else H_int - h * GetSpinfulPolTerms(nm=N, mat=np.array([[0, 1], [1, 0]]))
ex = H0Excitons(N, 2 * V)
ks = KSectors(N, lz)
meta = dict(args=vars(args), N=N, lz=lz, z2=z2, h=h, members=[])

# ------------------------------------------------------------------ the angle
if h == 0 and args.theta is None:
    theta = 0.0
    print("h = 0: exact exciton states, no rotation", flush=True)
else:
    if h != 0:
        th_opt, E_vac, th_hf = vacuum_angle(N, 2 * V, h, z2, H_op)
        meta.update(theta_vacuum=th_opt, E_vacuum=E_vac, theta_hf=th_hf)
        print(f"h = {h}: optimal vacuum angle {np.degrees(th_opt):.4f} deg (projected energy {E_vac:.6f}; "
              f"closed-form HF {np.degrees(th_hf):.4f} deg)", flush=True)
    theta = th_opt if args.theta is None else np.radians(args.theta)
    print(f"every member is rotated by {np.degrees(theta):.4f} deg", flush=True)
meta["theta"] = theta


def new_net():
    return Transformer(nblocks=nb, d=d, heads=nh, final_sum=False)


def det_member(U):
    """A determinant with no backflow (W = 0), so exactly det(U[occ])."""
    return RotatedDetBackflow(new_net(), jnp.asarray(U), W=jnp.zeros((N, d // 2)), charge_only=True)


def amplitudes(model, spins):
    out = jax.vmap(model)(jnp.asarray(spins))
    return np.asarray(out.sign * jnp.exp(out.logabs - jnp.max(out.logabs)))


# ------------------------------------------------------------------ the k=2 targets
nroots2 = members.count(2)
if nroots2:
    spins2 = ks.configs(2)
    E2, V2 = ks.eigenstates(H_int, 2, nstates=nroots2, L2=GetL2Terms(N, 2))
    H2 = ks.block(H_int, 2, 2)
    print(f"k=2 sector: {len(spins2)} configurations, exact L={lz} energies {np.round(E2, 6)}", flush=True)

# ------------------------------------------------------------------ the members
models, counts = [], {}
for i, k in enumerate(members):
    root = counts.get(k, 0)
    counts[k] = root + 1
    info = dict(index=i, k=k, root=root, file=f"state{i}_{tag}.eqx")
    t0 = time.time()
    if k == 0:
        model = det_member(ex.vacuum_orbitals())
        info.update(E_h0=0.0, infidelity_h0=0.0)
    elif k == 1:
        model = det_member(ex.exciton_orbitals(lz, lz))
        info.update(E_h0=float(ex.exciton_energy(lz)), infidelity_h0=0.0)
    else:
        target = V2[:, root]
        model, masks = multidet_block_reference(new_net(), d, N, 2, args.nbr_dets, gap=args.gap,
                                                seed=args.seed + i)
        model, hist, _ = pretrain_exact(model, spins2, target, masks, steps=args.pretrain_steps,
                                        lr=args.pretrain_lr, abort_at=None,
                                        log_every=args.pretrain_log_every)
        a = amplitudes(model, spins2)
        inf = 1 - (a @ target) ** 2 / (a @ a) / (target @ target)
        E = float(a @ (H2 @ a) / (a @ a))
        info.update(E_h0=float(E2[root]), infidelity_h0=float(inf), E_model_h0=E,
                    history=[(int(s), float(v)) for s, v in hist[:-1]])
        print(f"member {i} (k=2 root {root}, {args.nbr_dets} dets): h=0 1-F {inf:.3e}, <H> {E:.6f} "
              f"(exact {E2[root]:.6f}) [{time.time() - t0:.0f}s] | 1-F along the way: "
              + " ".join(f"{s}:{v:.1e}" for s, v in hist[:-1]), flush=True)
    model = model.rebuild(theta=theta)                  # the exact rotation
    info.update(spec=member_spec(model, d, nb, nh), time=time.time() - t0)
    models.append(model)
    eqx.tree_serialise_leaves(path / info["file"], model)
    meta["members"].append(info)
    json.dump(meta, open(path / f"meta_{tag}.json", "w"), indent=1, default=float)
    print(f"member {i} (k={k}) written: {info['file']}", flush=True)

# ------------------------------------------------------------------ optional ED check at h
if args.check:
    from scipy.sparse import identity
    from scipy.sparse.linalg import eigsh

    from qhuantax.quantumhall_userbasis import LzUserBasisSymmetry

    sy = LzUserBasisSymmetry(lz, z2)
    sy.basis_make()
    Hs, Qs = (op.get_quspin_op(sy).tocsr().real for op in (H_op, GetL2Terms(N, 2)))
    L2 = lz * (lz + 1)
    shift = 30.0 * (Qs - L2 * identity(Hs.shape[0])) + (abs(h) * N + 10.0) * identity(Hs.shape[0])
    _, Vv = eigsh(Hs + shift, k=min(16, Hs.shape[0] - 2), which="SA", tol=1e-10)
    q = Vv.T @ (Qs @ Vv)
    a_, B = np.linalg.eigh((q + q.T) / 2)
    W = Vv @ B[:, np.abs(a_ - L2) < 1e-6]
    E, C = np.linalg.eigh(W.T @ (Hs @ W))
    U = (W @ C)[:, :len(members)]
    E = E[:len(members)]
    vecs = []
    for model in models:
        st = qtx.state.Variational(model, symm=symm, max_parallel=16384, use_ref=False)
        v = np.array(st.todense(sy).psi).real
        vecs.append(v / np.linalg.norm(v))
    Bm = np.array(vecs).T
    print(f"\nED h={h} L={lz} z2={z2:+d}: {np.round(E, 4)}")
    rows = []
    for i, w in enumerate(vecs):
        ov = (U.T @ w) ** 2
        rows.append(dict(overlaps=ov.tolist(), E=float(w @ (Hs @ w)), L2=float(w @ (Qs @ w))))
        print(f"  member {i} (k={members[i]}): overlap^2 {np.round(ov, 4)}, <H> {rows[-1]['E']:.4f}, "
              f"<L^2> {rows[-1]['L2']:.4f}")
    S = Bm.T @ Bm
    s_ = np.linalg.eigvalsh(S)
    Hm = Bm.T @ (Hs @ Bm)
    X = np.linalg.eigh(S)[1] / np.sqrt(s_)
    ee, Cr = np.linalg.eigh(X.T @ ((Hm + Hm.T) / 2) @ X)
    Y = Bm @ X @ Cr
    Y /= np.linalg.norm(Y, axis=0)
    angles = np.linalg.svd(U.T @ np.linalg.qr(Bm)[0], compute_uv=False) ** 2
    root_ov = [(U[:, j] @ Y[:, j]) ** 2 for j in range(len(ee))]
    print(f"  overlap matrix cond {s_[-1] / s_[0]:.2f}; cos^2 principal angles to span(ED) "
          f"{np.round(angles, 4)}")
    print(f"  Rayleigh-Ritz roots {np.round(ee, 4)}, overlap^2 with ED {np.round(root_ov, 4)}")
    meta["check"] = dict(E_ED=E.tolist(), members=rows, cond=float(s_[-1] / s_[0]),
                         principal_cos2=angles.tolist(), ritz=ee.tolist(), ritz_overlaps=root_ov)
    json.dump(meta, open(path / f"meta_{tag}.json", "w"), indent=1, default=float)
