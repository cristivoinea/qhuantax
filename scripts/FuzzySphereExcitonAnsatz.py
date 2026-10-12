r"""Exciton starting states for a set of members at one transverse field h.

Each member is named by its exciton number k (``--members``, e.g. "0,1,2,2"; repeated k take the
next root of that sector). L = Lz = ``--lz``, one Z2 sector.

h = 0 (exact, ``docs/exciton_h0.pdf``):
    k=0  |Up>, one determinant (W = 0)
    k=1  B_{L,L}|Up>, one determinant (W = 0); at L = 0 this is the S^- exciton
    k=2  the block-reference DetBackflow, pretrained on the exact two-exciton state by an exact
         sum over sector k=2 (``pretrain_masked``); ``--release`` opens its structural zeros for a
         later transfer to h > 0 (``release_block``)

h > 0 (rotated):
    k=0  the canted vacuum R(theta)|Up>, one determinant, at the angle that minimises the exact
         Z2-projected energy (``vacuum_angle``)
    k>0  R(theta) applied to the h = 0 member, at the same angle unless ``--theta`` is given:
         k=1 stays one determinant (rows rotated); k=2 is first pretrained at h = 0 as above and
         then refitted to its exact rotation by an importance-sampled fidelity
         (``FlavourRotation``, ``pretrain_rotated``)

States are built by ``transformer_backflow_state`` and saved with ``Variational.save``, so the files
load as ``--init-states`` of FuzzySphereNESHTransfer.py (same --attn-dim/--nbr-blocks/--nbr-heads).
``--check`` (N <= 12) compares every member and the whole set with ED at the same h.

Written, with tag = n_<N>_2s_<N-1>_lz_<lz>_z2_<z2>_h_<h>_id0<run_id>:
    state<i>_<tag>.eqx    member i, in --members order
    meta_<tag>.json       angle, energies, pretraining and rotation losses, check results

Examples:
    python FuzzySphereExcitonAnsatz.py -n 12 --h 0 --members 0,1,2 --release --path runs/ansatz
    python FuzzySphereExcitonAnsatz.py -n 12 --h 3.16 --members 0,1,2,2 --check --path runs/ansatz
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
parser.add_argument("--h", type=float, required=True, help="transverse field")
parser.add_argument("--members", required=True,
                    help="exciton number k of each member, comma separated; repeated k take the "
                         "next root of that sector (k <= 2)")
parser.add_argument("--theta", type=float, default=None,
                    help="rotation angle in degrees for the k > 0 members at h > 0; default: the "
                         "optimal vacuum angle at this h")
parser.add_argument("--ps-pot", default="4.75,1",
                    help="inter-flavour pseudopotentials V_0,V_1,... (the factor 2 is applied here)")

parser.add_argument("--attn-dim", type=int, default=16, help="transformer width d")
parser.add_argument("--nbr-blocks", type=int, default=4, help="transformer blocks")
parser.add_argument("--nbr-heads", type=int, default=4, help="attention heads")

parser.add_argument("--gap", type=float, default=20.0,
                    help="k=2 block reference: up-feature suppression at h = 0, x_up ~ exp(-gap)")
parser.add_argument("--pretrain-steps", type=int, default=4000, help="h = 0 k=2 pretraining steps")
parser.add_argument("--pretrain-lr", type=float, default=1e-2)
parser.add_argument("--pretrain-tol", type=float, default=1e-4,
                    help="retry the h = 0 pretraining from a new network initialisation while "
                         "1-F stays above this (some initialisations stall near 4e-2 at d=16)")
parser.add_argument("--pretrain-retries", type=int, default=4)
parser.add_argument("--pretrain-log-every", type=int, default=0,
                    help="print the h = 0 pretraining loss every this many steps (0: silent)")
parser.add_argument("--release", action="store_true",
                    help="h = 0 only: fill the structural zeros of the k=2 members (release_block) "
                         "so that a transfer to h > 0 can open the neighbouring sectors")
parser.add_argument("--release-delta-up", type=float, default=1e-3)
parser.add_argument("--release-delta-down", type=float, default=1e-3)

parser.add_argument("--rot-steps", type=int, default=3000, help="h > 0: rotation fit steps")
parser.add_argument("--rot-samples", type=int, default=1024, help="h > 0: i.i.d. samples per step")
parser.add_argument("--rot-lr", type=float, default=1e-2)
parser.add_argument("--rot-gap", type=float, default=8.0,
                    help="h > 0: the k=2 member's up-feature gap is lowered to this before the "
                         "rotation fit, so the up rows can acquire backflow")
parser.add_argument("--rot-uniform-mix", type=float, default=0.1,
                    help="fraction of the rotation batch drawn with flip probability 1/2")

parser.add_argument("--check", action="store_true", help="compare with ED (N <= 12)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--run-id", type=int, default=1)
parser.add_argument("--path", required=True, help="output directory")
args = parser.parse_args()

N, lz, z2, h = args.n, args.lz, args.z2_sect, args.h
members = [int(v) for v in args.members.replace(" ", "").split(",") if v != ""]
if any(k not in (0, 1, 2) for k in members):
    parser.error("--members: only k = 0, 1, 2 are implemented")
if 0 in members and lz != 0:
    parser.error("k=0 (the vacuum) exists only at Lz = 0")
if args.check and N > 12:
    parser.error("--check needs ED; use N <= 12")
if args.release and h != 0:
    parser.error("--release applies to h = 0 output only")
V = np.array([float(v) for v in args.ps_pot.split(",")])
path = Path(args.path)
path.mkdir(parents=True, exist_ok=True)
tag = f"n_{N}_2s_{N-1}_lz_{lz}_z2_{z2}_h_{h:g}_id0{args.run_id}"

lattice = qtx.sites.Chain(L=N, boundary=0, particle_type="spinful_fermion", Nparticles=N)

from qhuantax.quantumhall_excitons import (FlavourRotation, H0Excitons, block_backflow,  # noqa: E402
                                           block_orbitals, k2_sector_checks, pretrain_masked,
                                           pretrain_rotated, release_block, rotation_orbitals,
                                           vacuum_angle)
from qhuantax.quantumhall_operators import GetSpinfulDenIntTerms, GetSpinfulPolTerms  # noqa: E402
from qhuantax.quantumhall_symmetries import FlavourPermQH  # noqa: E402
from qhuantax.quantumhall_transformer import transformer_backflow_state  # noqa: E402

symm = FlavourPermQH(eigval=z2)
H_int = GetSpinfulDenIntTerms(nm=N, ps_pot=2 * V, mat_a=np.diag([1, 0]), mat_b=np.diag([0, 1]))
H_op = H_int if h == 0 else H_int - h * GetSpinfulPolTerms(nm=N, mat=np.array([[0, 1], [1, 0]]))
ex = H0Excitons(N, 2 * V)
meta = dict(args=vars(args), N=N, lz=lz, z2=z2, h=h, members=[])

# ------------------------------------------------------------------ the angle
if h == 0:
    theta = 0.0
    print("h = 0: exact exciton states, no rotation", flush=True)
else:
    th_opt, E_vac, th_hf = vacuum_angle(N, 2 * V, h, z2, H_op)
    theta = th_opt if args.theta is None else np.radians(args.theta)
    meta.update(theta_vacuum=th_opt, E_vacuum=E_vac, theta_hf=th_hf)
    print(f"h = {h}: optimal vacuum angle {np.degrees(th_opt):.4f} deg (projected energy {E_vac:.6f}; "
          f"closed-form HF {np.degrees(th_hf):.4f} deg); k > 0 members rotated by "
          f"{np.degrees(theta):.4f} deg", flush=True)
meta["theta"] = theta
R = rotation_orbitals(N, theta)


def build(U, index, backflow_scale):
    return transformer_backflow_state(index, N, N, args.attn_dim, args.nbr_blocks, args.nbr_heads,
                                      symm, False, U, orbital_noise=0,
                                      backflow_scale=backflow_scale).model


def amplitudes(model, spins):
    out = jax.vmap(model)(jnp.asarray(spins))
    return np.asarray(out.sign * jnp.exp(out.logabs - jnp.max(out.logabs)))


# ------------------------------------------------------------------ the members
support = ex.support(lz, kmax=2)
spins, ks, P, Hh = support
models, counts = [], {}
for i, k in enumerate(members):
    root = counts.get(k, 0)
    counts[k] = root + 1
    info = dict(index=i, k=k, root=root)
    t0 = time.time()
    if k == 0:
        model = build(R @ ex.vacuum_orbitals(), i, 0.0)          # W = 0: exactly the determinant
        info["E_h0"] = 0.0
    elif k == 1:
        if root > 0:
            parser.error("k=1 has one state per L; it cannot be repeated")
        model = build(R @ ex.exciton_orbitals(lz, lz), i, 0.0)
        info["E_h0"] = ex.exciton_energy(lz)
    else:
        E2, Phis = ex.two_exciton_states(lz, root + 1)
        target = ex.target_amplitudes(Phis[root], P, Hh)
        for attempt in range(args.pretrain_retries):
            model, masks = block_backflow(build(block_orbitals(N, 2, seed=args.seed + root)[0], i, 1.0),
                                          2, gap=args.gap)
            model, hist = pretrain_masked(model, spins, target, masks, steps=args.pretrain_steps,
                                          lr=args.pretrain_lr, lr_t0=args.pretrain_steps / 4,
                                          log_every=max(args.pretrain_log_every, 1),
                                          verbose=args.pretrain_log_every > 0)
            chk = k2_sector_checks(ex, model, spins, ks, P, Hh, target, E2[root])
            print(f"member {i} (k=2 root {root}): h=0 pretraining attempt {attempt}: 1-F "
                  f"{1 - chk['F']:.2e}, <H> {chk['E']:.6f} (exact {E2[root]:.6f}) "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            if 1 - chk["F"] < args.pretrain_tol:
                break
        info.update(E_h0=float(E2[root]), pretrain_infidelity=1 - chk["F"],
                    pretrain_attempts=attempt + 1)
        if h == 0 and args.release:
            model = release_block(model, masks, args.release_delta_up, args.release_delta_down,
                                  seed=args.seed + i)
        elif h != 0:
            sk = spins[ks == 2]
            rot = FlavourRotation(N, sk, amplitudes(model, sk), theta)
            b2 = model.net.layers[-2].b2.at[0::2].set(-args.rot_gap * np.sqrt(model.net.nblocks + 1))
            model = eqx.tree_at(lambda m: m.net.layers[-2].b2,
                                eqx.tree_at(lambda m: m.U0, model, jnp.asarray(R) @ model.U0), b2)
            t1 = time.time()
            model, rh = pretrain_rotated(model, rot, theta, steps=args.rot_steps,
                                         nsamples=args.rot_samples, lr=args.rot_lr,
                                         uniform_mix=args.rot_uniform_mix, seed=args.seed + i,
                                         log_every=max(args.rot_steps // 6, 1))
            info["rotation_sampled_infidelity"] = float(np.mean(rh[-50:]))
            print(f"member {i}: rotation fit, sampled 1-F {info['rotation_sampled_infidelity']:.2e} "
                  f"[{time.time() - t1:.0f}s]", flush=True)
    info["time"] = time.time() - t0
    models.append(model)
    eqx.tree_serialise_leaves(path / f"state{i}_{tag}.eqx", model)
    meta["members"].append(info)
    json.dump(meta, open(path / f"meta_{tag}.json", "w"), indent=1, default=float)
    print(f"member {i} (k={k}) written: state{i}_{tag}.eqx", flush=True)

# ------------------------------------------------------------------ optional ED check
if args.check:
    from scipy.sparse import identity
    from scipy.sparse.linalg import eigsh

    from qhuantax.quantumhall_operators import GetL2Terms
    from qhuantax.quantumhall_userbasis import LzUserBasisSymmetry

    sy = LzUserBasisSymmetry(lz, z2)
    sy.basis_make()
    Hs, Qs = (op.get_quspin_op(sy).tocsr().real for op in (H_op, GetL2Terms(N, 2)))
    L2 = lz * (lz + 1)
    shift = 30.0 * (Qs - L2 * identity(Hs.shape[0])) + (abs(h) * N + 10.0) * identity(Hs.shape[0])
    _, Vv = eigsh(Hs + shift, k=min(16, Hs.shape[0] - 2), which="SA", tol=1e-10)
    q = Vv.T @ (Qs @ Vv)
    a, B = np.linalg.eigh((q + q.T) / 2)
    W = Vv @ B[:, np.abs(a - L2) < 1e-6]
    E, C = np.linalg.eigh(W.T @ (Hs @ W))
    U = (W @ C)[:, :len(members)]
    E = E[:len(members)]
    vecs = []
    for i, model in enumerate(models):
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
    print(f"  Rayleigh-Ritz roots {np.round(ee, 4)}, overlap^2 with ED {np.round(root_ov, 4)}, "
          f"coefficient norms {np.round(np.linalg.norm(X @ Cr, axis=0), 3)}")
    meta["check"] = dict(E_ED=E.tolist(), members=rows, cond=float(s_[-1] / s_[0]),
                         principal_cos2=angles.tolist(), ritz=ee.tolist(), ritz_overlaps=root_ov)
    json.dump(meta, open(path / f"meta_{tag}.json", "w"), indent=1, default=float)
