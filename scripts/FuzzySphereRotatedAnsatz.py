# Rotation-closed exciton members at transverse field h: the starting set of FuzzySphereRotatedNES.py.
# Every member is built at h = 0, where the exciton number k is conserved, and then rotated exactly to h:
#   k=0  the vacuum |Up>, one determinant (W = 0)
#   k=1  the exciton B_{L,L}|Up>, one determinant (W = 0)
#   k=2  --nbr-dets charge-only block determinants on one shared network, fitted once to the exact
#        two-exciton state; 1-F is reported (a stuck fit plateaus at 1e-3 - 1e-2: more dets or another seed)
# With charge-only features R(theta) only rotates the orbital rows, so setting theta is the rotation.
#
#   python FuzzySphereRotatedAnsatz.py -n 8 --h 3.16 --k-sect 0,1,2,2 --nbr-dets 4 --check --path runs/rot
#   python FuzzySphereRotatedAnsatz.py -n 8 --h 3.16 --k-sect 2 --k-root 1 --run-id 2 --path runs/rot
# --parallel fits the k=2 members at the same time: the parent computes the targets once and starts one child
# process per member (this script with --fit-member), each pinned to an equal share of the job's CPUs, over which
# the fit splits the configurations (JAX_NUM_CPU_DEVICES); it then collects them into the one meta file.
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import quantax as qtx
import scipy as sp

from qhuantax.quantumhall_excitons import H0Excitons, KSectors, vacuum_angle
from qhuantax.quantumhall_operators import GetL2Terms, GetSpinfulDenIntTerms, GetSpinfulPolTerms
from qhuantax.quantumhall_symmetries import FlavourPermQH
from qhuantax.quantumhall_transformer import Transformer
from qhuantax.quantumhall_userbasis import LzUserBasisSymmetry
from qhuantax.rotated import (RotatedDetBackflow, load_member, member_spec, multidet_block_reference,
                              pretrain_exact)


S1 = np.array([[1, 0], [0, 0]])
S2 = np.array([[0, 0], [0, 1]])
SX = np.array([[0, 1], [1, 0]])


parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, required=True, help="number of particles = number of orbitals")
parser.add_argument("--lz", type=int, default=0, help="Lz sector; every member has L = Lz")
parser.add_argument("--z2-sect", type=int, default=1, choices=(1, -1), help="Z2 symmetry sector")
parser.add_argument("--h", type=float, required=True, help="transverse field")
parser.add_argument("--k-sect", required=True, help="exciton number k <= 2 of each member, e.g. 0,1,2,2")
parser.add_argument("--k-root", default=None,
                    help="root of each member within its sector k, 0 the lowest L = lz state, e.g. 0,0,0,1; "
                         "defaults to the next root for a repeated k")
parser.add_argument("--theta", type=float, default=None,
                    help="rotation angle in degrees; defaults to the optimal vacuum angle at h")
parser.add_argument("--ps-pot", default="4.75,1", help="inter-flavour pseudopotentials")
parser.add_argument("--attn-dim", type=int, default=16, help="attention dimension of the transformer")
parser.add_argument("--nbr-blocks", type=int, default=4, help="number of layers in the transformer")
parser.add_argument("--nbr-heads", type=int, default=4, help="number of attention heads in the transformer")
parser.add_argument("--nbr-dets", type=int, default=6, help="k=2: number of block determinants")
parser.add_argument("--gap", type=float, default=20.0, help="k=2: up features suppressed by exp(-gap)")
parser.add_argument("--pretrain-steps", type=int, default=4000, help="k=2: number of fit steps")
parser.add_argument("--pretrain-lr", type=float, default=1e-2, help="k=2: fit learning rate")
parser.add_argument("--pretrain-log-every", type=int, default=500, help="k=2: record 1-F every this many steps")
parser.add_argument("--parallel", action="store_true",
                    help="fit the k=2 members at the same time, one process each on an equal share of the job's CPUs "
                         "(Linux)")
parser.add_argument("--fit-member", type=int, default=None, help=argparse.SUPPRESS)    # a child of --parallel
parser.add_argument("--check", action="store_true", help="compare the members with ED at h (N <= 12)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--run-id", type=int, default=1)
parser.add_argument("--path", required=True, help="output directory")
args = vars(parser.parse_args())

N = args["n"]
lz = args["lz"]
z2 = args["z2_sect"]
h = args["h"]
members = [int(k) for k in args["k_sect"].replace(" ", "").split(",") if k]
if args["k_root"] is None:
    roots = [members[:i].count(k) for i, k in enumerate(members)]
else:
    roots = [int(r) for r in args["k_root"].replace(" ", "").split(",") if r]
pspot_inter = np.array([float(v) for v in args["ps_pot"].split(",")])
d, nb, nh = args["attn_dim"], args["nbr_blocks"], args["nbr_heads"]
ndets = args["nbr_dets"]
path = args["path"]
run_id = f"n_{N}_2s_{N-1}_lz_{lz}_z2_{z2}_h_{h:g}_id0{args['run_id']}"

if any(k not in (0, 1, 2) for k in members):
    parser.error("--k-sect: only k = 0, 1, 2 are implemented")
if len(roots) != len(members):
    parser.error(f"--k-root gives {len(roots)} roots for {len(members)} members")
if 0 in members and lz != 0:
    parser.error("k=0 (the vacuum) exists only at Lz = 0")
if any(r != 0 for k, r in zip(members, roots) if k < 2):
    parser.error("k=0 and k=1 have one state at L = Lz: root 0 only")
if len(set(zip(members, roots))) < len(members):
    parser.error("--k-sect/--k-root: a state (k, root) appears twice")
if args["check"] and N > 12:
    parser.error("--check needs ED; use N <= 12")
if args["parallel"] and not hasattr(os, "sched_setaffinity"):
    parser.error("--parallel pins the child processes to CPUs, which needs Linux")

Path(path).mkdir(parents=True, exist_ok=True)

qtx.set_default_dtype(jnp.float64)
lattice = qtx.sites.Chain(L=N, boundary=0, particle_type="spinful_fermion", Nparticles=N)
symm = FlavourPermQH(eigval=z2)
work = f"{path}/work_{run_id}"         # --parallel: the k=2 targets from the parent, the members from the children


def new_net():
    return Transformer(nblocks=nb, d=d, heads=nh, final_sum=False)


def amplitudes(model, spins):
    out = jax.vmap(model)(jnp.asarray(spins))
    return np.asarray(out.sign * jnp.exp(out.logabs - jnp.max(out.logabs)))


def save_member(i, k, root, model, start_time, **fit):
    """Member i rotated by theta and written; returns its meta entry and the rotated model."""
    model = model.rebuild(theta=theta)
    info = dict(index=i, k=k, root=root, file=f"state{i}_{run_id}.eqx", **fit,
                spec=member_spec(model, d, nb, nh), time=time.time() - start_time)
    eqx.tree_serialise_leaves(f"{path}/{info['file']}", model)
    return info, model


def fit_k2(i, root, spins2, target, E_exact, H2):
    """Member i: --nbr-dets block determinants fitted to root `root` of the k=2 sector at h = 0."""
    start_time = time.time()
    label = f"member {i} (k=2 root {root}):"
    model, masks = multidet_block_reference(new_net(), d, N, 2, ndets, gap=args["gap"], seed=args["seed"] + i)
    model, hist, _ = pretrain_exact(model, spins2, target, masks, steps=args["pretrain_steps"], lr=args["pretrain_lr"],
                                    abort_at=None, log_every=args["pretrain_log_every"], log_prefix=label)
    a = amplitudes(model, spins2)
    infidelity = 1 - (a @ target)**2 / (a @ a) / (target @ target)
    E = float(a @ (H2 @ a) / (a @ a))
    print(f"{label} h=0 1-F {infidelity:.3e}, <H> {E:.6f} (exact {E_exact:.6f}), {ndets} dets "
          f"[{time.time() - start_time:.0f}s]", flush=True)
    return save_member(i, 2, root, model, start_time, E_h0=float(E_exact), infidelity_h0=float(infidelity),
                       E_model_h0=E, history=[(int(s), float(v)) for s, v in hist[:-1]])


# --parallel child: one k=2 member from the parent's targets; its meta entry goes back through a file
if args["fit_member"] is not None:
    i = args["fit_member"]
    qtx.set_random_seed(args["seed"] + i)          # otherwise every child starts from the same network
    with np.load(f"{work}.npz") as w:
        theta = float(w["theta"])
        info, _ = fit_k2(i, roots[i], w["spins2"], w["V2"][:, roots[i]], float(w["E2"][roots[i]]),
                         sp.sparse.load_npz(f"{work}_H2.npz"))
    with open(f"{work}_member{i}.json", "w") as f:
        json.dump(info, f, default=float)
    raise SystemExit


# quantum Hall Hamiltonian: tms_H0 at h = 0, where the members are built, and tms_H at h
tms_H0 = GetSpinfulDenIntTerms(nm=N, ps_pot=2*pspot_inter, mat_a=S1, mat_b=S2)
tms_H = tms_H0 if h == 0 else tms_H0 - h * GetSpinfulPolTerms(nm=N, mat=SX)
tms_L2 = GetL2Terms(N, 2)
excitons = H0Excitons(N, 2*pspot_inter)
ksectors = KSectors(N, lz)

meta = dict(args=args, N=N, lz=lz, z2=z2, h=h, members=[])


def save_meta():
    with open(f"{path}/meta_{run_id}.json", "w") as f:
        json.dump(meta, f, indent=1, default=float)


# The rotation angle: the one minimising the Z2-projected energy of the rotated vacuum, unless given
theta = 0.0
if h != 0:
    theta, E_vac, theta_hf = vacuum_angle(N, 2*pspot_inter, h, z2, tms_H)
    meta.update(theta_vacuum=theta, E_vacuum=E_vac, theta_hf=theta_hf)
    print(f"h = {h}: optimal vacuum angle {np.degrees(theta):.4f} deg (projected energy {E_vac:.6f}; "
          f"closed-form HF {np.degrees(theta_hf):.4f} deg)", flush=True)
if args["theta"] is not None:
    theta = np.radians(args["theta"])
meta["theta"] = theta
print(f"every member is rotated by {np.degrees(theta):.4f} deg", flush=True)


# The k=2 targets: the lowest L = lz states of the two-exciton sector at h = 0
if 2 in members:
    spins2 = ksectors.configs(2)
    E2, V2 = ksectors.eigenstates(tms_H0, 2, nstates=1 + max(r for k, r in zip(members, roots) if k == 2),
                                  L2=tms_L2)
    H2 = ksectors.block(tms_H0, 2, 2)
    print(f"k=2 sector: {len(spins2)} configurations, exact L={lz} energies {np.round(E2, 6)}", flush=True)


# --parallel: the k=2 fits in child processes. Each is pinned to its share of the CPUs by setting it on this thread
# while the child starts (a child inherits it), and splits its fit over that many JAX devices
children = {}
k2 = [i for i, k in enumerate(members) if k == 2]
if args["parallel"] and len(k2) > 1:
    np.savez(f"{work}.npz", spins2=spins2, V2=V2, E2=E2, theta=theta)
    sp.sparse.save_npz(f"{work}_H2.npz", H2)
    all_cpus = os.sched_getaffinity(0)
    if len(all_cpus) < len(k2):
        raise SystemExit(f"--parallel: {len(all_cpus)} CPUs for {len(k2)} k=2 members")
    shares = [set(map(int, c)) for c in np.array_split(sorted(all_cpus), len(k2))]
    for i, cpus in zip(k2, shares):
        os.sched_setaffinity(0, cpus)
        children[i] = subprocess.Popen([sys.executable, __file__, *sys.argv[1:], "--fit-member", str(i)],
                                       env={**os.environ, "JAX_NUM_CPU_DEVICES": str(len(cpus))})
    os.sched_setaffinity(0, all_cpus)
    print(f"k=2 members {k2} fitting in parallel on {[len(c) for c in shares]} CPUs", flush=True)


# The members, built at h = 0 and rotated by setting theta; with --parallel the k=2 ones come from the children
models = []
for i, (k, root) in enumerate(zip(members, roots)):
    if k < 2:
        # an exact determinant: no backflow (W = 0)
        U = excitons.vacuum_orbitals() if k == 0 else excitons.exciton_orbitals(lz, lz)
        model = RotatedDetBackflow(new_net(), jnp.asarray(U), W=jnp.zeros((N, d // 2)), charge_only=True)
        info, model = save_member(i, k, root, model, time.time(), infidelity_h0=0.0,
                                  E_h0=0.0 if k == 0 else float(excitons.exciton_energy(lz)))
    elif i in children:
        if children[i].wait() != 0:
            raise SystemExit(f"member {i}: its fit failed (exit code {children[i].returncode}); inputs in {work}*")
        with open(f"{work}_member{i}.json") as f:
            info = json.load(f)
        model = load_member(info["spec"], N, f"{path}/{info['file']}")
    else:
        info, model = fit_k2(i, root, spins2, V2[:, root], float(E2[root]), H2)
    models.append(model)
    meta["members"].append(info)
    save_meta()
    print(f"member {i} (k={k}) written: {info['file']}", flush=True)
for f in [f"{work}.npz", f"{work}_H2.npz", *(f"{work}_member{i}.json" for i in children)]:
    Path(f).unlink(missing_ok=True)


# ED check at h: the members against the lowest L = lz levels, one by one and as a span
if args["check"]:
    dense_symm = LzUserBasisSymmetry(lz, z2)
    H_sp, L2_sp = (op.get_quspin_op(dense_symm).tocsr().real for op in (tms_H, tms_L2))
    E_all, V_all = tms_H.diagonalize(symm=dense_symm, k=min(16, H_sp.shape[0] - 2))
    q = np.einsum("si,si->i", V_all, L2_sp @ V_all)
    L_all = np.round((-1 + np.sqrt(1 + 4*np.clip(q, 0, None))) / 2)
    keep = [j for j in np.argsort(E_all) if L_all[j] == lz][:len(members)]
    if len(keep) < len(members):
        raise SystemExit(f"only {len(keep)} L={lz} levels among the lowest 16 of the Lz block")
    E_ED, U_ED = E_all[keep], V_all[:, keep]

    B = []
    for model in models:
        state = qtx.state.Variational(model, symm=symm, max_parallel=16384, use_ref=False)
        b = np.array(state.todense(dense_symm).psi).real
        B.append(b / np.linalg.norm(b))
    B = np.array(B).T
    print(f"\nED h={h} L={lz} z2={z2:+d}: {np.round(E_ED, 4)}")
    rows = []
    for i, b in enumerate(B.T):
        rows.append(dict(overlaps=((U_ED.T @ b)**2).tolist(), E=float(b @ (H_sp @ b)), L2=float(b @ (L2_sp @ b))))
        print(f"  member {i} (k={members[i]}): overlap^2 {np.round(rows[-1]['overlaps'], 4)}, "
              f"<H> {rows[-1]['E']:.4f}, <L^2> {rows[-1]['L2']:.4f}")

    # Rayleigh-Ritz in the span of the members
    S = B.T @ B
    H_red = B.T @ (H_sp @ B)
    ritz, C = sp.linalg.eigh((H_red + H_red.T) / 2, S)
    Y = B @ C
    Y /= np.linalg.norm(Y, axis=0)
    s = np.linalg.eigvalsh(S)
    cos2 = np.linalg.svd(U_ED.T @ sp.linalg.orth(B), compute_uv=False)**2
    ritz_overlaps = [float((U_ED[:, j] @ Y[:, j])**2) for j in range(len(ritz))]
    print(f"  overlap matrix cond {s[-1] / s[0]:.2f}; cos^2 principal angles to span(ED) {np.round(cos2, 4)}")
    print(f"  Rayleigh-Ritz roots {np.round(ritz, 4)}, overlap^2 with ED {np.round(ritz_overlaps, 4)}")
    meta["check"] = dict(E_ED=E_ED.tolist(), members=rows, cond=float(s[-1] / s[0]), principal_cos2=cos2.tolist(),
                         ritz=ritz.tolist(), ritz_overlaps=ritz_overlaps)
    save_meta()
