# Rotation-closed exciton members at transverse field h: the starting set of FuzzySphereRotatedNES.py.
# Every member is built at h = 0, where the exciton number k is conserved, and then rotated exactly to h:
#   k=0  the vacuum |Up>, one determinant (W = 0)
#   k=1  the exciton B_{L,L}|Up>, one determinant (W = 0)
#   k=2  --nbr-dets charge-only block determinants on one shared network, fitted once to the exact
#        two-exciton state; 1-F is reported (a stuck fit plateaus at 1e-3 - 1e-2: more dets or another seed)
# With charge-only features R(theta) only rotates the orbital rows, so setting theta is the rotation.
#
#   python FuzzySphereRotatedAnsatz.py -n 8 --h 3.16 --members 0,1,2,2 --nbr-dets 4 --check --path runs/rot
import argparse
import json
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
from qhuantax.rotated import RotatedDetBackflow, member_spec, multidet_block_reference, pretrain_exact


S1 = np.array([[1, 0], [0, 0]])
S2 = np.array([[0, 0], [0, 1]])
SX = np.array([[0, 1], [1, 0]])


parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, required=True, help="number of particles = number of orbitals")
parser.add_argument("--lz", type=int, default=0, help="Lz sector; every member has L = Lz")
parser.add_argument("--z2-sect", type=int, default=1, choices=(1, -1), help="Z2 symmetry sector")
parser.add_argument("--h", type=float, required=True, help="transverse field")
parser.add_argument("--members", required=True,
                    help="exciton number k <= 2 of each member, e.g. 0,1,2,2; a repeated k is the next root")
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
parser.add_argument("--check", action="store_true", help="compare the members with ED at h (N <= 12)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--run-id", type=int, default=1)
parser.add_argument("--path", required=True, help="output directory")
args = vars(parser.parse_args())

N = args["n"]
lz = args["lz"]
z2 = args["z2_sect"]
h = args["h"]
members = [int(k) for k in args["members"].replace(" ", "").split(",") if k]
pspot_inter = np.array([float(v) for v in args["ps_pot"].split(",")])
d, nb, nh = args["attn_dim"], args["nbr_blocks"], args["nbr_heads"]
ndets = args["nbr_dets"]
path = args["path"]
run_id = f"n_{N}_2s_{N-1}_lz_{lz}_z2_{z2}_h_{h:g}_id0{args['run_id']}"

if any(k not in (0, 1, 2) for k in members):
    parser.error("--members: only k = 0, 1, 2 are implemented")
if 0 in members and lz != 0:
    parser.error("k=0 (the vacuum) exists only at Lz = 0")
if members.count(1) > 1:
    parser.error("k=1 has one state per L; it cannot be repeated")
if args["check"] and N > 12:
    parser.error("--check needs ED; use N <= 12")

Path(path).mkdir(parents=True, exist_ok=True)

qtx.set_default_dtype(jnp.float64)
lattice = qtx.sites.Chain(L=N, boundary=0, particle_type="spinful_fermion", Nparticles=N)
symm = FlavourPermQH(eigval=z2)


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


def new_net():
    return Transformer(nblocks=nb, d=d, heads=nh, final_sum=False)


def amplitudes(model, spins):
    out = jax.vmap(model)(jnp.asarray(spins))
    return np.asarray(out.sign * jnp.exp(out.logabs - jnp.max(out.logabs)))


# The k=2 targets: the lowest L = lz states of the two-exciton sector at h = 0
if 2 in members:
    spins2 = ksectors.configs(2)
    E2, V2 = ksectors.eigenstates(tms_H0, 2, nstates=members.count(2), L2=tms_L2)
    H2 = ksectors.block(tms_H0, 2, 2)
    print(f"k=2 sector: {len(spins2)} configurations, exact L={lz} energies {np.round(E2, 6)}", flush=True)


# The members, built at h = 0 and rotated by setting theta
models = []
for i, k in enumerate(members):
    root = members[:i].count(k)
    info = dict(index=i, k=k, root=root, file=f"state{i}_{run_id}.eqx")
    start_time = time.time()
    if k < 2:
        # an exact determinant: no backflow (W = 0)
        U = excitons.vacuum_orbitals() if k == 0 else excitons.exciton_orbitals(lz, lz)
        model = RotatedDetBackflow(new_net(), jnp.asarray(U), W=jnp.zeros((N, d // 2)), charge_only=True)
        info.update(E_h0=0.0 if k == 0 else float(excitons.exciton_energy(lz)), infidelity_h0=0.0)
    else:
        target = V2[:, root]
        model, masks = multidet_block_reference(new_net(), d, N, 2, ndets, gap=args["gap"], seed=args["seed"] + i)
        model, hist, _ = pretrain_exact(model, spins2, target, masks, steps=args["pretrain_steps"],
                                        lr=args["pretrain_lr"], abort_at=None, log_every=args["pretrain_log_every"])
        a = amplitudes(model, spins2)
        infidelity = 1 - (a @ target)**2 / (a @ a) / (target @ target)
        E = float(a @ (H2 @ a) / (a @ a))
        hist = [(int(s), float(v)) for s, v in hist[:-1]]
        info.update(E_h0=float(E2[root]), infidelity_h0=float(infidelity), E_model_h0=E, history=hist)
        print(f"member {i} (k=2 root {root}, {ndets} dets): h=0 1-F {infidelity:.3e}, <H> {E:.6f} "
              f"(exact {E2[root]:.6f}) [{time.time() - start_time:.0f}s] | 1-F along the way: "
              + " ".join(f"{s}:{v:.1e}" for s, v in hist), flush=True)
    model = model.rebuild(theta=theta)
    info.update(spec=member_spec(model, d, nb, nh), time=time.time() - start_time)
    models.append(model)
    eqx.tree_serialise_leaves(f"{path}/{info['file']}", model)
    meta["members"].append(info)
    save_meta()
    print(f"member {i} (k={k}) written: {info['file']}", flush=True)


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
