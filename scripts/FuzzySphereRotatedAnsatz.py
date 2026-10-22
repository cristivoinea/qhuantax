# Rotation-closed exciton members at transverse field h: the starting set of FuzzySphereRotatedNES.py.
# Every member is built at h = 0, where the exciton number k is conserved, in the Lz sector --lz with its own angular
# momentum L >= Lz (--L-sect, default L = Lz), and then rotated exactly to h:
#   k=0  the vacuum |Up> (L = Lz = 0), one determinant (W = 0)
#   k=1  the exciton B_{L,Lz}|Up>, one determinant (W = 0); at Lz = 0 and L >= 1 that determinant minus the vacuum
#   k=2  --nbr-dets charge-only block determinants on one shared network, fitted once to the exact
#        two-exciton state; 1-F is reported (a stuck fit plateaus at 1e-3 - 1e-2: more dets or another seed)
# With charge-only features R(theta) only rotates the orbital rows, so setting theta is the rotation.
#
#   python FuzzySphereRotatedAnsatz.py -n 8 --h 3.16 --k-sect 0,1,2,2 --nbr-dets 4 --check --path runs/rot
#   python FuzzySphereRotatedAnsatz.py -n 8 --h 3.16 --k-sect 2 --k-root 1 --run-id 2 --path runs/rot
#   python FuzzySphereRotatedAnsatz.py -n 8 --h 3.16 --k-sect 0,1,2,2,2,1 --L-sect 0,0,0,0,2,2 --path runs/rot
# --parallel fits the k=2 members at the same time: the parent computes the targets once and starts one child
# process per member (this script with --fit-member), each pinned to an equal share of the job's CPUs and splitting
# its fit over half as many JAX devices; it then collects them into the one meta file.
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

from qhuantax.quantumhall_excitons import H0Excitons, KSectors, spins_key, vacuum_angle, vacuum_energy
from qhuantax.quantumhall_operators import GetL2Terms, GetSpinfulDenIntTerms, GetSpinfulPolTerms
from qhuantax.quantumhall_symmetries import PHRyQH, fuzzy_sphere_symmetry
from qhuantax.quantumhall_transformer import Transformer
from qhuantax.quantumhall_userbasis import LzUserBasisSymmetry
from qhuantax.rotated import (RotatedDetBackflow, RotatedMultiDetBackflow, load_member, member_spec,
                              multidet_block_reference, pretrain_exact)


S1 = np.array([[1, 0], [0, 0]])
S2 = np.array([[0, 0], [0, 1]])
SX = np.array([[0, 1], [1, 0]])


parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, required=True, help="number of particles = number of orbitals")
parser.add_argument("--lz", type=int, default=0, help="Lz sector (>= 0)")
parser.add_argument("--z2-sect", type=int, default=1, choices=(1, -1), help="Z2 symmetry sector")
parser.add_argument("--h", type=float, required=True, help="transverse field")
parser.add_argument("--k-sect", required=True, help="exciton number k <= 2 of each member, e.g. 0,1,2,2")
parser.add_argument("--L-sect", default=None,
                    help="angular momentum L >= Lz of each member, e.g. 0,0,0,0,2,2; defaults to L = Lz for all")
parser.add_argument("--k-root", default=None,
                    help="root of each member among the states of its (k, L), 0 the lowest, e.g. 0,0,0,1; defaults "
                         "to the next root for a repeated (k, L)")
parser.add_argument("--ry-ph-sect", type=int, default=None, choices=(1, -1),
                    help="put every member in this sector of particle-hole x Ry (eigenvalue P (-1)^L): the k=2 roots "
                         "are counted within it (--k-root) and fitted as projected models, k=0 and k=1 must have it; "
                         "not imposed if omitted. Pass the same value to FuzzySphereRotatedNES.py")
parser.add_argument("--theta", type=float, default=None,
                    help="rotation angle in degrees; defaults to the optimal angle of the Z2 = +1 vacuum at h, "
                         "whatever --z2-sect")
parser.add_argument("--ps-pot", default="4.75,1", help="inter-flavour pseudopotentials")
parser.add_argument("--attn-dim", type=int, default=16, help="attention dimension of the transformer")
parser.add_argument("--nbr-blocks", type=int, default=4, help="number of layers in the transformer")
parser.add_argument("--nbr-heads", type=int, default=4, help="number of attention heads in the transformer")
parser.add_argument("--nbr-dets", type=int, default=6, help="k=2: number of block determinants")
parser.add_argument("--gap", type=float, default=20.0, help="k=2: up features suppressed by exp(-gap)")
parser.add_argument("--pretrain-steps", type=int, default=4000, help="k=2: number of fit steps")
parser.add_argument("--pretrain-lr", type=float, default=1e-2, help="k=2: fit learning rate")
parser.add_argument("--pretrain-lr-t0", type=float, default=None,
                    help="k=2: learning rate decay, lr / (1 + step / t0); defaults to a quarter of --pretrain-steps")
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
ry_ph = args["ry_ph_sect"]
h = args["h"]


def int_list(text):
    return [int(v) for v in text.replace(" ", "").split(",") if v]


members = int_list(args["k_sect"])
Ls = [lz] * len(members) if args["L_sect"] is None else int_list(args["L_sect"])
kL = list(zip(members, Ls))
roots = [kL[:i].count(s) for i, s in enumerate(kL)] if args["k_root"] is None else int_list(args["k_root"])
k2 = [i for i, k in enumerate(members) if k == 2]
pspot_inter = np.array([float(v) for v in args["ps_pot"].split(",")])
d, nb, nh = args["attn_dim"], args["nbr_blocks"], args["nbr_heads"]
ndets = args["nbr_dets"]
path = args["path"]
run_id = f"n_{N}_2s_{N-1}_lz_{lz}_z2_{z2}_ryph_{ry_ph or 0}_h_{h:g}_id0{args['run_id']}"     # ryph 0: not imposed

if any(k not in (0, 1, 2) for k in members):
    parser.error("--k-sect: only k = 0, 1, 2 are implemented")
if len(roots) != len(members) or len(Ls) != len(members):
    parser.error(f"--k-root/--L-sect give {len(roots)}/{len(Ls)} values for {len(members)} members")
if lz < 0 or any(L < lz for L in Ls):
    parser.error("--lz >= 0 and every --L-sect >= Lz")
if any(k == 0 and (L != 0 or lz != 0) for k, L in kL):
    parser.error("k=0 (the vacuum) has L = 0 and exists only at Lz = 0")
if any(k == 1 and L > N - 1 for k, L in kL):
    parser.error(f"k=1: one exciton has L <= {N - 1}")
if any(r != 0 for k, r in zip(members, roots) if k < 2):
    parser.error("k=0 and k=1 have one state per L: root 0 only")
if len(set(zip(members, Ls, roots))) < len(members):
    parser.error("--k-sect/--L-sect/--k-root: a state (k, L, root) appears twice")
if args["check"] and N > 12:
    parser.error("--check needs ED; use N <= 12")
if args["parallel"] and not hasattr(os, "sched_setaffinity"):
    parser.error("--parallel pins the child processes to CPUs, which needs Linux")

Path(path).mkdir(parents=True, exist_ok=True)

qtx.set_default_dtype(jnp.float64)
lattice = qtx.sites.Chain(L=N, boundary=0, particle_type="spinful_fermion", Nparticles=N)
symm = fuzzy_sphere_symmetry(z2=z2, phry=ry_ph)
ksectors = KSectors(N, lz)
work = f"{path}/work_{run_id}"         # --parallel: the k=2 targets from the parent, the members from the children
U_phry = PHRyQH(1)


def image_map(spins):
    """PH x Ry on a list of configurations it maps onto itself (every sector k at fixed Lz):
    <s|U|psi> = sign[s] psi(spins[index[s]]); returns (index, sign)."""
    t, sign = jax.vmap(U_phry.images)(jnp.asarray(spins))
    position = {key: i for i, key in enumerate(spins_key(spins))}
    return np.array([position[key] for key in spins_key(np.asarray(t[:, 1]))]), np.asarray(sign[:, 1], float)


def eigenvalue(vec, images):
    """<U> of a vector on such a list: the PH x Ry eigenvalue of an eigenstate."""
    index, sign = images
    return float(vec @ (sign * vec[index]) / (vec @ vec))


def sector_vector(a, k):
    """Amplitudes a of a member on sector k, projected onto --ry-ph-sect (as NES's member is) when it is given."""
    if ry_ph is None:
        return a
    index, sign = image_map(ksectors.configs(k))
    return 0.5 * (a + ry_ph * sign * a[index])


def new_net():
    return Transformer(nblocks=nb, d=d, heads=nh, final_sum=False)


def amplitudes(model, spins):
    out = jax.vmap(model)(jnp.asarray(spins))
    return np.asarray(out.sign * jnp.exp(out.logabs - jnp.max(out.logabs)))


def save_member(i, k, root, model, start_time, **fit):
    """Member i scaled to unit norm, rotated by theta and written; returns its meta entry and the rotated model.
    The norm is the exact sum over sector k, where the h=0 member lives, and R(theta) keeps it: the members then enter
    NES's det[psi_k(x_j)] as columns of one size, which float32 needs. With --ry-ph-sect it is the norm of the projected
    member, the one NES uses. The meta keeps the norm it had."""
    out = jax.vmap(model)(jnp.asarray(ksectors.configs(k)))
    top = float(jnp.max(out.logabs))
    log_norm = top + np.log(np.linalg.norm(sector_vector(np.asarray(out.sign * jnp.exp(out.logabs - top)), k)))
    # every orbital matrix times f, so every determinant term times f^N: the scale stays in the matrix entries, which
    # float32 (NES) holds, rather than in the multi-det coefficients, which it would round to 0 for a large norm
    f = np.exp(-log_norm / N)
    model = eqx.tree_at(lambda m: (m.U0, m.W), model, (model.U0 * f, model.W * f))
    model = model.rebuild(theta=theta)
    info = dict(index=i, k=k, L=Ls[i], root=root, file=f"state{i}_{run_id}.eqx", **fit,
                log10_norm=float(log_norm / np.log(10)), spec=member_spec(model, d, nb, nh),
                time=time.time() - start_time)
    eqx.tree_serialise_leaves(f"{path}/{info['file']}", model)
    return info, model


def fit_k2(i, root, spins2, target, E_exact, H2):
    """Member i: --nbr-dets block determinants fitted to root `root` of its L in the k=2 sector at h = 0; with
    --ry-ph-sect the projected model is fitted (what NES uses), at no extra cost since the images are k=2
    configurations too."""
    start_time = time.time()
    label = f"member {i} (k=2 L={Ls[i]} root {root}):"
    images = None if ry_ph is None else image_map(spins2)
    model, masks = multidet_block_reference(new_net(), d, N, 2, ndets, gap=args["gap"], seed=args["seed"] + i)
    model, hist, _ = pretrain_exact(model, spins2, target, masks, steps=args["pretrain_steps"], lr=args["pretrain_lr"],
                                    lr_t0=args["pretrain_lr_t0"], abort_at=None, log_every=args["pretrain_log_every"],
                                    log_prefix=label, projection=None if images is None else (*images, ry_ph))
    a = amplitudes(model, spins2)
    if images is not None:
        a = 0.5 * (a + ry_ph * images[1] * a[images[0]])
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
        col = k2.index(i)
        info, _ = fit_k2(i, roots[i], w["spins2"], w["V2"][:, col], float(w["E2"][col]),
                         sp.sparse.load_npz(f"{work}_H2.npz"))
    with open(f"{work}_member{i}.json", "w") as f:
        json.dump(info, f, default=float)
    raise SystemExit


# quantum Hall Hamiltonian: tms_H0 at h = 0, where the members are built, and tms_H at h
tms_H0 = GetSpinfulDenIntTerms(nm=N, ps_pot=2*pspot_inter, mat_a=S1, mat_b=S2)
tms_H = tms_H0 if h == 0 else tms_H0 - h * GetSpinfulPolTerms(nm=N, mat=SX)
tms_L2 = GetL2Terms(N, 2)
excitons = H0Excitons(N, 2*pspot_inter)

meta = dict(args=args, N=N, lz=lz, z2=z2, ry_ph=ry_ph, h=h, members=[])


def save_meta():
    with open(f"{path}/meta_{run_id}.json", "w") as f:
        json.dump(meta, f, indent=1, default=float)


# The rotation angle: the one minimising the Z2 = +1 projected energy of the rotated vacuum (the global vacuum), unless
# given; the same for every --z2-sect, so the sectors share one rotated basis
theta = 0.0
if args["theta"] is not None:
    theta = np.radians(args["theta"])
elif h != 0:
    theta, _, theta_hf = vacuum_angle(N, 2*pspot_inter, h, 1, tms_H)
    meta.update(theta_vacuum=theta, theta_hf=theta_hf)
    print(f"h = {h}: optimal vacuum angle {np.degrees(theta):.4f} deg (closed-form HF {np.degrees(theta_hf):.4f} deg)",
          flush=True)
meta["theta"] = theta
# the vacuum rotated by theta and projected onto --z2-sect: an upper bound on that sector's lowest Lz = 0 level
meta["E_vacuum"] = vacuum_energy(N, theta, z2, tms_H)
print(f"every member is rotated by {np.degrees(theta):.4f} deg (rotated vacuum in Z2 = {z2:+d}: projected energy "
      f"{meta['E_vacuum']:.6f})", flush=True)


def k2_roots(L, need):
    """The lowest `need` L states of the two-exciton sector at h = 0; with --ry-ph-sect those of that PH x Ry
    eigenvalue (the sector is closed under it), more roots being computed until there are enough."""
    if ry_ph is None:
        return ksectors.eigenstates(tms_H0, 2, nstates=need, L2=tms_L2, L=L)
    images2 = image_map(spins2)
    nst = need
    E, V = ksectors.eigenstates(tms_H0, 2, nstates=nst, L2=tms_L2, L=L)
    while True:
        ev = np.array([eigenvalue(V[:, j], images2) for j in range(V.shape[1])])
        keep = np.flatnonzero(ev * ry_ph > 0.5)
        if len(keep) >= need:
            break
        nst *= 2
        try:
            E, V = ksectors.eigenstates(tms_H0, 2, nstates=nst, L2=tms_L2, L=L)
        except ValueError:                                   # fewer L levels in the whole sector
            break
    print(f"k=2 L={L} roots {np.round(E, 6)}: PH x Ry {np.round(ev, 4)}", flush=True)
    if len(keep) < need:
        raise SystemExit(f"--ry-ph-sect {ry_ph:+d}: only {len(keep)} k=2 L={L} roots in that sector, {need} needed")
    return E[keep[:need]], V[:, keep[:need]]


# The k=2 targets, one column per k=2 member (in member order)
if k2:
    spins2 = ksectors.configs(2)
    roots_by_L = {L: k2_roots(L, 1 + max(roots[i] for i in k2 if Ls[i] == L)) for L in sorted({Ls[i] for i in k2})}
    E2 = np.array([roots_by_L[Ls[i]][0][roots[i]] for i in k2])
    V2 = np.stack([roots_by_L[Ls[i]][1][:, roots[i]] for i in k2], 1)
    H2 = ksectors.block(tms_H0, 2, 2)
    print(f"k=2 sector: {len(spins2)} configurations; targets (L, root) {[(Ls[i], roots[i]) for i in k2]}"
          f"{'' if ry_ph is None else f' in PH x Ry {ry_ph:+d}'}, exact energies {np.round(E2, 6)}", flush=True)


# --parallel: the k=2 fits in child processes. Each is pinned to its share of the CPUs by setting it on this thread
# while the child starts (a child inherits it), and splits its fit over half as many JAX devices: with as many devices
# as CPUs the children hung at 0% CPU on the cluster (2026-10-15), probably waiting in the cross-device sum
children = {}
if args["parallel"] and len(k2) > 1:
    np.savez(f"{work}.npz", spins2=spins2, V2=V2, E2=E2, theta=theta)
    sp.sparse.save_npz(f"{work}_H2.npz", H2)
    all_cpus = sorted(os.sched_getaffinity(0))
    if len(all_cpus) < len(k2):
        raise SystemExit(f"--parallel: {len(all_cpus)} CPUs for {len(k2)} k=2 members")
    shares = [set(map(int, c)) for c in np.array_split(all_cpus, len(k2))]
    for i, cpus in zip(k2, shares):
        os.sched_setaffinity(0, cpus)
        children[i] = subprocess.Popen([sys.executable, __file__, *sys.argv[1:], "--fit-member", str(i)],
                                       env={**os.environ, "JAX_NUM_CPU_DEVICES": str(max(1, len(cpus) // 2))})
    os.sched_setaffinity(0, all_cpus)
    print(f"k=2 members {k2} fitting in parallel, CPUs {[sorted(c) for c in shares]}", flush=True)


# The members, built at h = 0 and rotated by setting theta; with --parallel the k=2 ones come from the children
models = []
for i, (k, L, root) in enumerate(zip(members, Ls, roots)):
    if k < 2:
        # exact: one determinant, or two that cancel the vacuum term (k=1 at Lz = 0, L >= 1); no backflow (W = 0)
        Us, coeffs = ([excitons.vacuum_orbitals()], [1.0]) if k == 0 else excitons.exciton_terms(L, lz)
        if len(Us) == 1:
            model = RotatedDetBackflow(new_net(), jnp.asarray(Us[0]), W=jnp.zeros((N, d // 2)), charge_only=True)
        else:            # no rescaling, which would change the relative weight of the terms
            model = RotatedMultiDetBackflow(new_net(), jnp.asarray(np.stack(Us)), W=jnp.zeros((len(Us), N, d // 2)),
                                            coeffs=jnp.asarray(coeffs), charge_only=True, rescale=False)
        configs = ksectors.configs(k)
        a = amplitudes(model, configs)
        if ry_ph is not None:
            # one state per (k, L): its PH x Ry eigenvalue is fixed, so it is checked rather than chosen
            e = eigenvalue(a, image_map(configs))
            if e * ry_ph < 0.5:
                raise SystemExit(f"--ry-ph-sect {ry_ph:+d}: the k={k} L={L} member has PH x Ry {e:+.4f}")
        target = ksectors.eigenstates(tms_H0, k, nstates=1, L2=tms_L2, L=L)[1][:, 0]
        infidelity = 1 - (a @ target)**2 / (a @ a) / (target @ target)
        print(f"member {i} (k={k} L={L}): exact construction, h=0 1-F {infidelity:.1e} against the sector's L={L} "
              f"state", flush=True)
        info, model = save_member(i, k, root, model, time.time(), infidelity_h0=float(infidelity),
                                  E_h0=0.0 if k == 0 else float(excitons.exciton_energy(L)))
    elif i in children:
        if children[i].wait() != 0:
            raise SystemExit(f"member {i}: its fit failed (exit code {children[i].returncode}); inputs in {work}*")
        with open(f"{work}_member{i}.json") as f:
            info = json.load(f)
        model = load_member(info["spec"], N, f"{path}/{info['file']}")
    else:
        col = k2.index(i)
        info, model = fit_k2(i, root, spins2, V2[:, col], float(E2[col]), H2)
    models.append(model)
    meta["members"].append(info)
    save_meta()
    print(f"member {i} (k={k} L={L}) written: {info['file']}", flush=True)
for f in [f"{work}.npz", f"{work}_H2.npz", *(f"{work}_member{i}.json" for i in children)]:
    Path(f).unlink(missing_ok=True)


# Conditioning as NES members. The overlap matrix of the unprojected members is that of the h = 0 ones (R(theta) is
# unitary), block diagonal in k: only the fits of the same k (the k=2 roots) can make it ill-conditioned, and that
# part is exact here. The Z2 projection adds cross terms <psi_i|F R(2 theta)|psi_j>: sin(theta)^N for the vacuum
# (0.34 at N=8, 0.12 at N=24), larger between different k by combinatorial factors; --check gives the full overlap
# matrix at N <= 12 (condition number 4.4 at N=8)
vecs = [(k, sector_vector(amplitudes(m.rebuild(theta=0.0), ksectors.configs(k)), k)) for k, m in zip(members, models)]
S = np.array([[a @ b if ka == kb else 0.0 for kb, b in vecs] for ka, a in vecs])
S /= np.sqrt(np.outer(np.diag(S), np.diag(S)))
s = np.linalg.eigvalsh(S)
max_overlap = float(np.max(np.abs(S - np.eye(len(S))))) if len(S) > 1 else 0.0
meta["conditioning"] = dict(log10_norm=[m["log10_norm"] for m in meta["members"]], overlap_cond_h0=float(s[-1] / s[0]),
                            max_overlap_h0=max_overlap)
save_meta()
print(f"as NES members:\n"
      f"  each scaled to norm 1 (log10 of the norms they had: {np.round(meta['conditioning']['log10_norm'], 2)})\n"
      f"  overlap matrix before the Z2 projection{'' if ry_ph is None else ' (PH x Ry projected)'} condition number "
      f"{s[-1] / s[0]:.4f}, largest |overlap| {max_overlap:.2e}",
      flush=True)


# ED check at h: the members against the lowest levels of their L's (as many of each L as there are members of it),
# one by one and as a span
if args["check"]:
    dense_symm = LzUserBasisSymmetry(lz, z2)
    H_sp, L2_sp = (op.get_quspin_op(dense_symm).tocsr().real for op in (tms_H, tms_L2))
    nev = min(40, H_sp.shape[0] - 2)
    E_all, V_all = tms_H.diagonalize(symm=dense_symm, k=nev)
    q = np.einsum("si,si->i", V_all, L2_sp @ V_all)
    L_all = np.round((-1 + np.sqrt(1 + 4*np.clip(q, 0, None))) / 2)
    order = np.argsort(E_all)
    if ry_ph is not None:
        # the PH x Ry eigenvalue of an ED level, from <s|U|v> = u v(s) on its largest entries
        states = qtx.utils.ints_to_array(dense_symm.basis.states)

        def ed_eigenvalue(v):
            ds = qtx.state.DenseState(np.asarray(v), dense_symm)
            s = states[np.argsort(-np.abs(v))[:8]]
            t, sign = jax.vmap(U_phry.images)(jnp.asarray(s))
            return float(np.median(np.asarray(sign[:, 1]) * ds(np.asarray(t[:, 1])) / ds(s)))

        order = [j for j in order if ed_eigenvalue(V_all[:, j]) * ry_ph > 0.5]
    keep = []
    for L in sorted(set(Ls)):
        found = [j for j in order if L_all[j] == L][:Ls.count(L)]
        if len(found) < Ls.count(L):
            raise SystemExit(f"only {len(found)} L={L} levels{'' if ry_ph is None else ' of that PH x Ry'} among the "
                             f"lowest {nev} of the Lz block, {Ls.count(L)} members")
        keep += found
    keep = sorted(keep, key=lambda j: E_all[j])
    E_ED, U_ED = E_all[keep], V_all[:, keep]

    B = []
    for model in models:
        state = qtx.state.Variational(model, symm=symm, max_parallel=16384, use_ref=False)
        b = np.array(state.todense(dense_symm).psi).real
        B.append(b / np.linalg.norm(b))
    B = np.array(B).T
    sector_label = "" if ry_ph is None else f" PH x Ry {ry_ph:+d}"
    print(f"\nED h={h} Lz={lz} z2={z2:+d}{sector_label}: " + " ".join(f"{E_all[j]:.4f}(L{L_all[j]:.0f})" for j in keep))
    rows = []
    for i, b in enumerate(B.T):
        rows.append(dict(overlaps=((U_ED.T @ b)**2).tolist(), E=float(b @ (H_sp @ b)), L2=float(b @ (L2_sp @ b))))
        print(f"  member {i} (k={members[i]} L={Ls[i]}): overlap^2 {np.round(rows[-1]['overlaps'], 4)}, "
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
    ritz_L2 = np.einsum("si,si->i", Y, L2_sp @ Y)
    print(f"  overlap matrix cond {s[-1] / s[0]:.2f}; cos^2 principal angles to span(ED) {np.round(cos2, 4)}")
    print(f"  Rayleigh-Ritz roots {np.round(ritz, 4)} (<L^2> {np.round(ritz_L2, 3)}), overlap^2 with ED "
          f"{np.round(ritz_overlaps, 4)}")
    meta["check"] = dict(E_ED=E_ED.tolist(), L_ED=L_all[keep].tolist(), members=rows, cond=float(s[-1] / s[0]),
                         principal_cos2=cos2.tolist(), ritz=ritz.tolist(), ritz_L2=ritz_L2.tolist(),
                         ritz_overlaps=ritz_overlaps)
    save_meta()
