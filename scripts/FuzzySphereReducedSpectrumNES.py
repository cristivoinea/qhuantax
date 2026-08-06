import numpy as np

from datetime import datetime


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-n", action="store", required=True,
                    help="number of particles")
parser.add_argument("-s", action="store", required=True,
                    help="number of orbitals in the system (2s)")
parser.add_argument("--nstates", action="store", default=2,
                    help="number of NES states")
parser.add_argument("--lz-sect", action="store", required=True,
                    help="Lz symmetry sector")
parser.add_argument("--z2-sect", action="store", default=0,
                    help="Z2 symmetry sector")
parser.add_argument("--ph-sect", action="store", default=0,
                    help="PH symmetry sector (without spin flip)")
parser.add_argument("--nbr-samples", action="store", default=2048,
                    help="number of samples used for each matrix measurement")
parser.add_argument("--nbr-mc", action="store", default=10000,
                    help="number of reduced-matrix MC samples")
parser.add_argument("--seed", action="store", default=42,
                    help="random seed")

parser.add_argument("--run-id", action="store", default=1,
                    help="")
parser.add_argument("--path", action="store", required=True,
                    help="path")

args = vars(parser.parse_args())

N = int(args["n"])
L = int(args["s"])+1
nstates = int(args["nstates"])
lz = int(args["lz_sect"])
z2 = int(args["z2_sect"])
ph = int(args["ph_sect"])
id = int(args["run_id"])
path = str(args["path"])
run_id = f"nes_n_{N}_2s_{L-1}_lz_{lz}_z2_{z2}_ph_{ph}_id0{id}"

nsamples = int(args["nbr_samples"])
nmc = int(args["nbr_mc"])
seed = int(args["seed"])

startTime = datetime.now()
rng = np.random.default_rng(seed)

H_mean = np.zeros((nstates, nstates))
H_err = np.zeros((nstates, nstates))

for row in range(nstates):
    for col in range(row, nstates):
        filename = f"{path}/meastest_H_{row}_{col}_nsamples_{nsamples}_{run_id}.txt"
        data = np.loadtxt(filename)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        H_mean[row,col] = np.mean(data[:,0])
        if data.shape[0] > 1:
            H_err[row,col] = np.std(data[:,0], ddof=1) / np.sqrt(data.shape[0])
        else:
            H_err[row,col] = np.sqrt(max(data[0,1], 0) / nsamples)

        H_mean[col,row] = H_mean[row,col]
        H_err[col,row] = H_err[row,col]

eigvals = np.zeros((nmc, nstates))

for i in range(nmc):
    H_sample = np.zeros((nstates, nstates))
    for row in range(nstates):
        for col in range(row, nstates):
            lo = H_mean[row,col] - H_err[row,col]
            hi = H_mean[row,col] + H_err[row,col]
            H_sample[row,col] = rng.uniform(lo, hi)
            H_sample[col,row] = H_sample[row,col]

    eigvals[i] = np.linalg.eigvalsh(H_sample)

eig_mean = np.mean(eigvals, axis=0)
eig_err = np.std(eigvals, axis=0, ddof=1)
eig_lo = np.percentile(eigvals, 16, axis=0)
eig_hi = np.percentile(eigvals, 84, axis=0)

np.savetxt(
    f"{path}/data_reduced_energy_mc_{run_id}.txt",
    np.vstack((np.arange(nstates), eig_mean, eig_err, eig_lo, eig_hi)).T,
    header="index energy error percentile_16 percentile_84",
)
np.savetxt(
    f"{path}/data_reduced_energy_samples_{run_id}.txt",
    eigvals,
)
np.savetxt(
    f"{path}/data_reduced_hamiltonian_mean_{run_id}.txt",
    H_mean,
)
np.savetxt(
    f"{path}/data_reduced_hamiltonian_error_{run_id}.txt",
    H_err,
)

print("Reduced-spectrum MC completed in: ", datetime.now() - startTime)
for index, energy in enumerate(eig_mean):
    print(index, energy, eig_err[index])
