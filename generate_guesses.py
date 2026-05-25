import argparse
import pyscf
import ffsim
import numpy as np
import time

SENIORITY_ANGLES = (np.arccos(-2.0 / np.sqrt(6.0)), np.pi / 4.0)

OR_POP_ANGLES = (np.arccos(-1 / np.sqrt(3.0)), np.pi / 4.0) # a = b = -c

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--norb", type=int)
    parser.add_argument("--molpath")
    parser.add_argument("--npoints", default=3, type=int)
    parser.add_argument("--mode", default="perturb_U")
    parser.add_argument("--noise_scale", default=-6, type=int)

    args = parser.parse_args()

    if args.norb is not None:
        norb = args.norb
    elif args.molpath is not None:
        mol = pyscf.lib.chkfile.load_mol(args.molpath)
        norb = mol.nao
    else:
        raise ValueError("supply --norb or --molpath")

    iu = np.triu_indices(norb, k=1)
    m = iu[0].shape[0]

    rng = np.random.default_rng()

    if args.mode == "perturb_U":
        xs = np.zeros((args.npoints, m + 2))
        for i in range(args.npoints):
            xs[i, :m] = rng.normal(scale=10**(args.noise_scale), size=m)
            xs[i, m:] = SENIORITY_ANGLES
    elif args.mode == "perturb_all":
        xs = np.zeros((args.npoints, m + 2))
        for i in range(args.npoints):
            xs[i, m:] = SENIORITY_ANGLES
            xs[i, :] += rng.normal(scale=10**(args.noise_scale), size=m + 2)
    elif args.mode == "only_U":
        xs = np.zeros((args.npoints, m))
        for i in range(args.npoints):
            xs[i, :m] = rng.normal(scale=10**(args.noise_scale), size=m)
    else:
        raise ValueError("--mode can be 'perturb_U', 'perturb_all', 'only_U'")

    np.savetxt("x0_" + str(norb)
               + "_" + args.mode + "_" + str(args.noise_scale)
               + "_" + time.strftime("%Y%m%d_%H%M%S", time.localtime())
               + ".txt", xs)