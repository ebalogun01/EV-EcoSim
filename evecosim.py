"""
This file runs the evecosim.py file with optional arguments. It uses the shell scripts to run the desired
simulations. The shell scripts are located in the same directory as this file. This file requires WSL2 to run without errors.
"""

import subprocess
import argparse


def run_mpc_grid_centralized():
    subprocess.call(['sh', './shell_scripts/run-mpc-grid-central.sh'])
    return


def run_mpc_grid_collocated():
    subprocess.call(['sh', './shell_scripts/run-mpc-grid.sh'])
    return


def run_oneshot_opt():
    subprocess.call(['sh', './shell_scripts/run-oneshot.sh'])
    return


def main(mode):
    if mode == 'oneshot':
        run_oneshot_opt()
    elif mode == 'mpc-grid':
        run_mpc_grid_collocated()
    elif mode == 'mpc-grid-central':
        raise NotImplementedError
    else:
        raise ValueError(f'Invalid mode: {mode}. Please choose from: oneshot, mpc-grid')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='oneshot',
                        help='This flag only included for testing deployment, do not change.')
    args = parser.parse_args()
    main(args.mode)
