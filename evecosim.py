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


def run_grid_base_case():
    subprocess.call(['sh', './shell_scripts/run-basecase-grid.sh'])
    return


def main(mode):
    if mode == 'oneshot':
        run_oneshot_opt()
    elif mode == 'mpc-grid':
        run_mpc_grid_collocated()
    elif mode == 'base-case-grid':
        run_grid_base_case()
    elif mode == 'mpc-grid-central':
        raise NotImplementedError
    else:
        raise ValueError(f'Invalid mode: {mode}. Please choose from: oneshot, mpc-grid, base-case-grid')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='oneshot',
                        help='Oneshot (offline) optimization mode is default. '
                             'Choose from: oneshot, mpc-grid, base-case-grid')
    args = parser.parse_args()
    main(args.mode)
