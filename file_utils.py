import os
import shutil
import numpy as np
from datetime import datetime

def save_iteration_files(optimizer, iteration):
    """Save all input/output files for the current iteration"""
    # Create directories if they don't exist
    os.makedirs('iterations/scf', exist_ok=True)
    os.makedirs('iterations/nscf', exist_ok=True)
    os.makedirs('iterations/band', exist_ok=True)
    
    # Save current U and gap values
    with open('iterations/parameters.log', 'a') as f:
        if len(optimizer.u_history) > 0 and len(optimizer.gap_history) > 0:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"Iteration {iteration} | "
                    f"U = {optimizer.u_history[-1]:.4f} eV | "
                    f"Gap = {optimizer.gap_history[-1]:.4f} eV\n")
    
    # Save all input/output files
    compound = optimizer.compound
    files_to_save = [
        (f'{compound}.scf.in', f'iterations/scf/{compound}_iter_{iteration:02d}.scf.in'),
        (f'{compound}.scf.out', f'iterations/scf/{compound}_iter_{iteration:02d}.scf.out'),
        (f'{compound}.nscf.in', f'iterations/nscf/{compound}_iter_{iteration:02d}.nscf.in'),
        (f'{compound}.nscf.out', f'iterations/nscf/{compound}_iter_{iteration:02d}.nscf.out'),
        (f'{compound}.band.in', f'iterations/band/{compound}_iter_{iteration:02d}.band.in'),
        (f'{compound}.band.out', f'iterations/band/{compound}_iter_{iteration:02d}.band.out'),
        (f'{compound}_bands.pp.in', f'iterations/band/{compound}_iter_{iteration:02d}_bands.pp.in'),
        (f'{compound}_bands.pp.out', f'iterations/band/{compound}_iter_{iteration:02d}_bands.pp.out'),
        (f'{compound}.bands.dat.gnu', f'iterations/band/{compound}_iter_{iteration:02d}.bands.dat.gnu')
    ]
    
    for src, dst in files_to_save:
        if os.path.exists(src):
            shutil.copy2(src, dst)

def save_to_npz(optimizer, filename='optimization_data.npz'):
    """Save all plot data to .npz archive"""
    # [Keep the original save_to_npz implementation]
    pass
