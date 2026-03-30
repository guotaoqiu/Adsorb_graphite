#!/usr/bin/env python3
"""
Utility functions shared across the workflow:
- VASP output parsing
- POSCAR reading/writing
- Job status checking
- POTCAR generation
- KPOINTS generation
"""

import os
import re
import json
import shutil
import numpy as np
from pathlib import Path


# ─── VASP Output Parsing ───

def parse_outcar_energy(outcar_path):
    """Extract final energy(sigma->0) from OUTCAR."""
    energy = None
    with open(outcar_path, 'r') as f:
        for line in f:
            if 'energy  without entropy' in line:
                match = re.search(r'energy\(sigma->0\)\s*=\s*([-\d.]+)', line)
                if match:
                    energy = float(match.group(1))
    return energy


def parse_outcar_converged(outcar_path):
    """Check if VASP calculation converged (ionic + electronic)."""
    ionic_converged = False
    with open(outcar_path, 'r') as f:
        content = f.read()
        if 'reached required accuracy' in content:
            ionic_converged = True
    return ionic_converged


def check_vasp_done(calc_dir):
    """Check if a VASP calculation completed successfully."""
    outcar = os.path.join(calc_dir, 'OUTCAR')
    if not os.path.exists(outcar):
        return False, "OUTCAR not found"

    energy = parse_outcar_energy(outcar)
    if energy is None:
        return False, "Could not parse energy"

    converged = parse_outcar_converged(outcar)
    if not converged:
        return False, f"Not converged (E={energy:.4f} eV)"

    return True, f"Converged (E={energy:.6f} eV)"


# ─── POSCAR Utilities ───

def read_poscar_atoms(filepath):
    """Read atom count and species from POSCAR/CONTCAR."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    species = lines[5].split()
    counts = list(map(int, lines[6].split()))
    return species, counts


def get_surface_area(filepath):
    """Calculate surface area |a x b| from POSCAR/CONTCAR."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    scale = float(lines[1].strip())
    a = np.array([float(x) for x in lines[2].split()]) * scale
    b = np.array([float(x) for x in lines[3].split()]) * scale
    return np.linalg.norm(np.cross(a, b))


# ─── INCAR Generation ───

def generate_incar(template_path, output_path, overrides=None):
    """Copy INCAR template and apply overrides."""
    with open(template_path, 'r') as f:
        lines = f.readlines()

    params = {}
    for line in lines:
        line = line.strip()
        if line.startswith('#') or '=' not in line:
            continue
        key, val = line.split('=', 1)
        params[key.strip()] = val.strip()

    if overrides:
        params.update(overrides)

    with open(output_path, 'w') as f:
        for key, val in sorted(params.items()):
            f.write(f"{key} = {val}\n")


# ─── POTCAR Helper ───

def generate_potcar(species_list, potcar_dir, output_path='POTCAR',
                    suffix_map=None):
    """
    Concatenate POTCAR files for each species.

    suffix_map: dict mapping element -> POTCAR variant, e.g. {'C': 'C', 'Ca': 'Ca_sv'}
    Default suffixes follow Materials Project recommendations.
    """
    # MP-recommended POTCAR variants
    default_map = {
        'H': 'H', 'He': 'He', 'Li': 'Li_sv', 'Be': 'Be', 'B': 'B',
        'C': 'C', 'N': 'N', 'O': 'O', 'F': 'F', 'Na': 'Na_pv',
        'Mg': 'Mg', 'Al': 'Al', 'Si': 'Si', 'P': 'P', 'S': 'S',
        'Cl': 'Cl', 'K': 'K_sv', 'Ca': 'Ca_sv', 'Sc': 'Sc_sv',
        'Ti': 'Ti_sv', 'V': 'V_sv', 'Cr': 'Cr_pv', 'Mn': 'Mn_pv',
        'Fe': 'Fe', 'Co': 'Co', 'Ni': 'Ni', 'Cu': 'Cu', 'Zn': 'Zn',
        'Ga': 'Ga_d', 'Ge': 'Ge_d', 'Y': 'Y_sv', 'Zr': 'Zr_sv',
        'Nb': 'Nb_sv', 'Mo': 'Mo_sv', 'La': 'La', 'Ce': 'Ce',
        'Pr': 'Pr_3', 'Nd': 'Nd_3',
    }

    if suffix_map:
        default_map.update(suffix_map)

    with open(output_path, 'w') as fout:
        for sp in species_list:
            variant = default_map.get(sp, sp)
            potcar_path = os.path.join(potcar_dir, variant, 'POTCAR')
            if not os.path.exists(potcar_path):
                # Try without variant
                potcar_path = os.path.join(potcar_dir, sp, 'POTCAR')
            if not os.path.exists(potcar_path):
                raise FileNotFoundError(f"POTCAR not found for {sp} at {potcar_path}")
            with open(potcar_path, 'r') as fin:
                fout.write(fin.read())


# ─── Job Status ───

def scan_job_status(base_dir, pattern='*'):
    """Scan all calculation directories and report status."""
    import glob
    dirs = sorted(glob.glob(os.path.join(base_dir, pattern)))
    dirs = [d for d in dirs if os.path.isdir(d)]

    status = {'done': [], 'running': [], 'pending': [], 'failed': []}

    for d in dirs:
        outcar = os.path.join(d, 'OUTCAR')
        if not os.path.exists(outcar):
            if os.path.exists(os.path.join(d, 'POSCAR')):
                status['pending'].append(d)
            continue

        done, msg = check_vasp_done(d)
        if done:
            status['done'].append((d, msg))
        else:
            # Check if still running (OUTCAR exists but not converged)
            # Simple heuristic: if OUTCAR was modified recently
            import time
            mtime = os.path.getmtime(outcar)
            if time.time() - mtime < 3600:  # Modified within last hour
                status['running'].append((d, msg))
            else:
                status['failed'].append((d, msg))

    return status
