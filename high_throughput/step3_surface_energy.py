#!/usr/bin/env python3
"""
Step 3: Calculate surface energies and identify the most stable surface.

After slab relaxations (Step 2) converge, this script:
1. Computes γ = (E_slab - N * E_bulk/atom) / (2A) for each surface
2. Ranks surfaces by stability
3. Records the most stable surface for Step 4 (adsorption)

Usage:
    python3 step3_surface_energy.py --slab_dir ./2_slabs --bulk_dir ./1_bulk
    python3 step3_surface_energy.py status --slab_dir ./2_slabs
"""

import os
import sys
import json
import argparse
import re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.job_manager import check_vasp_converged, parse_energy


def get_natoms(calc_dir):
    """Read atom count from POSCAR/CONTCAR."""
    for fname in ['CONTCAR', 'POSCAR']:
        fpath = os.path.join(calc_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                lines = f.readlines()
            counts = list(map(int, lines[6].split()))
            return sum(counts)
    return None


def get_surface_area(calc_dir):
    """Calculate surface area |a x b| from CONTCAR/POSCAR."""
    for fname in ['CONTCAR', 'POSCAR']:
        fpath = os.path.join(calc_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                lines = f.readlines()
            scale = float(lines[1].strip())
            a = np.array([float(x) for x in lines[2].split()]) * scale
            b = np.array([float(x) for x in lines[3].split()]) * scale
            return np.linalg.norm(np.cross(a, b))
    return None


def parse_surface_info(slab_dir):
    """Parse surface_info.log if present."""
    info = {}
    info_path = os.path.join(slab_dir, 'surface_info.log')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, val = line.split(':', 1)
                    info[key.strip()] = val.strip()
    return info


def calculate_surface_energies(slab_dir, bulk_dir):
    """Calculate surface energies for all compounds."""
    slab_manifest_path = os.path.join(slab_dir, 'manifest.json')
    bulk_manifest_path = os.path.join(bulk_dir, 'manifest.json')

    if not os.path.exists(slab_manifest_path):
        print(f"No slab manifest found at {slab_manifest_path}")
        return

    with open(slab_manifest_path) as f:
        slab_manifest = json.load(f)

    # Build bulk energy lookup
    bulk_energies = {}
    if os.path.exists(bulk_manifest_path):
        with open(bulk_manifest_path) as f:
            bulk_manifest = json.load(f)
        for entry in bulk_manifest:
            converged, _ = check_vasp_converged(entry['dir'])
            if converged:
                e = parse_energy(entry['dir'])
                n = get_natoms(entry['dir'])
                if e and n:
                    bulk_energies[entry['compound']] = e / n

    results = {}

    for entry in slab_manifest:
        compound = entry['compound']
        bulk_e_per_atom = bulk_energies.get(compound)

        if bulk_e_per_atom is None:
            print(f"\n  {compound}: no bulk energy, skipping")
            continue

        print(f"\n  {compound}: E_bulk/atom = {bulk_e_per_atom:.6f} eV")

        surface_results = []
        for slab_d in entry['slab_dirs']:
            slab_name = os.path.basename(slab_d)
            converged, msg = check_vasp_converged(slab_d)

            if not converged:
                print(f"    {slab_name}: not converged - {msg}")
                continue

            e_slab = parse_energy(slab_d)
            n_slab = get_natoms(slab_d)
            area = get_surface_area(slab_d)
            info = parse_surface_info(slab_d)

            if None in (e_slab, n_slab, area) or area < 1e-6:
                print(f"    {slab_name}: missing data")
                continue

            gamma_ev = (e_slab - n_slab * bulk_e_per_atom) / (2.0 * area)
            gamma_jm2 = gamma_ev * 16.0217663

            surface_results.append({
                'directory': slab_d,
                'name': slab_name,
                'hkl': info.get('HKL Index', 'Unknown'),
                'e_slab': e_slab,
                'n_slab': n_slab,
                'area': area,
                'gamma_eV_Ang2': gamma_ev,
                'gamma_J_m2': gamma_jm2,
            })
            print(f"    {slab_name:<25} γ = {gamma_jm2:>8.4f} J/m²")

        if surface_results:
            surface_results.sort(key=lambda x: x['gamma_J_m2'])
            best = surface_results[0]
            print(f"    → Most stable: {best['name']} (γ = {best['gamma_J_m2']:.4f} J/m²)")

            results[compound] = {
                'bulk_e_per_atom': bulk_e_per_atom,
                'surfaces': surface_results,
                'most_stable': best,
            }

    # Save results
    output_path = os.path.join(slab_dir, 'surface_energies.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'=' * 70}")
    print(f"Results saved: {output_path}")
    print(f"Compounds with surface energies: {len(results)}")

    # Summary table
    if results:
        print(f"\n{'Compound':<30}  {'Best Surface':<25}  {'γ (J/m²)':>10}")
        print("-" * 70)
        for compound, data in sorted(results.items()):
            best = data['most_stable']
            print(f"  {compound:<28}  {best['name']:<25}  {best['gamma_J_m2']:>10.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Step 3: Surface energy calculation")
    parser.add_argument('--slab_dir', default='./2_slabs')
    parser.add_argument('--bulk_dir', default='./1_bulk')

    args = parser.parse_args()
    calculate_surface_energies(args.slab_dir, args.bulk_dir)


if __name__ == "__main__":
    main()
