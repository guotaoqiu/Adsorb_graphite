#!/usr/bin/env python3
"""
Calculate surface energy from VASP slab calculations.

Surface energy: gamma = (E_slab - N * E_bulk_per_atom) / (2 * A)

Usage:
    # Single surface
    python3 calc_surface_energy.py --slab_dir surf_001_term_0 --bulk_energy -123.456 --bulk_natoms 8

    # Batch mode: process all surf_* directories
    python3 calc_surface_energy.py --batch --bulk_energy -123.456 --bulk_natoms 8

    # Using bulk OUTCAR directly
    python3 calc_surface_energy.py --batch --bulk_outcar /path/to/bulk/OUTCAR --bulk_natoms 8

    # With convergence test (multiple thicknesses)
    python3 calc_surface_energy.py --convergence --bulk_energy -123.456 --bulk_natoms 8
"""

import os
import re
import glob
import json
import argparse
import numpy as np


def parse_outcar_energy(outcar_path):
    """Extract the final total energy (sigma->0) from OUTCAR."""
    energy = None
    with open(outcar_path, 'r') as f:
        for line in f:
            if 'energy  without entropy' in line:
                # "energy  without entropy=     -123.456  energy(sigma->0) =     -123.457"
                match = re.search(r'energy\(sigma->0\)\s*=\s*([-\d.]+)', line)
                if match:
                    energy = float(match.group(1))
    if energy is None:
        raise ValueError(f"Could not parse energy from {outcar_path}")
    return energy


def parse_oszicar_energy(oszicar_path):
    """Extract the final energy from OSZICAR (last line with E0)."""
    energy = None
    with open(oszicar_path, 'r') as f:
        for line in f:
            if 'E0=' in line:
                match = re.search(r'E0=\s*([-\d.Ee+]+)', line)
                if match:
                    energy = float(match.group(1))
    if energy is None:
        raise ValueError(f"Could not parse energy from {oszicar_path}")
    return energy


def get_energy(calc_dir):
    """Get final energy from a VASP calculation directory (OUTCAR preferred, OSZICAR fallback)."""
    outcar = os.path.join(calc_dir, 'OUTCAR')
    oszicar = os.path.join(calc_dir, 'OSZICAR')

    if os.path.exists(outcar):
        return parse_outcar_energy(outcar)
    elif os.path.exists(oszicar):
        return parse_oszicar_energy(oszicar)
    else:
        raise FileNotFoundError(f"No OUTCAR or OSZICAR found in {calc_dir}")


def get_slab_natoms(calc_dir):
    """Get number of atoms from POSCAR/CONTCAR in the slab directory."""
    for fname in ['CONTCAR', 'POSCAR']:
        fpath = os.path.join(calc_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                lines = f.readlines()
            counts = list(map(int, lines[6].split()))
            return sum(counts)
    raise FileNotFoundError(f"No POSCAR/CONTCAR found in {calc_dir}")


def get_surface_area(calc_dir):
    """Calculate surface area from the slab's lattice vectors (|a x b|)."""
    for fname in ['CONTCAR', 'POSCAR']:
        fpath = os.path.join(calc_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                lines = f.readlines()
            scale = float(lines[1].strip())
            a = np.array([float(x) for x in lines[2].split()]) * scale
            b = np.array([float(x) for x in lines[3].split()]) * scale
            return np.linalg.norm(np.cross(a, b))
    raise FileNotFoundError(f"No POSCAR/CONTCAR found in {calc_dir}")


def parse_surface_info(calc_dir):
    """Parse surface_info.log if present."""
    info_path = os.path.join(calc_dir, 'surface_info.log')
    info = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, val = line.split(':', 1)
                    info[key.strip()] = val.strip()
    return info


def calc_surface_energy(e_slab, n_slab, e_bulk_per_atom, area):
    """
    gamma = (E_slab - N_slab * E_bulk_per_atom) / (2 * A)

    Returns surface energy in eV/Ang^2 and J/m^2.
    """
    gamma_ev_ang2 = (e_slab - n_slab * e_bulk_per_atom) / (2.0 * area)
    gamma_j_m2 = gamma_ev_ang2 * 16.0217663  # 1 eV/Ang^2 = 16.0217663 J/m^2
    return gamma_ev_ang2, gamma_j_m2


def process_surface(slab_dir, e_bulk_per_atom, verbose=True):
    """Process a single surface directory."""
    e_slab = get_energy(slab_dir)
    n_slab = get_slab_natoms(slab_dir)
    area = get_surface_area(slab_dir)
    info = parse_surface_info(slab_dir)

    gamma_ev, gamma_j = calc_surface_energy(e_slab, n_slab, e_bulk_per_atom, area)

    result = {
        'directory': slab_dir,
        'hkl': info.get('HKL Index', 'Unknown'),
        'tasker': info.get('Tasker Type', 'Unknown'),
        'e_slab': e_slab,
        'n_slab': n_slab,
        'area': area,
        'gamma_eV_Ang2': gamma_ev,
        'gamma_J_m2': gamma_j,
    }

    if verbose:
        print(f"\n  Directory: {slab_dir}")
        print(f"  HKL: {result['hkl']}")
        print(f"  E_slab = {e_slab:.6f} eV  ({n_slab} atoms)")
        print(f"  Area = {area:.4f} Ang^2")
        print(f"  Surface energy = {gamma_ev:.6f} eV/Ang^2 = {gamma_j:.4f} J/m^2")

    return result


def main():
    parser = argparse.ArgumentParser(description="Calculate surface energy from VASP outputs.")
    parser.add_argument('--slab_dir', default=None, help="Single slab directory")
    parser.add_argument('--batch', action='store_true', help="Process all surf_* directories")
    parser.add_argument('--pattern', default='surf_*', help="Glob pattern for batch mode (default: surf_*)")
    parser.add_argument('--bulk_energy', type=float, default=None, help="Total bulk energy (eV)")
    parser.add_argument('--bulk_natoms', type=int, default=None, help="Number of atoms in bulk cell")
    parser.add_argument('--bulk_outcar', default=None, help="Path to bulk OUTCAR (alternative to --bulk_energy)")
    parser.add_argument('--output', default='surface_energies.json', help="Output JSON file")
    parser.add_argument('--convergence', action='store_true',
                        help="Convergence test mode: expect subdirs with different thicknesses")

    args = parser.parse_args()

    # Get bulk energy per atom
    if args.bulk_outcar:
        e_bulk_total = parse_outcar_energy(args.bulk_outcar)
        if args.bulk_natoms is None:
            raise ValueError("--bulk_natoms is required when using --bulk_outcar")
        e_bulk_per_atom = e_bulk_total / args.bulk_natoms
        print(f"Bulk energy from OUTCAR: {e_bulk_total:.6f} eV / {args.bulk_natoms} atoms = {e_bulk_per_atom:.6f} eV/atom")
    elif args.bulk_energy is not None:
        if args.bulk_natoms is None:
            raise ValueError("--bulk_natoms is required when using --bulk_energy")
        e_bulk_per_atom = args.bulk_energy / args.bulk_natoms
        print(f"Bulk energy: {args.bulk_energy:.6f} eV / {args.bulk_natoms} atoms = {e_bulk_per_atom:.6f} eV/atom")
    else:
        parser.error("Either --bulk_energy or --bulk_outcar must be provided")

    results = []

    if args.batch:
        slab_dirs = sorted(glob.glob(args.pattern))
        slab_dirs = [d for d in slab_dirs if os.path.isdir(d)]

        if not slab_dirs:
            print(f"No directories found matching pattern: {args.pattern}")
            return

        print(f"\nProcessing {len(slab_dirs)} surface directories...")
        print("=" * 70)

        for slab_dir in slab_dirs:
            try:
                result = process_surface(slab_dir, e_bulk_per_atom)
                results.append(result)
            except Exception as e:
                print(f"\n  ERROR in {slab_dir}: {e}")
                continue

        # Summary table sorted by surface energy
        results.sort(key=lambda x: x['gamma_J_m2'])

        print("\n" + "=" * 70)
        print("SURFACE ENERGY SUMMARY (sorted by gamma)")
        print("=" * 70)
        print(f"  {'Directory':<30}  {'HKL':<10}  {'Tasker':<8}  {'Area':>8}  {'gamma (J/m2)':>12}")
        print("  " + "-" * 72)
        for r in results:
            print(f"  {r['directory']:<30}  {r['hkl']:<10}  {r['tasker']:<8}  "
                  f"{r['area']:>7.2f}  {r['gamma_J_m2']:>12.4f}")

        if results:
            best = results[0]
            print(f"\n  Most stable surface: {best['directory']}")
            print(f"  HKL = {best['hkl']}, gamma = {best['gamma_J_m2']:.4f} J/m^2")

    elif args.slab_dir:
        result = process_surface(args.slab_dir, e_bulk_per_atom)
        results.append(result)
    else:
        parser.error("Either --slab_dir or --batch must be specified")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
