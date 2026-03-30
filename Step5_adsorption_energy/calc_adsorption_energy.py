#!/usr/bin/env python3
"""
Calculate adsorption energy from VASP outputs.

E_ads = E(slab+adsorbate) - E(clean_slab) - E(adsorbate_gas)

For carbon species:
  - Single C: E_ads = E(slab+C) - E(slab) - E(C_gas)
  - C_n chain/ring: E_ads = E(slab+C_n) - E(slab) - n * E(C_gas)
    or alternatively:  E_ads = E(slab+C_n) - E(slab) - E(C_n_gas)
  - Graphene: E_ads = [E(slab+graphene) - E(slab) - E(graphene_freestanding)] / Area

Usage:
    # Single configuration
    python3 calc_adsorption_energy.py --ads_dir ads_C_ontop_0 --slab_energy -200.0 --c_energy -1.36

    # Batch: all ads_* directories under a surface
    python3 calc_adsorption_energy.py --batch --slab_energy -200.0 --c_energy -1.36

    # Using OUTCAR paths directly
    python3 calc_adsorption_energy.py --batch --slab_outcar surf_001/OUTCAR --c_energy -1.36

    # Graphene mode (per-area normalization)
    python3 calc_adsorption_energy.py --batch --slab_energy -200.0 --c_energy -1.36 --per_area
"""

import os
import re
import glob
import json
import argparse
import numpy as np


def parse_outcar_energy(outcar_path):
    """Extract final energy(sigma->0) from OUTCAR."""
    energy = None
    with open(outcar_path, 'r') as f:
        for line in f:
            if 'energy  without entropy' in line:
                match = re.search(r'energy\(sigma->0\)\s*=\s*([-\d.]+)', line)
                if match:
                    energy = float(match.group(1))
    if energy is None:
        raise ValueError(f"Could not parse energy from {outcar_path}")
    return energy


def parse_oszicar_energy(oszicar_path):
    """Extract final E0 from OSZICAR."""
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
    """Get energy from OUTCAR or OSZICAR."""
    outcar = os.path.join(calc_dir, 'OUTCAR')
    oszicar = os.path.join(calc_dir, 'OSZICAR')
    if os.path.exists(outcar):
        return parse_outcar_energy(outcar)
    elif os.path.exists(oszicar):
        return parse_oszicar_energy(oszicar)
    else:
        raise FileNotFoundError(f"No OUTCAR/OSZICAR in {calc_dir}")


def get_natoms_and_species(calc_dir):
    """Get atom counts from POSCAR/CONTCAR."""
    for fname in ['CONTCAR', 'POSCAR']:
        fpath = os.path.join(calc_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                lines = f.readlines()
            species = lines[5].split()
            counts = list(map(int, lines[6].split()))
            return species, counts
    raise FileNotFoundError(f"No POSCAR/CONTCAR in {calc_dir}")


def get_surface_area(calc_dir):
    """Calculate surface area |a x b|."""
    for fname in ['CONTCAR', 'POSCAR']:
        fpath = os.path.join(calc_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                lines = f.readlines()
            scale = float(lines[1].strip())
            a = np.array([float(x) for x in lines[2].split()]) * scale
            b = np.array([float(x) for x in lines[3].split()]) * scale
            return np.linalg.norm(np.cross(a, b))
    raise FileNotFoundError(f"No POSCAR/CONTCAR in {calc_dir}")


def count_adsorbate_C(ads_dir, slab_species, slab_counts):
    """
    Determine how many C atoms are adsorbate atoms by comparing
    with the clean slab composition.
    """
    ads_species, ads_counts = get_natoms_and_species(ads_dir)

    slab_composition = {}
    for sp, cnt in zip(slab_species, slab_counts):
        slab_composition[sp] = slab_composition.get(sp, 0) + cnt

    ads_composition = {}
    for sp, cnt in zip(ads_species, ads_counts):
        ads_composition[sp] = ads_composition.get(sp, 0) + cnt

    # The difference in C count is the number of adsorbate C atoms
    n_C_ads = ads_composition.get('C', 0) - slab_composition.get('C', 0)
    return max(n_C_ads, 0)


def calc_adsorption_energy(e_ads_slab, e_clean_slab, n_C_adsorbate, e_C_ref,
                            area=None, per_area=False):
    """
    E_ads = E(slab+adsorbate) - E(clean_slab) - n_C * E_C_ref

    If per_area: normalize by surface area (for graphene-like coverage).
    Returns eV (or eV/Ang^2 if per_area).
    """
    e_ads = e_ads_slab - e_clean_slab - n_C_adsorbate * e_C_ref

    if per_area and area:
        return e_ads / area
    return e_ads


def main():
    parser = argparse.ArgumentParser(description="Calculate adsorption energy from VASP outputs.")
    parser.add_argument('--ads_dir', default=None, help="Single adsorption directory")
    parser.add_argument('--batch', action='store_true', help="Process all ads_* directories")
    parser.add_argument('--pattern', default='ads_*', help="Glob pattern for batch mode")
    parser.add_argument('--slab_energy', type=float, default=None, help="Clean slab energy (eV)")
    parser.add_argument('--slab_outcar', default=None, help="Path to clean slab OUTCAR")
    parser.add_argument('--slab_dir', default='.', help="Clean slab directory (for species comparison)")
    parser.add_argument('--c_energy', type=float, required=True,
                        help="Reference energy per C atom (eV). "
                             "Typically from isolated C atom or graphite per atom.")
    parser.add_argument('--per_area', action='store_true',
                        help="Normalize by surface area (for graphene adsorption)")
    parser.add_argument('--output', default='adsorption_energies.json', help="Output JSON file")

    args = parser.parse_args()

    # Get clean slab energy
    if args.slab_outcar:
        e_slab = parse_outcar_energy(args.slab_outcar)
    elif args.slab_energy is not None:
        e_slab = args.slab_energy
    else:
        # Try to find OUTCAR in slab_dir
        e_slab = get_energy(args.slab_dir)

    # Get slab composition for adsorbate counting
    slab_species, slab_counts = get_natoms_and_species(args.slab_dir)

    print(f"Clean slab energy: {e_slab:.6f} eV")
    print(f"C reference energy: {args.c_energy:.6f} eV/atom")
    print(f"Slab composition: {dict(zip(slab_species, slab_counts))}")

    results = []

    if args.batch:
        ads_dirs = sorted(glob.glob(args.pattern))
        ads_dirs = [d for d in ads_dirs if os.path.isdir(d)]

        if not ads_dirs:
            print(f"No directories matching: {args.pattern}")
            return

        print(f"\nProcessing {len(ads_dirs)} adsorption directories...")
        print("=" * 80)

        for ads_dir in ads_dirs:
            try:
                e_ads_slab = get_energy(ads_dir)
                n_C_ads = count_adsorbate_C(ads_dir, slab_species, slab_counts)
                area = get_surface_area(ads_dir) if args.per_area else None

                e_ads = calc_adsorption_energy(e_ads_slab, e_slab, n_C_ads,
                                                args.c_energy, area, args.per_area)

                result = {
                    'directory': ads_dir,
                    'e_ads_slab': e_ads_slab,
                    'n_C_adsorbate': n_C_ads,
                    'e_adsorption': e_ads,
                    'unit': 'eV/Ang^2' if args.per_area else 'eV',
                }
                if args.per_area:
                    result['area'] = area
                    result['e_ads_per_area_J_m2'] = e_ads * 16.0217663

                results.append(result)
                unit = 'eV/Ang^2' if args.per_area else 'eV'
                print(f"  {ads_dir:<40}  n_C={n_C_ads:>3}  E_ads={e_ads:>10.4f} {unit}")

            except Exception as e:
                print(f"  {ads_dir:<40}  ERROR: {e}")
                continue

        # Summary
        if results:
            results.sort(key=lambda x: x['e_adsorption'])
            print(f"\n{'=' * 80}")
            print("ADSORPTION ENERGY SUMMARY (sorted, most negative = most stable)")
            print("=" * 80)
            unit = 'eV/Ang^2' if args.per_area else 'eV'
            print(f"  {'Config':<40}  {'n_C':>4}  {'E_ads':>12}  {'Unit'}")
            print("  " + "-" * 65)
            for r in results:
                print(f"  {r['directory']:<40}  {r['n_C_adsorbate']:>4}  "
                      f"{r['e_adsorption']:>12.4f}  {unit}")

            best = results[0]
            print(f"\n  Most stable: {best['directory']}")
            print(f"  E_ads = {best['e_adsorption']:.4f} {unit}")

    elif args.ads_dir:
        e_ads_slab = get_energy(args.ads_dir)
        n_C_ads = count_adsorbate_C(args.ads_dir, slab_species, slab_counts)
        area = get_surface_area(args.ads_dir) if args.per_area else None

        e_ads = calc_adsorption_energy(e_ads_slab, e_slab, n_C_ads,
                                        args.c_energy, area, args.per_area)

        print(f"\nDirectory: {args.ads_dir}")
        print(f"E(slab+ads) = {e_ads_slab:.6f} eV")
        print(f"N_C (adsorbate) = {n_C_ads}")
        unit = 'eV/Ang^2' if args.per_area else 'eV'
        print(f"E_adsorption = {e_ads:.6f} {unit}")

        results.append({
            'directory': args.ads_dir,
            'e_ads_slab': e_ads_slab,
            'n_C_adsorbate': n_C_ads,
            'e_adsorption': e_ads,
            'unit': unit,
        })
    else:
        parser.error("Either --ads_dir or --batch must be specified")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
