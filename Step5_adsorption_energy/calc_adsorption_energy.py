#!/usr/bin/env python3
"""
Calculate adsorption energy from VASP outputs, with optional ZPE correction.

Without ZPE:
  E_ads = E(slab+adsorbate) - E(clean_slab) - n_C * E_C_ref

With ZPE correction:
  E_ads = [E(slab+ads) + ZPE(slab+ads)] - [E(slab) + ZPE(slab)] - [n_C * E_C_ref + ZPE(C_ref)]
  Note: For slab, ZPE(slab) is often negligible and can be set to 0.

For carbon species:
  - Single C: E_ads = E(slab+C) - E(slab) - E(C_gas)  (C atom has no ZPE)
  - C_n chain/ring: includes ZPE from frequency calc
  - Graphene: E_ads = [E(slab+graphene) - E(slab) - E(graphene)] / Area  (no ZPE needed)

C reference energy (--c_energy):
  Use YOUR OWN graphite calculation for consistency with your POTCAR/INCAR.
  Example: graphite E_total = -39.787 eV / 4 atoms => --c_energy -9.9468
  Do NOT mix with Materials Project values unless using the same settings.

Usage:
    # Without ZPE (basic), using graphite per-atom energy as C reference
    python3 calc_adsorption_energy.py --batch --slab_energy -200.0 --c_energy -9.9468

    # With ZPE correction (expects freq/OUTCAR in each ads_* directory)
    python3 calc_adsorption_energy.py --batch --slab_energy -200.0 --c_energy -9.9468 --zpe

    # With ZPE + reference ZPE for gas-phase adsorbate
    python3 calc_adsorption_energy.py --batch --slab_energy -200.0 --c_energy -9.9468 \
        --zpe --zpe_ref 0.05 --zpe_slab 0.0

    # With ZPE from pre-computed JSON (from parse_frequency.py)
    python3 calc_adsorption_energy.py --batch --slab_energy -200.0 --c_energy -9.9468 \
        --zpe --zpe_json zpe_results.json

    # Graphene mode (per-area, no ZPE)
    python3 calc_adsorption_energy.py --batch --slab_energy -200.0 --c_energy -9.9468 --per_area
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


def parse_zpe_from_freq_outcar(outcar_path):
    """
    Parse vibrational frequencies from a frequency-calc OUTCAR and compute ZPE.
    ZPE = 0.5 * sum(real_freq_i)  in meV, converted to eV.
    """
    real_freqs_mev = []
    n_imag = 0
    with open(outcar_path, 'r') as f:
        for line in f:
            match_real = re.match(
                r'\s*\d+\s+f\s+=\s+[\d.]+\s+THz\s+[\d.]+\s+2PiTHz\s+'
                r'[\d.]+\s+cm-1\s+([\d.]+)\s+meV', line)
            if match_real:
                real_freqs_mev.append(float(match_real.group(1)))
                continue
            match_imag = re.match(r'\s*\d+\s+f/i\s*=', line)
            if match_imag:
                n_imag += 1

    if not real_freqs_mev and n_imag == 0:
        return None, 0  # No frequency data found

    zpe_ev = 0.5 * sum(real_freqs_mev) / 1000.0
    return zpe_ev, n_imag


def get_zpe(ads_dir, freq_subdir='freq', zpe_json=None):
    """
    Get ZPE for an adsorption directory.

    Priority:
    1. From zpe_json (pre-computed by parse_frequency.py)
    2. From freq/OUTCAR in the ads directory
    """
    # Try JSON first
    if zpe_json and os.path.exists(zpe_json):
        with open(zpe_json) as f:
            zpe_data = json.load(f)
        for entry in zpe_data:
            # Match by parent directory
            outcar_parent = os.path.dirname(os.path.dirname(entry.get('outcar', '')))
            if os.path.abspath(outcar_parent) == os.path.abspath(ads_dir):
                return entry['zpe_eV'], entry.get('n_imaginary', 0)

    # Try freq/OUTCAR
    freq_outcar = os.path.join(ads_dir, freq_subdir, 'OUTCAR')
    if os.path.exists(freq_outcar):
        return parse_zpe_from_freq_outcar(freq_outcar)

    return None, 0


def calc_adsorption_energy(e_ads_slab, e_clean_slab, n_C_adsorbate, e_C_ref,
                            area=None, per_area=False,
                            zpe_ads=None, zpe_slab=0.0, zpe_ref=0.0):
    """
    E_ads = [E(slab+ads) + ZPE(slab+ads)] - [E(slab) + ZPE(slab)]
            - [n_C * E_C_ref + ZPE(ref)]

    ZPE terms default to 0 if not provided (no correction).
    If per_area: normalize by surface area (for graphene).
    Returns eV (or eV/Ang^2 if per_area).
    """
    zpe_correction = 0.0
    if zpe_ads is not None:
        zpe_correction = zpe_ads - zpe_slab - zpe_ref

    e_ads = (e_ads_slab - e_clean_slab - n_C_adsorbate * e_C_ref) + zpe_correction

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
                        help="Reference energy per C atom (eV) from YOUR OWN graphite calc. "
                             "E.g. graphite -39.787 eV / 4 atoms = -9.9468 eV/atom.")
    parser.add_argument('--per_area', action='store_true',
                        help="Normalize by surface area (for graphene adsorption)")
    parser.add_argument('--zpe', action='store_true',
                        help="Include ZPE correction from frequency calculations")
    parser.add_argument('--zpe_json', default=None,
                        help="Pre-computed ZPE JSON file (from parse_frequency.py)")
    parser.add_argument('--freq_subdir', default='freq',
                        help="Subdirectory name for frequency calc (default: freq)")
    parser.add_argument('--zpe_slab', type=float, default=0.0,
                        help="ZPE of clean slab in eV (default: 0.0, usually negligible)")
    parser.add_argument('--zpe_ref', type=float, default=0.0,
                        help="ZPE of gas-phase reference per adsorbate in eV "
                             "(default: 0.0; single C atom has no ZPE)")
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
    if args.zpe:
        print(f"ZPE correction: ENABLED")
        print(f"  ZPE(slab) = {args.zpe_slab:.6f} eV")
        print(f"  ZPE(ref)  = {args.zpe_ref:.6f} eV")
        if args.zpe_json:
            print(f"  ZPE source: {args.zpe_json}")
        else:
            print(f"  ZPE source: {args.freq_subdir}/OUTCAR in each ads directory")

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

                # ZPE handling
                zpe_ads = None
                n_imag = 0
                is_graphene = 'graphene' in os.path.basename(ads_dir)
                if args.zpe and not is_graphene:
                    zpe_ads, n_imag = get_zpe(ads_dir, args.freq_subdir, args.zpe_json)
                    if zpe_ads is None:
                        print(f"  {ads_dir:<40}  WARNING: no ZPE data, using E without ZPE")

                e_ads = calc_adsorption_energy(
                    e_ads_slab, e_slab, n_C_ads, args.c_energy,
                    area, args.per_area,
                    zpe_ads=zpe_ads, zpe_slab=args.zpe_slab, zpe_ref=args.zpe_ref
                )

                result = {
                    'directory': ads_dir,
                    'e_ads_slab': e_ads_slab,
                    'n_C_adsorbate': n_C_ads,
                    'e_adsorption': e_ads,
                    'unit': 'eV/Ang^2' if args.per_area else 'eV',
                    'zpe_corrected': zpe_ads is not None,
                }
                if zpe_ads is not None:
                    result['zpe_ads_eV'] = zpe_ads
                    result['zpe_correction_eV'] = zpe_ads - args.zpe_slab - args.zpe_ref
                    result['n_imaginary_freq'] = n_imag
                if args.per_area:
                    result['area'] = area
                    result['e_ads_per_area_J_m2'] = e_ads * 16.0217663

                results.append(result)
                unit = 'eV/Ang^2' if args.per_area else 'eV'
                zpe_str = f"  ZPE={zpe_ads:.4f}" if zpe_ads is not None else ""
                imag_str = f"  [{n_imag} imag!]" if n_imag > 0 else ""
                print(f"  {ads_dir:<40}  n_C={n_C_ads:>3}  E_ads={e_ads:>10.4f} {unit}{zpe_str}{imag_str}")

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

        zpe_ads = None
        n_imag = 0
        is_graphene = 'graphene' in os.path.basename(args.ads_dir)
        if args.zpe and not is_graphene:
            zpe_ads, n_imag = get_zpe(args.ads_dir, args.freq_subdir, args.zpe_json)

        e_ads = calc_adsorption_energy(
            e_ads_slab, e_slab, n_C_ads, args.c_energy,
            area, args.per_area,
            zpe_ads=zpe_ads, zpe_slab=args.zpe_slab, zpe_ref=args.zpe_ref
        )

        print(f"\nDirectory: {args.ads_dir}")
        print(f"E(slab+ads) = {e_ads_slab:.6f} eV")
        print(f"N_C (adsorbate) = {n_C_ads}")
        if zpe_ads is not None:
            zpe_corr = zpe_ads - args.zpe_slab - args.zpe_ref
            print(f"ZPE(ads) = {zpe_ads:.6f} eV, ZPE correction = {zpe_corr:.6f} eV")
            if n_imag > 0:
                print(f"WARNING: {n_imag} imaginary frequency(ies) found!")
        unit = 'eV/Ang^2' if args.per_area else 'eV'
        print(f"E_adsorption = {e_ads:.6f} {unit}")

        result = {
            'directory': args.ads_dir,
            'e_ads_slab': e_ads_slab,
            'n_C_adsorbate': n_C_ads,
            'e_adsorption': e_ads,
            'unit': unit,
            'zpe_corrected': zpe_ads is not None,
        }
        if zpe_ads is not None:
            result['zpe_ads_eV'] = zpe_ads
            result['zpe_correction_eV'] = zpe_ads - args.zpe_slab - args.zpe_ref
        results.append(result)
    else:
        parser.error("Either --ads_dir or --batch must be specified")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
