#!/usr/bin/env python3
"""
Compare DFT (VASP) and MACE energies for converged calculations.

Reads energies from both DFT directories and MACE directories,
matches by structure name, and reports statistics.

Usage:
    # Compare slab energies
    python3 compare_dft_mace.py \
        --dft_dir ./2_slabs \
        --mace_dir ./2_slabs_mace \
        --pattern "*/*/surf_*"

    # Compare adsorption energies
    python3 compare_dft_mace.py \
        --dft_dir ./3_adsorption \
        --mace_dir ./3_adsorption_mace \
        --pattern "*/ads_*"

    # Compare with per-atom normalization
    python3 compare_dft_mace.py \
        --dft_dir ./2_slabs \
        --mace_dir ./2_slabs_mace \
        --pattern "*/*/surf_*" \
        --per_atom
"""

import os
import sys
import json
import glob
import argparse
import re
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def parse_energy_from_dir(calc_dir):
    """Parse energy from energy.json (MACE) or OUTCAR (VASP)."""
    # energy.json (MACE)
    ej = os.path.join(calc_dir, 'energy.json')
    if os.path.exists(ej):
        with open(ej) as f:
            info = json.load(f)
        if info.get('converged', False):
            return info.get('energy_sigma0'), 'MACE'

    # OUTCAR (VASP)
    outcar = os.path.join(calc_dir, 'OUTCAR')
    if os.path.exists(outcar):
        energy = None
        converged = False
        with open(outcar, 'r') as f:
            for line in f:
                if 'energy(sigma->0)' in line:
                    match = re.search(r'energy\(sigma->0\)\s*=\s*([-\d.]+)', line)
                    if match:
                        energy = float(match.group(1))
                if 'reached required accuracy' in line:
                    converged = True
        if energy and converged:
            return energy, 'VASP'

    return None, None


def get_natoms(calc_dir):
    """Get atom count from CONTCAR or POSCAR."""
    for fname in ['CONTCAR', 'POSCAR']:
        fpath = os.path.join(calc_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                lines = f.readlines()
            try:
                counts = list(map(int, lines[6].split()))
                return sum(counts)
            except (ValueError, IndexError):
                continue
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Compare DFT and MACE energies."
    )
    parser.add_argument('--dft_dir', required=True, help="DFT results directory")
    parser.add_argument('--mace_dir', required=True, help="MACE results directory")
    parser.add_argument('--pattern', default='*/*/surf_*',
                        help="Glob pattern relative to each root dir")
    parser.add_argument('--per_atom', action='store_true',
                        help="Normalize energy differences by number of atoms")
    parser.add_argument('--output', default=None,
                        help="Save comparison to JSON file")

    args = parser.parse_args()

    # Find matching directories
    dft_dirs = sorted(glob.glob(os.path.join(args.dft_dir, args.pattern)))
    mace_dirs = sorted(glob.glob(os.path.join(args.mace_dir, args.pattern)))

    # Build lookup by relative path
    dft_lookup = {}
    for d in dft_dirs:
        rel = os.path.relpath(d, args.dft_dir)
        dft_lookup[rel] = d

    mace_lookup = {}
    for d in mace_dirs:
        rel = os.path.relpath(d, args.mace_dir)
        mace_lookup[rel] = d

    common = sorted(set(dft_lookup.keys()) & set(mace_lookup.keys()))

    if not common:
        print(f"No matching directories found.")
        print(f"  DFT:  {len(dft_dirs)} dirs under {args.dft_dir}/{args.pattern}")
        print(f"  MACE: {len(mace_dirs)} dirs under {args.mace_dir}/{args.pattern}")
        return

    print(f"{'=' * 80}")
    print(f"DFT vs MACE Energy Comparison")
    print(f"{'=' * 80}")
    print(f"  DFT root:  {args.dft_dir}")
    print(f"  MACE root: {args.mace_dir}")
    print(f"  Matched:   {len(common)} directories")
    print()

    # Compare
    results = []
    header = f"  {'Structure':<45} {'E_DFT':>12} {'E_MACE':>12} {'diff':>10}"
    if args.per_atom:
        header += f" {'diff/atom':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for rel in common:
        dft_d = dft_lookup[rel]
        mace_d = mace_lookup[rel]

        e_dft, src_dft = parse_energy_from_dir(dft_d)
        e_mace, src_mace = parse_energy_from_dir(mace_d)

        if e_dft is None or e_mace is None:
            continue

        diff = e_mace - e_dft
        natoms = get_natoms(dft_d)

        row = f"  {rel:<45} {e_dft:>12.4f} {e_mace:>12.4f} {diff:>+10.4f}"
        if args.per_atom and natoms:
            diff_pa = diff / natoms
            row += f" {diff_pa:>+10.4f}"
        print(row)

        entry = {
            'structure': rel,
            'e_dft': e_dft,
            'e_mace': e_mace,
            'diff_eV': diff,
            'n_atoms': natoms,
        }
        if natoms:
            entry['diff_per_atom_eV'] = diff / natoms
        results.append(entry)

    if not results:
        print("\n  No converged pairs found for comparison.")
        return

    # Statistics
    diffs = [r['diff_eV'] for r in results]
    diffs_pa = [r['diff_per_atom_eV'] for r in results if 'diff_per_atom_eV' in r]

    print(f"\n{'=' * 80}")
    print(f"STATISTICS ({len(results)} pairs)")
    print(f"{'=' * 80}")
    print(f"  Total energy difference (MACE - DFT):")
    print(f"    Mean:    {np.mean(diffs):>+10.4f} eV")
    print(f"    Std:     {np.std(diffs):>10.4f} eV")
    print(f"    MAE:     {np.mean(np.abs(diffs)):>10.4f} eV")
    print(f"    Max |diff|: {np.max(np.abs(diffs)):>10.4f} eV")

    if diffs_pa:
        print(f"\n  Per-atom difference:")
        print(f"    Mean:    {np.mean(diffs_pa):>+10.4f} eV/atom")
        print(f"    Std:     {np.std(diffs_pa):>10.4f} eV/atom")
        print(f"    MAE:     {np.mean(np.abs(diffs_pa)):>10.4f} eV/atom")
        print(f"    Max:     {np.max(np.abs(diffs_pa)):>10.4f} eV/atom")

    # Check if ranking is preserved (most important for screening)
    if len(results) >= 2:
        dft_energies = [r['e_dft'] for r in results]
        mace_energies = [r['e_mace'] for r in results]
        dft_rank = np.argsort(dft_energies)
        mace_rank = np.argsort(mace_energies)

        # Spearman rank correlation
        from scipy.stats import spearmanr
        try:
            corr, pval = spearmanr(dft_energies, mace_energies)
            print(f"\n  Ranking correlation:")
            print(f"    Spearman rho:  {corr:.4f}")
            print(f"    p-value:       {pval:.2e}")
            if corr > 0.95:
                print(f"    Verdict:       EXCELLENT — ranking is preserved")
            elif corr > 0.85:
                print(f"    Verdict:       GOOD — ranking mostly preserved")
            elif corr > 0.70:
                print(f"    Verdict:       FAIR — some reordering")
            else:
                print(f"    Verdict:       POOR — significant reordering")
        except ImportError:
            print(f"\n  (install scipy for ranking correlation analysis)")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'results': results,
                'stats': {
                    'n_pairs': len(results),
                    'mean_diff_eV': float(np.mean(diffs)),
                    'std_diff_eV': float(np.std(diffs)),
                    'mae_diff_eV': float(np.mean(np.abs(diffs))),
                    'max_abs_diff_eV': float(np.max(np.abs(diffs))),
                }
            }, f, indent=2)
        print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
