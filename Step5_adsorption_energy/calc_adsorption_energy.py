#!/usr/bin/env python3
"""
Calculate adsorption energy from VASP outputs, with optional ZPE correction.

Uses the TOTAL gas-phase energy of each adsorbate species as reference:
  E_ads = E(slab+adsorbate) - E(slab) - E(adsorbate_gas)

Each adsorbate type has its own reference energy (from gas-phase calculation):
  - single_C:  E_ref = E(C_atom)     = -1.3206 eV   (1 C)
  - C_chain:   E_ref = E(C3_chain)   = -19.0208 eV  (3 C)
  - C_ring:    E_ref = E(C6_ring)    = -42.8269 eV  (6 C)
  - graphene:  E_ref = E(graphene_freestanding) per area

Do NOT use graphite_per_atom * n_C - that gives the wrong reference state.

Usage:
    # Provide per-adsorbate reference energies
    python3 calc_adsorption_energy.py --batch --slab_dir . \\
        --e_single_C -1.32064672 \\
        --e_C_chain -19.02083563 \\
        --e_C_ring -42.82686827

    # Or use a reference JSON file
    python3 calc_adsorption_energy.py --batch --slab_dir . --ref_json references.json

    # With ZPE correction
    python3 calc_adsorption_energy.py --batch --slab_dir . \\
        --e_single_C -1.32064672 \\
        --e_C_chain -19.02083563 \\
        --e_C_ring -42.82686827 --zpe

    # Graphene mode (per-area normalization)
    python3 calc_adsorption_energy.py --batch --slab_dir . \\
        --e_graphene -39.78722576 --e_graphene_natoms 4 --per_area
"""

import os
import re
import glob
import json
import argparse
import numpy as np


# ─── Default reference energies (from your gas-phase calculations) ───

DEFAULT_REFERENCES = {
    'single_C': -1.32064672,       # 1 C atom in box
    'C_chain':  -19.02083563,      # C3 chain
    'C_ring':   -42.82686827,      # C6 ring
    'graphene_per_atom': None,     # Set from --e_graphene / --e_graphene_natoms
}


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


def detect_supercell_factor(ads_dir):
    """
    Detect supercell scale factor from config name or supercell_info.json.
    Returns scale factor (int, default 1).
    """
    parent = os.path.dirname(os.path.abspath(ads_dir))
    for f in glob.glob(os.path.join(parent, 'supercell_*_info.json')):
        with open(f) as fh:
            info = json.load(fh)
        return info.get('scale_factor', 1)

    dirname = os.path.basename(ads_dir)
    match = re.search(r'_(\d+)x(\d+)_', dirname)
    if match:
        return int(match.group(1)) * int(match.group(2))

    return 1


def detect_adsorbate_type(ads_dir):
    """
    Detect adsorbate type from directory name.

    Naming convention from generate_adsorption.py:
      ads_C_ontop_0         -> single_C
      ads_C_2x2_bridge_1    -> single_C
      ads_Cchain3v_ontop_0  -> C_chain
      ads_Cchain3h_bridge_1 -> C_chain
      ads_Cring6_hollow_0   -> C_ring
      ads_Cring6v_ontop_0   -> C_ring
      ads_graphene_*        -> graphene
    """
    dirname = os.path.basename(ads_dir)

    # Remove ads_ prefix
    name = dirname
    if name.startswith('ads_'):
        name = name[4:]

    if name.startswith('graphene'):
        return 'graphene'
    elif name.startswith('Cring'):
        return 'C_ring'
    elif name.startswith('Cchain'):
        return 'C_chain'
    elif name.startswith('C_') or name.startswith('C '):
        return 'single_C'

    # Fallback: try to detect from adsorbate atom count
    return 'unknown'


def count_adsorbate_C(ads_dir, slab_species, slab_counts, supercell_factor=1):
    """
    Determine how many C atoms are adsorbate atoms by comparing
    with the clean slab composition (scaled by supercell_factor).
    """
    ads_species, ads_counts = get_natoms_and_species(ads_dir)

    slab_composition = {}
    for sp, cnt in zip(slab_species, slab_counts):
        slab_composition[sp] = slab_composition.get(sp, 0) + cnt * supercell_factor

    ads_composition = {}
    for sp, cnt in zip(ads_species, ads_counts):
        ads_composition[sp] = ads_composition.get(sp, 0) + cnt

    n_C_ads = ads_composition.get('C', 0) - slab_composition.get('C', 0)
    return max(n_C_ads, 0)


def get_reference_energy(ads_type, n_C_ads, ref_energies):
    """
    Get the total gas-phase reference energy for this adsorbate.

    Returns E_ref (total energy of the gas-phase adsorbate, eV).
    """
    if ads_type == 'single_C':
        e_ref = ref_energies.get('single_C')
        if e_ref is None:
            raise ValueError("No reference energy for single_C. Use --e_single_C")
        return e_ref

    elif ads_type == 'C_chain':
        e_ref = ref_energies.get('C_chain')
        if e_ref is None:
            raise ValueError("No reference energy for C_chain. Use --e_C_chain")
        return e_ref

    elif ads_type == 'C_ring':
        e_ref = ref_energies.get('C_ring')
        if e_ref is None:
            raise ValueError("No reference energy for C_ring. Use --e_C_ring")
        return e_ref

    elif ads_type == 'graphene':
        e_per_atom = ref_energies.get('graphene_per_atom')
        if e_per_atom is None:
            raise ValueError("No reference energy for graphene. Use --e_graphene + --e_graphene_natoms")
        return n_C_ads * e_per_atom

    else:
        raise ValueError(f"Unknown adsorbate type '{ads_type}' for directory. "
                         f"Cannot determine reference energy.")


# ─── ZPE ───

def parse_zpe_from_freq_outcar(outcar_path):
    """Parse OUTCAR frequencies and compute ZPE."""
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
        return None, 0

    zpe_ev = 0.5 * sum(real_freqs_mev) / 1000.0
    return zpe_ev, n_imag


def get_zpe(ads_dir, freq_subdir='freq', zpe_json=None):
    """Get ZPE for an adsorption directory."""
    if zpe_json and os.path.exists(zpe_json):
        with open(zpe_json) as f:
            zpe_data = json.load(f)
        for entry in zpe_data:
            outcar_parent = os.path.dirname(os.path.dirname(entry.get('outcar', '')))
            if os.path.abspath(outcar_parent) == os.path.abspath(ads_dir):
                return entry['zpe_eV'], entry.get('n_imaginary', 0)

    freq_outcar = os.path.join(ads_dir, freq_subdir, 'OUTCAR')
    if os.path.exists(freq_outcar):
        return parse_zpe_from_freq_outcar(freq_outcar)

    return None, 0


def calc_adsorption_energy(e_ads_slab, e_clean_slab, e_ref_total,
                            area=None, per_area=False,
                            zpe_ads=None, zpe_slab=0.0, zpe_ref=0.0):
    """
    E_ads = [E(slab+ads) + ZPE(slab+ads)] - [E(slab) + ZPE(slab)] - [E_ref + ZPE(ref)]

    e_ref_total: TOTAL energy of the gas-phase adsorbate (not per-atom).
    """
    zpe_correction = 0.0
    if zpe_ads is not None:
        zpe_correction = zpe_ads - zpe_slab - zpe_ref

    e_ads = (e_ads_slab - e_clean_slab - e_ref_total) + zpe_correction

    if per_area and area:
        return e_ads / area
    return e_ads


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(
        description="Calculate adsorption energy with per-adsorbate reference energies."
    )
    parser.add_argument('--ads_dir', default=None, help="Single adsorption directory")
    parser.add_argument('--batch', action='store_true', help="Process all ads_* directories")
    parser.add_argument('--pattern', default='ads_*', help="Glob pattern for batch mode")
    parser.add_argument('--slab_energy', type=float, default=None,
                        help="Clean 1x1 slab total energy (eV). Use when OUTCAR is not available.")
    parser.add_argument('--slab_outcar', default=None,
                        help="Path to clean slab OUTCAR (reads energy automatically)")
    parser.add_argument('--slab_dir', default='.',
                        help="Clean slab directory containing CONTCAR/POSCAR for species "
                             "comparison, and optionally OUTCAR for energy. "
                             "Default: current directory.")

    # Per-adsorbate reference energies
    ref_group = parser.add_argument_group('Reference energies (gas-phase, total energy)')
    ref_group.add_argument('--e_single_C', type=float, default=-1.32064672,
                           help="Energy of single C atom in box (default: -1.3206 eV)")
    ref_group.add_argument('--e_C_chain', type=float, default=-19.02083563,
                           help="Energy of C3 chain (default: -19.0208 eV)")
    ref_group.add_argument('--e_C_ring', type=float, default=-42.82686827,
                           help="Energy of C6 ring (default: -42.8269 eV)")
    ref_group.add_argument('--e_graphene', type=float, default=None,
                           help="Total energy of freestanding graphene (eV)")
    ref_group.add_argument('--e_graphene_natoms', type=int, default=None,
                           help="Number of C atoms in graphene reference cell")
    ref_group.add_argument('--ref_json', default=None,
                           help="JSON file with reference energies (overrides CLI flags)")

    parser.add_argument('--per_area', action='store_true',
                        help="Normalize by surface area (for graphene adsorption)")
    parser.add_argument('--zpe', action='store_true',
                        help="Include ZPE correction from frequency calculations")
    parser.add_argument('--zpe_json', default=None,
                        help="Pre-computed ZPE JSON file (from parse_frequency.py)")
    parser.add_argument('--freq_subdir', default='freq',
                        help="Subdirectory name for frequency calc (default: freq)")
    parser.add_argument('--zpe_slab', type=float, default=0.0,
                        help="ZPE of clean slab in eV (default: 0.0)")
    parser.add_argument('--zpe_ref', type=float, default=0.0,
                        help="ZPE of gas-phase reference in eV (default: 0.0)")
    parser.add_argument('--output', default='adsorption_energies.json', help="Output JSON file")

    args = parser.parse_args()

    # Build reference energy dict
    ref_energies = {
        'single_C': args.e_single_C,
        'C_chain': args.e_C_chain,
        'C_ring': args.e_C_ring,
        'graphene_per_atom': None,
    }

    if args.e_graphene is not None and args.e_graphene_natoms is not None:
        ref_energies['graphene_per_atom'] = args.e_graphene / args.e_graphene_natoms

    # Override from JSON if provided
    if args.ref_json and os.path.exists(args.ref_json):
        with open(args.ref_json) as f:
            ref_json = json.load(f)
        for key in ['single_C', 'C_chain', 'C_ring', 'graphene_per_atom']:
            if key in ref_json:
                ref_energies[key] = ref_json[key]
        print(f"Loaded reference energies from {args.ref_json}")

    # Get clean slab energy
    if args.slab_outcar:
        e_slab = parse_outcar_energy(args.slab_outcar)
    elif args.slab_energy is not None:
        e_slab = args.slab_energy
    else:
        # Try to find OUTCAR: first in slab_dir, then search common locations
        slab_outcar = None
        search_paths = [
            os.path.join(args.slab_dir, 'OUTCAR'),
            os.path.join(args.slab_dir, 'OSZICAR'),
        ]
        for p in search_paths:
            if os.path.exists(p):
                slab_outcar = p
                break

        if slab_outcar:
            e_slab = get_energy(args.slab_dir)
        else:
            # OUTCAR not found - give a helpful error
            parser.error(
                f"No OUTCAR/OSZICAR found in '{args.slab_dir}'.\n"
                f"The clean slab energy is required. Provide it via one of:\n"
                f"  --slab_energy -XXX.XXX     (total energy in eV)\n"
                f"  --slab_outcar /path/to/slab/OUTCAR\n"
                f"  --slab_dir /path/to/slab/  (directory with OUTCAR + CONTCAR)"
            )

    slab_species, slab_counts = get_natoms_and_species(args.slab_dir)

    print(f"Clean 1x1 slab energy: {e_slab:.6f} eV")
    print(f"Slab composition (1x1): {dict(zip(slab_species, slab_counts))}")
    print(f"Reference energies:")
    print(f"  single_C (1 C):  {ref_energies['single_C']:.6f} eV")
    print(f"  C_chain  (3 C):  {ref_energies['C_chain']:.6f} eV")
    print(f"  C_ring   (6 C):  {ref_energies['C_ring']:.6f} eV")
    if ref_energies['graphene_per_atom'] is not None:
        print(f"  graphene:        {ref_energies['graphene_per_atom']:.6f} eV/atom")
    if args.zpe:
        print(f"ZPE correction: ENABLED")

    results = []

    def process_one(ads_dir):
        e_ads_slab = get_energy(ads_dir)

        sc_factor = detect_supercell_factor(ads_dir)
        e_slab_scaled = e_slab * sc_factor

        n_C_ads = count_adsorbate_C(ads_dir, slab_species, slab_counts,
                                     supercell_factor=sc_factor)

        ads_type = detect_adsorbate_type(ads_dir)
        e_ref = get_reference_energy(ads_type, n_C_ads, ref_energies)

        area = get_surface_area(ads_dir) if args.per_area else None

        # ZPE
        zpe_ads = None
        n_imag = 0
        if args.zpe and ads_type != 'graphene':
            zpe_ads, n_imag = get_zpe(ads_dir, args.freq_subdir, args.zpe_json)
            if zpe_ads is None:
                print(f"  {ads_dir:<40}  WARNING: no ZPE data, using E without ZPE")

        e_ads = calc_adsorption_energy(
            e_ads_slab, e_slab_scaled, e_ref,
            area, args.per_area,
            zpe_ads=zpe_ads, zpe_slab=args.zpe_slab, zpe_ref=args.zpe_ref
        )

        result = {
            'directory': ads_dir,
            'adsorbate_type': ads_type,
            'e_ads_slab': e_ads_slab,
            'n_C_adsorbate': n_C_ads,
            'e_ref_gas': e_ref,
            'e_adsorption': e_ads,
            'unit': 'eV/Ang^2' if args.per_area else 'eV',
            'zpe_corrected': zpe_ads is not None,
            'supercell_factor': sc_factor,
        }
        if sc_factor > 1:
            result['e_slab_scaled'] = e_slab_scaled
        if zpe_ads is not None:
            result['zpe_ads_eV'] = zpe_ads
            result['zpe_correction_eV'] = zpe_ads - args.zpe_slab - args.zpe_ref
            result['n_imaginary_freq'] = n_imag
        if args.per_area and area:
            result['area'] = area
            result['e_ads_per_area_J_m2'] = e_ads * 16.0217663

        return result, ads_type, n_C_ads, e_ref, e_ads, sc_factor, zpe_ads, n_imag

    if args.batch:
        ads_dirs = sorted(glob.glob(args.pattern))
        ads_dirs = [d for d in ads_dirs if os.path.isdir(d)]

        if not ads_dirs:
            print(f"No directories matching: {args.pattern}")
            return

        print(f"\nProcessing {len(ads_dirs)} adsorption directories...")
        print("=" * 90)

        for ads_dir in ads_dirs:
            try:
                result, ads_type, n_C, e_ref, e_ads, sc_factor, zpe_ads, n_imag = \
                    process_one(ads_dir)
                results.append(result)

                unit = 'eV/Ang^2' if args.per_area else 'eV'
                sc_str = f" [{sc_factor}x]" if sc_factor > 1 else ""
                zpe_str = f" ZPE={zpe_ads:.4f}" if zpe_ads is not None else ""
                imag_str = f" [{n_imag}imag!]" if n_imag > 0 else ""
                print(f"  {ads_dir:<35} {ads_type:<10} ref={e_ref:>10.4f}  "
                      f"E_ads={e_ads:>10.4f} {unit}{sc_str}{zpe_str}{imag_str}")

            except Exception as e:
                print(f"  {ads_dir:<35} ERROR: {e}")
                continue

        # Summary
        if results:
            results.sort(key=lambda x: x['e_adsorption'])
            print(f"\n{'=' * 90}")
            print("ADSORPTION ENERGY SUMMARY (sorted, most negative = most stable)")
            print("=" * 90)
            unit = 'eV/Ang^2' if args.per_area else 'eV'
            print(f"  {'Config':<35}  {'Type':<10}  {'n_C':>4}  {'E_ref':>10}  {'E_ads':>10}  {unit}")
            print("  " + "-" * 80)
            for r in results:
                print(f"  {r['directory']:<35}  {r['adsorbate_type']:<10}  "
                      f"{r['n_C_adsorbate']:>4}  {r['e_ref_gas']:>10.4f}  "
                      f"{r['e_adsorption']:>10.4f}  {unit}")

            best = results[0]
            print(f"\n  Most stable: {best['directory']}")
            print(f"  Type: {best['adsorbate_type']}, E_ads = {best['e_adsorption']:.4f} {unit}")

    elif args.ads_dir:
        result, ads_type, n_C, e_ref, e_ads, sc_factor, zpe_ads, n_imag = \
            process_one(args.ads_dir)
        results.append(result)

        print(f"\nDirectory: {args.ads_dir}")
        print(f"Adsorbate type: {ads_type} ({n_C} C atoms)")
        if sc_factor > 1:
            print(f"Supercell factor: {sc_factor}x")
        print(f"E(slab+ads) = {result['e_ads_slab']:.6f} eV")
        print(f"E(ref_gas)  = {e_ref:.6f} eV")
        if zpe_ads is not None:
            zpe_corr = zpe_ads - args.zpe_slab - args.zpe_ref
            print(f"ZPE(ads) = {zpe_ads:.6f} eV, correction = {zpe_corr:.6f} eV")
            if n_imag > 0:
                print(f"WARNING: {n_imag} imaginary frequency(ies)!")
        unit = 'eV/Ang^2' if args.per_area else 'eV'
        print(f"E_adsorption = {e_ads:.6f} {unit}")
    else:
        parser.error("Either --ads_dir or --batch must be specified")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
