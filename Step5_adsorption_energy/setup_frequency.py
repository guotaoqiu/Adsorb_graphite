#!/usr/bin/env python3
"""
Set up VASP frequency calculations for adsorption configurations.

Workflow:
  1. Find the lowest-energy adsorption structure (from static calc results)
  2. Copy CONTCAR -> POSCAR for frequency calculation
  3. Generate INCAR for finite-difference phonon (IBRION=5 or IBRION=6)
  4. Set selective dynamics: fix bulk slab atoms, release only adsorbate atoms
     (and optionally the top surface layer)

The frequency calc gives vibrational modes -> ZPE = 0.5 * sum(hbar*omega_i)
Only REAL frequencies contribute to ZPE (imaginary = transition state).

Usage:
    # Set up freq calc for a single adsorption directory
    python3 setup_frequency.py --ads_dir ads_C_ontop_0 --slab_dir ../

    # Batch: set up for all ads_* directories
    python3 setup_frequency.py --batch --slab_dir ../

    # Only for the lowest-energy config (from adsorption_energies.json)
    python3 setup_frequency.py --best_only --ads_energies adsorption_energies.json --slab_dir ../

    # Include top surface layer atoms in frequency calculation
    python3 setup_frequency.py --ads_dir ads_C_ontop_0 --slab_dir ../ --relax_surface_layers 1
"""

import os
import re
import json
import glob
import shutil
import argparse
import numpy as np


def read_poscar(filepath):
    """Read POSCAR/CONTCAR and return structured data."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    comment = lines[0].strip()
    scale = float(lines[1].strip())
    lattice_lines = [lines[i].strip() for i in range(2, 5)]
    lattice_vectors = []
    for i in range(2, 5):
        lattice_vectors.append([float(x) for x in lines[i].split()])

    species = lines[5].split()
    counts = list(map(int, lines[6].split()))
    total_atoms = sum(counts)

    # Check for selective dynamics
    idx = 7
    has_selective = False
    if lines[idx].strip().lower().startswith('s'):
        has_selective = True
        idx += 1

    coord_type = lines[idx].strip()
    idx += 1

    positions = []
    flags = []
    for i in range(idx, idx + total_atoms):
        parts = lines[i].split()
        positions.append([float(parts[0]), float(parts[1]), float(parts[2])])
        if has_selective and len(parts) >= 6:
            flags.append(f"{parts[3]} {parts[4]} {parts[5]}")
        else:
            flags.append("T T T")

    return {
        'comment': comment,
        'scale': scale,
        'lattice_lines': lattice_lines,
        'lattice_vectors': lattice_vectors,
        'species': species,
        'counts': counts,
        'coord_type': coord_type,
        'positions': positions,
        'flags': flags,
        'total_atoms': total_atoms,
    }


def write_poscar_selective(data, flags, filepath):
    """Write POSCAR with selective dynamics."""
    with open(filepath, 'w') as f:
        f.write(data['comment'] + '\n')
        f.write(f"  {data['scale']}\n")
        for lat in data['lattice_lines']:
            f.write(f"  {lat}\n")
        f.write("  " + "  ".join(data['species']) + "\n")
        f.write("  " + "  ".join(map(str, data['counts'])) + "\n")
        f.write("Selective dynamics\n")
        f.write(data['coord_type'] + "\n")
        for i, pos in enumerate(data['positions']):
            f.write(f"  {pos[0]:.16f}  {pos[1]:.16f}  {pos[2]:.16f}  {flags[i]}\n")


def identify_adsorbate_atoms(ads_data, slab_species, slab_counts):
    """
    Identify which atom indices are adsorbate atoms by comparing
    species counts with the clean slab.

    Returns list of atom indices (0-based) that are adsorbate atoms.
    """
    slab_comp = {}
    for sp, cnt in zip(slab_species, slab_counts):
        slab_comp[sp] = slab_comp.get(sp, 0) + cnt

    ads_comp = {}
    for sp, cnt in zip(ads_data['species'], ads_data['counts']):
        ads_comp[sp] = ads_comp.get(sp, 0) + cnt

    # Find extra atoms per species
    extra = {}
    for sp in ads_comp:
        diff = ads_comp[sp] - slab_comp.get(sp, 0)
        if diff > 0:
            extra[sp] = diff

    # The extra atoms are at the END of each species block (VASP convention)
    adsorbate_indices = []
    atom_offset = 0
    for sp, cnt in zip(ads_data['species'], ads_data['counts']):
        if sp in extra:
            # Last 'extra[sp]' atoms of this species block are adsorbate
            n_extra = extra[sp]
            for i in range(cnt - n_extra, cnt):
                adsorbate_indices.append(atom_offset + i)
        atom_offset += cnt

    return adsorbate_indices


def get_top_surface_indices(data, n_layers=1, layer_tol=0.5):
    """
    Get indices of atoms in the top n_layers of the slab.
    layer_tol: tolerance in Angstrom for grouping atoms into layers.
    """
    c_z = data['lattice_vectors'][2][2] * data['scale']
    z_cart = np.array([p[2] * c_z for p in data['positions']])

    # Sort unique z-values to find layers
    z_sorted = np.sort(np.unique(np.round(z_cart, decimals=1)))[::-1]  # top to bottom

    # Group into layers
    layers = []
    current_layer = [z_sorted[0]]
    for z in z_sorted[1:]:
        if abs(z - current_layer[-1]) < layer_tol:
            current_layer.append(z)
        else:
            layers.append(current_layer)
            current_layer = [z]
            if len(layers) >= n_layers:
                break
    if len(layers) < n_layers:
        layers.append(current_layer)

    # Get z range for top n_layers
    top_z_values = []
    for layer in layers[:n_layers]:
        top_z_values.extend(layer)
    z_cutoff = min(top_z_values) - layer_tol

    indices = []
    for i, z in enumerate(z_cart):
        if z >= z_cutoff:
            indices.append(i)

    return indices


def generate_freq_incar(output_path, base_incar=None, nfree=2, potim=0.015):
    """
    Generate INCAR for frequency calculation.

    Key settings:
    - IBRION = 5: finite differences (symmetric, more accurate)
    - NFREE = 2: central differences (2 displacements per DOF)
    - NSW = 1: needed for IBRION=5
    - POTIM = 0.015: displacement step size (Angstrom)
    - ISIF = 0: no cell relaxation
    - LREAL = False: reciprocal space projection for accuracy
    - EDIFF = 1e-07: tighter convergence for frequencies
    """
    params = {}

    # Read base INCAR if provided
    if base_incar and os.path.exists(base_incar):
        with open(base_incar, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or '=' not in line:
                    continue
                key, val = line.split('=', 1)
                params[key.strip()] = val.strip()

    # Override with frequency-specific settings
    freq_params = {
        'IBRION': '5',
        'NFREE': str(nfree),
        'NSW': '1',
        'POTIM': str(potim),
        'ISIF': '0',
        'EDIFF': '1e-07',
        'LREAL': 'False',
        'PREC': 'Accurate',
        'ISMEAR': '0',
        'SIGMA': '0.05',
        'LWAVE': 'False',
        'LCHARG': 'False',
        'LAECHG': 'False',
        'LELF': 'False',
        'LVTOT': 'False',
        'LVHAR': 'False',
        'NELM': '200',
        'NELMIN': '5',
    }

    # Keep spin, ENCUT, ENAUG, KSPACING, etc. from base
    params.update(freq_params)

    # Remove dipole correction for frequency (can cause issues)
    # Actually keep LDIPOL/IDIPOL for slab calculations
    # Remove NSW-related tags that conflict
    for key_to_remove in ['LAECHG', 'LELF', 'LVTOT']:
        if key_to_remove in params:
            params[key_to_remove] = 'False'

    with open(output_path, 'w') as f:
        f.write("# INCAR for frequency calculation (finite differences)\n")
        f.write("# Only adsorbate atoms (+ optional surface layer) are displaced\n")
        f.write("# via selective dynamics T T T / F F F\n\n")
        for key in sorted(params.keys()):
            f.write(f"{key} = {params[key]}\n")


def setup_freq_calc(ads_dir, slab_dir, output_suffix='freq',
                    relax_surface_layers=0, base_incar=None,
                    nfree=2, potim=0.015):
    """
    Set up a frequency calculation directory from a completed adsorption calc.

    Steps:
    1. Read CONTCAR from adsorption calc
    2. Read slab composition to identify adsorbate atoms
    3. Set selective dynamics: T T T for adsorbate (+ top layers), F F F for rest
    4. Write POSCAR + INCAR for frequency calc
    """
    # Read the relaxed adsorption structure
    contcar = os.path.join(ads_dir, 'CONTCAR')
    if not os.path.exists(contcar):
        raise FileNotFoundError(f"No CONTCAR in {ads_dir}. Has the relaxation finished?")

    ads_data = read_poscar(contcar)

    # Read slab composition
    slab_poscar = None
    for fname in ['CONTCAR', 'POSCAR']:
        p = os.path.join(slab_dir, fname)
        if os.path.exists(p):
            slab_poscar = p
            break
    if slab_poscar is None:
        raise FileNotFoundError(f"No POSCAR/CONTCAR in slab dir: {slab_dir}")

    slab_data = read_poscar(slab_poscar)

    # Identify adsorbate atom indices
    ads_indices = identify_adsorbate_atoms(ads_data, slab_data['species'], slab_data['counts'])

    # Optionally include top surface layer atoms
    surface_indices = []
    if relax_surface_layers > 0:
        surface_indices = get_top_surface_indices(ads_data, n_layers=relax_surface_layers)
        # Remove adsorbate indices from surface indices (avoid double counting)
        surface_indices = [i for i in surface_indices if i not in ads_indices]

    relax_indices = set(ads_indices + surface_indices)

    # Build selective dynamics flags
    flags = []
    for i in range(ads_data['total_atoms']):
        if i in relax_indices:
            flags.append("T T T")
        else:
            flags.append("F F F")

    n_relax = flags.count("T T T")
    n_fixed = flags.count("F F F")

    # Create output directory
    freq_dir = os.path.join(ads_dir, output_suffix)
    os.makedirs(freq_dir, exist_ok=True)

    # Write POSCAR with selective dynamics
    write_poscar_selective(ads_data, flags, os.path.join(freq_dir, 'POSCAR'))

    # Generate INCAR
    generate_freq_incar(
        os.path.join(freq_dir, 'INCAR'),
        base_incar=base_incar,
        nfree=nfree,
        potim=potim,
    )

    # Copy KPOINTS if exists
    kpoints = os.path.join(ads_dir, 'KPOINTS')
    if os.path.exists(kpoints):
        shutil.copy2(kpoints, os.path.join(freq_dir, 'KPOINTS'))

    # Copy POTCAR if exists
    potcar = os.path.join(ads_dir, 'POTCAR')
    if os.path.exists(potcar):
        shutil.copy2(potcar, os.path.join(freq_dir, 'POTCAR'))

    return {
        'ads_dir': ads_dir,
        'freq_dir': freq_dir,
        'n_adsorbate_atoms': len(ads_indices),
        'n_surface_atoms': len(surface_indices),
        'n_relaxed': n_relax,
        'n_fixed': n_fixed,
        'n_total': ads_data['total_atoms'],
        'n_displacements': n_relax * 3 * nfree,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Set up VASP frequency calculations for ZPE correction."
    )
    parser.add_argument('--ads_dir', default=None, help="Single adsorption directory")
    parser.add_argument('--batch', action='store_true', help="Process all ads_* directories")
    parser.add_argument('--pattern', default='ads_*', help="Glob pattern for batch mode")
    parser.add_argument('--best_only', action='store_true',
                        help="Only set up freq calc for lowest-energy config")
    parser.add_argument('--ads_energies', default='adsorption_energies.json',
                        help="JSON file with adsorption energies (for --best_only)")
    parser.add_argument('--slab_dir', required=True, help="Clean slab directory (for species comparison)")
    parser.add_argument('--base_incar', default=None, help="Base INCAR to inherit settings from")
    parser.add_argument('--output_suffix', default='freq', help="Subdirectory name for freq calc (default: freq)")
    parser.add_argument('--relax_surface_layers', type=int, default=0,
                        help="Number of top surface layers to include in frequency calc (default: 0)")
    parser.add_argument('--nfree', type=int, default=2, help="NFREE: 2=central differences (default: 2)")
    parser.add_argument('--potim', type=float, default=0.015, help="Displacement step in Ang (default: 0.015)")
    parser.add_argument('--skip_graphene', action='store_true', default=True,
                        help="Skip graphene configurations (default: True)")

    args = parser.parse_args()

    if args.best_only:
        # Only process the lowest-energy config
        if not os.path.exists(args.ads_energies):
            print(f"ERROR: {args.ads_energies} not found. Run adsorption energy calculation first.")
            return

        with open(args.ads_energies) as f:
            energies = json.load(f)

        if not energies:
            print("No adsorption energies found.")
            return

        # Sort and take the best (most negative)
        energies.sort(key=lambda x: x['e_adsorption'])
        best = energies[0]
        ads_dirs = [best['directory']]
        print(f"Best config: {best['directory']} (E_ads = {best['e_adsorption']:.4f} eV)")

    elif args.batch:
        ads_dirs = sorted(glob.glob(args.pattern))
        ads_dirs = [d for d in ads_dirs if os.path.isdir(d)]
        if not ads_dirs:
            print(f"No directories matching: {args.pattern}")
            return
    elif args.ads_dir:
        ads_dirs = [args.ads_dir]
    else:
        parser.error("Specify --ads_dir, --batch, or --best_only")

    # Filter out graphene if requested
    if args.skip_graphene:
        original_count = len(ads_dirs)
        ads_dirs = [d for d in ads_dirs if 'graphene' not in os.path.basename(d)]
        skipped = original_count - len(ads_dirs)
        if skipped > 0:
            print(f"Skipping {skipped} graphene configuration(s) (no freq calc needed)")

    print(f"\nSetting up frequency calculations for {len(ads_dirs)} directories...")
    print(f"Slab reference: {args.slab_dir}")
    print(f"NFREE={args.nfree}, POTIM={args.potim}")
    if args.relax_surface_layers > 0:
        print(f"Including top {args.relax_surface_layers} surface layer(s) in frequency calc")
    print("=" * 70)

    results = []
    for ads_dir in ads_dirs:
        try:
            result = setup_freq_calc(
                ads_dir, args.slab_dir,
                output_suffix=args.output_suffix,
                relax_surface_layers=args.relax_surface_layers,
                base_incar=args.base_incar,
                nfree=args.nfree,
                potim=args.potim,
            )
            results.append(result)
            print(f"\n  {ads_dir}:")
            print(f"    Freq dir: {result['freq_dir']}")
            print(f"    Atoms: {result['n_adsorbate_atoms']} adsorbate + "
                  f"{result['n_surface_atoms']} surface = {result['n_relaxed']} relaxed "
                  f"/ {result['n_fixed']} fixed / {result['n_total']} total")
            print(f"    Displacements: {result['n_displacements']} "
                  f"(~{result['n_displacements']} ionic steps)")
        except Exception as e:
            print(f"\n  {ads_dir}: ERROR - {e}")
            continue

    if results:
        print(f"\n{'=' * 70}")
        print(f"Set up {len(results)} frequency calculations successfully.")
        total_disp = sum(r['n_displacements'] for r in results)
        print(f"Total displacements across all configs: {total_disp}")


if __name__ == "__main__":
    main()
