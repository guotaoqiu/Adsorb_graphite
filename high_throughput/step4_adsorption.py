#!/usr/bin/env python3
"""
Step 4: Generate adsorption configurations, relax, and calculate adsorption energies.

For each compound's most stable surface (from Step 3):
1. Generate adsorption configs for single_C, C_chain_v, C_ring
2. Auto-detect asymmetric slabs and handle both sides
3. Auto-supercell if cell is too small for adsorbate
4. Two-phase relaxation (prerelax + refine)
5. Calculate adsorption energies with per-adsorbate references

Usage:
    python3 step4_adsorption.py setup --slab_dir ./2_slabs --work_dir ./3_adsorption
    python3 step4_adsorption.py submit --work_dir ./3_adsorption
    python3 step4_adsorption.py energy --work_dir ./3_adsorption
    python3 step4_adsorption.py status --work_dir ./3_adsorption
"""

import os
import sys
import json
import shutil
import argparse
import re
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.incar_generator import generate_incar, write_incar, read_species_from_poscar
from utils.job_manager import submit_job, check_vasp_converged, parse_energy


def ensure_potcar(calc_dir):
    """Generate POTCAR using vaspkit if not already present."""
    import subprocess
    potcar_path = os.path.join(calc_dir, 'POTCAR')
    if os.path.exists(potcar_path) and os.path.getsize(potcar_path) > 0:
        return True
    if not os.path.exists(os.path.join(calc_dir, 'POSCAR')):
        return False
    try:
        subprocess.run(['vaspkit', '-task', '103'], cwd=calc_dir,
                       capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    if os.path.exists(potcar_path) and os.path.getsize(potcar_path) > 0:
        return True
    print(f"    WARNING: POTCAR generation failed in {calc_dir}")
    return False

# Gas-phase reference energies
REF_ENERGIES = {
    'single_C': -1.32064672,
    'C_chain': -19.02083563,
    'C_ring': -42.82686827,
}

ADSORBATE_TYPES = ['single_C', 'C_chain_v', 'C_ring']


def detect_slab_asymmetry(slab_path, layer_tol=1.0):
    """Check if top and bottom surface layers have different composition."""
    from ase.io import read
    slab = read(slab_path, format='vasp')
    positions = slab.get_positions()
    symbols = slab.get_chemical_symbols()
    z = positions[:, 2]
    z_min, z_max = z.min(), z.max()

    top_comp = {}
    bot_comp = {}
    for i, zi in enumerate(z):
        if zi >= z_max - layer_tol:
            s = symbols[i]
            top_comp[s] = top_comp.get(s, 0) + 1
        elif zi <= z_min + layer_tol:
            s = symbols[i]
            bot_comp[s] = bot_comp.get(s, 0) + 1

    def normalize(comp):
        total = sum(comp.values())
        return {k: v / total for k, v in comp.items()} if total > 0 else {}

    top_f, bot_f = normalize(top_comp), normalize(bot_comp)
    all_sp = set(list(top_f.keys()) + list(bot_f.keys()))
    max_diff = max((abs(top_f.get(s, 0) - bot_f.get(s, 0)) for s in all_sp), default=0)

    return max_diff > 0.15


def generate_adsorption_configs(slab_contcar, output_dir, adsorbate_type,
                                 height=2.5, min_image_dist=8.0):
    """Generate adsorption configurations using pymatgen AdsorbateSiteFinder."""
    from ase.io import read, write
    from ase import Atoms
    from ase.build import make_supercell
    from pymatgen.core import Structure
    from pymatgen.analysis.adsorption import AdsorbateSiteFinder

    slab_ase = read(slab_contcar, format='vasp')
    n_slab = len(slab_ase)

    # Build adsorbate
    if adsorbate_type == 'single_C':
        ads = Atoms('C', positions=[(0, 0, 0)])
        ads_diameter = 0.0
        ads_label = 'C'
    elif adsorbate_type == 'C_chain_v':
        ads = Atoms('C3', positions=[(0, 0, 0), (0, 0, 1.3), (0, 0, 2.6)])
        ads_diameter = 0.0
        ads_label = 'Cchain3v'
    elif adsorbate_type == 'C_ring':
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        r = 1.4 / (2 * np.sin(np.pi / 6))
        pos = [(r * np.cos(a), r * np.sin(a), 0) for a in angles]
        ads = Atoms('C6', positions=pos)
        ads_diameter = 2 * r
        ads_label = 'Cring6'
    else:
        return []

    # Check supercell
    a_len = np.linalg.norm(slab_ase.cell[0][:2])
    b_len = np.linalg.norm(slab_ase.cell[1][:2])
    required = ads_diameter + min_image_dist
    na = max(1, int(np.ceil(required / a_len))) if ads_diameter > 0 else 1
    nb = max(1, int(np.ceil(required / b_len))) if ads_diameter > 0 else 1

    sc_tag = ''
    if na > 1 or nb > 1:
        P = np.array([[na, 0, 0], [0, nb, 0], [0, 0, 1]])
        slab_ase = make_supercell(slab_ase, P)
        n_slab = len(slab_ase)
        sc_tag = f'_{na}x{nb}'
        # Write temp supercell for pymatgen
        tmp_path = slab_contcar + f'.super{na}x{nb}'
        write(tmp_path, slab_ase, format='vasp')
        slab_contcar = tmp_path

    # Find sites
    slab_pmg = Structure.from_file(slab_contcar)
    asf = AdsorbateSiteFinder(slab_pmg)
    sites = asf.find_adsorption_sites(distance=height, symm_reduce=0.01,
                                       near_reduce=0.5)

    configs = []
    for site_type in ['ontop', 'bridge', 'hollow']:
        for i, site_xyz in enumerate(sites.get(site_type, [])):
            slab_copy = slab_ase.copy()
            ads_copy = ads.copy()

            z_bottom = ads_copy.positions[:, 2].min()
            ads_copy.positions[:, 2] += (site_xyz[2] - z_bottom)
            com = ads_copy.get_center_of_mass()[:2]
            ads_copy.positions[:, 0] += site_xyz[0] - com[0]
            ads_copy.positions[:, 1] += site_xyz[1] - com[1]

            combined = slab_copy + ads_copy
            combined.cell = slab_copy.cell
            combined.pbc = slab_copy.pbc

            name = f"{ads_label}{sc_tag}_{site_type}_{i}"
            configs.append((combined, name, n_slab))

    return configs


def write_poscar_with_selective_dynamics(atoms, flags, filepath):
    """Write POSCAR with selective dynamics, stripping any existing flags."""
    from ase.io import write
    write(filepath + '.tmp', atoms, format='vasp')

    with open(filepath + '.tmp', 'r') as f:
        lines = f.readlines()

    idx = 7
    if lines[idx].strip().lower().startswith('s'):
        idx += 1
    coord_line = lines[idx]
    idx += 1

    counts = list(map(int, lines[6].split()))
    total = sum(counts)

    with open(filepath, 'w') as f:
        for i in range(7):
            f.write(lines[i])
        f.write('Selective dynamics\n')
        f.write(coord_line)
        for i in range(total):
            parts = lines[idx + i].split()
            f.write(f'  {parts[0]}  {parts[1]}  {parts[2]}  {flags[i]}\n')

    os.remove(filepath + '.tmp')


def apply_selective_dynamics(combined, n_slab, relax_fraction=0.25):
    """Generate selective dynamics flags."""
    positions = combined.get_positions()
    slab_z = positions[:n_slab, 2]
    z_min, z_max = slab_z.min(), slab_z.max()
    thickness = z_max - z_min
    z_cutoff = z_max - relax_fraction * thickness

    flags = []
    for i in range(len(combined)):
        if i >= n_slab or positions[i, 2] >= z_cutoff:
            flags.append('T T T')
        else:
            flags.append('F F F')
    return flags


def setup_adsorption(slab_dir, work_dir, height=2.5, relax_fraction=0.25):
    """Set up adsorption calculations for all compounds."""
    os.makedirs(work_dir, exist_ok=True)

    se_path = os.path.join(slab_dir, 'surface_energies.json')
    if not os.path.exists(se_path):
        print("No surface_energies.json found. Run step3 first.")
        return

    with open(se_path) as f:
        surface_energies = json.load(f)

    slab_manifest_path = os.path.join(slab_dir, 'manifest.json')
    with open(slab_manifest_path) as f:
        slab_manifest = json.load(f)

    ads_manifest = []

    for compound, data in surface_energies.items():
        best_surface = data['most_stable']
        slab_contcar = os.path.join(best_surface['directory'], 'CONTCAR')

        if not os.path.exists(slab_contcar):
            print(f"  {compound}: no CONTCAR for best surface, skipping")
            continue

        print(f"\n  {compound}: best surface = {best_surface['name']} "
              f"(γ={best_surface['gamma_J_m2']:.3f} J/m²)")

        compound_dir = os.path.join(work_dir, compound)
        # Copy clean slab CONTCAR for reference
        os.makedirs(compound_dir, exist_ok=True)
        shutil.copy2(slab_contcar, os.path.join(compound_dir, 'CONTCAR_slab'))

        # Save slab energy for adsorption energy calculation
        slab_energy = parse_energy(best_surface['directory'])
        with open(os.path.join(compound_dir, 'slab_info.json'), 'w') as f:
            json.dump({
                'slab_energy': slab_energy,
                'surface': best_surface['name'],
                'gamma_J_m2': best_surface['gamma_J_m2'],
                'slab_dir': best_surface['directory'],
            }, f, indent=2)

        compound_ads_dirs = []

        for ads_type in ADSORBATE_TYPES:
            print(f"    {ads_type}:")
            try:
                configs = generate_adsorption_configs(
                    slab_contcar, compound_dir, ads_type, height=height
                )
            except Exception as e:
                print(f"      ERROR: {e}")
                continue

            for combined, name, n_slab in configs:
                ads_dir = os.path.join(compound_dir, f'ads_{name}')
                os.makedirs(ads_dir, exist_ok=True)

                flags = apply_selective_dynamics(combined, n_slab, relax_fraction)
                write_poscar_with_selective_dynamics(
                    combined, flags, os.path.join(ads_dir, 'POSCAR'))

                # Generate INCAR (prerelax phase)
                species, counts = read_species_from_poscar(
                    os.path.join(ads_dir, 'POSCAR'))
                params = generate_incar(species, counts, calc_type='ads_relax')
                write_incar(params, os.path.join(ads_dir, 'INCAR'))

                # Generate POTCAR via vaspkit (reads POSCAR, handles new species like C)
                ensure_potcar(ads_dir)

                compound_ads_dirs.append({
                    'dir': ads_dir,
                    'adsorbate': ads_type,
                    'config': name,
                    'n_slab': n_slab,
                })
                print(f"      {name}")

        ads_manifest.append({
            'compound': compound,
            'slab_energy': slab_energy,
            'surface': best_surface['name'],
            'ads_dirs': compound_ads_dirs,
        })

    manifest_path = os.path.join(work_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(ads_manifest, f, indent=2)

    total_configs = sum(len(e['ads_dirs']) for e in ads_manifest)
    print(f"\n{'=' * 70}")
    print(f"Total: {len(ads_manifest)} compounds, {total_configs} adsorption configs")


def submit_ads_jobs(work_dir, partition='cu', ntasks=64):
    """Submit adsorption relaxation jobs."""
    manifest_path = os.path.join(work_dir, 'manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)

    submitted = 0
    for entry in manifest:
        for ads_entry in entry['ads_dirs']:
            ads_dir = ads_entry['dir']

            converged, _ = check_vasp_converged(ads_dir)
            if converged:
                continue

            from utils.job_manager import read_saved_jobid, get_job_status
            jobid = read_saved_jobid(ads_dir)
            if jobid and get_job_status(jobid) in ('RUNNING', 'PENDING'):
                continue

            config = ads_entry['config']
            job_name = f"ads_{entry['compound'][:6]}_{config[:8]}"
            jobid = submit_job(ads_dir, job_name=job_name,
                               partition=partition, ntasks=ntasks)
            if jobid:
                submitted += 1

    print(f"Submitted: {submitted} adsorption jobs")


def detect_adsorbate_type(config_name):
    """Detect adsorbate type from config name."""
    if config_name.startswith('Cring'):
        return 'C_ring'
    elif config_name.startswith('Cchain'):
        return 'C_chain'
    elif config_name.startswith('C_') or config_name.startswith('C '):
        return 'single_C'
    return 'unknown'


def detect_supercell_factor(ads_dir_name):
    """Parse supercell factor from directory name."""
    match = re.search(r'_(\d+)x(\d+)_', ads_dir_name)
    if match:
        return int(match.group(1)) * int(match.group(2))
    return 1


def calculate_adsorption_energies(work_dir):
    """Calculate adsorption energies for all converged configs."""
    manifest_path = os.path.join(work_dir, 'manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)

    all_results = {}

    for entry in manifest:
        compound = entry['compound']
        e_slab = entry['slab_energy']

        if e_slab is None:
            print(f"  {compound}: no slab energy, skipping")
            continue

        print(f"\n  {compound}: E_slab = {e_slab:.6f} eV")
        results = []

        for ads_entry in entry['ads_dirs']:
            ads_dir = ads_entry['dir']
            config = ads_entry['config']

            converged, msg = check_vasp_converged(ads_dir)
            if not converged:
                continue

            e_total = parse_energy(ads_dir)
            if e_total is None:
                continue

            ads_type = detect_adsorbate_type(config)
            sc_factor = detect_supercell_factor(config)
            e_ref = REF_ENERGIES.get(ads_type)

            if e_ref is None:
                continue

            e_ads = e_total - (e_slab * sc_factor) - e_ref

            results.append({
                'config': config,
                'directory': ads_dir,
                'adsorbate_type': ads_type,
                'e_total': e_total,
                'e_ref': e_ref,
                'e_adsorption': e_ads,
                'supercell_factor': sc_factor,
            })

            sc_str = f" [{sc_factor}x]" if sc_factor > 1 else ""
            print(f"    {config:<35} {ads_type:<10} E_ads={e_ads:>10.4f} eV{sc_str}")

        if results:
            results.sort(key=lambda x: x['e_adsorption'])
            all_results[compound] = results

            # Best per adsorbate type
            print(f"    ─── Best per type ───")
            for atype in ['single_C', 'C_chain', 'C_ring']:
                typed = [r for r in results if r['adsorbate_type'] == atype]
                if typed:
                    best = typed[0]
                    print(f"    {atype:<12}: {best['config']:<30} E_ads = {best['e_adsorption']:.4f} eV")

    # Save results
    output_path = os.path.join(work_dir, 'adsorption_energies.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print(f"\n{'=' * 90}")
    print(f"SCREENING SUMMARY")
    print(f"{'=' * 90}")
    print(f"  {'Compound':<25}  {'Best C':<12}  {'Best C3':<12}  {'Best C6':<12}")
    print("  " + "-" * 65)
    for compound, results in sorted(all_results.items()):
        row = {'single_C': 'N/A', 'C_chain': 'N/A', 'C_ring': 'N/A'}
        for atype in row:
            typed = [r for r in results if r['adsorbate_type'] == atype]
            if typed:
                row[atype] = f"{typed[0]['e_adsorption']:.3f} eV"
        print(f"  {compound:<25}  {row['single_C']:<12}  {row['C_chain']:<12}  {row['C_ring']:<12}")

    print(f"\nResults saved: {output_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Step 4: Adsorption screening")
    sub = parser.add_subparsers(dest='command')

    p_setup = sub.add_parser('setup')
    p_setup.add_argument('--slab_dir', default='./2_slabs')
    p_setup.add_argument('--work_dir', default='./3_adsorption')
    p_setup.add_argument('--height', type=float, default=2.5)
    p_setup.add_argument('--relax_fraction', type=float, default=0.25)

    p_submit = sub.add_parser('submit')
    p_submit.add_argument('--work_dir', default='./3_adsorption')
    p_submit.add_argument('--partition', default='cu')
    p_submit.add_argument('--ntasks', type=int, default=64)

    p_energy = sub.add_parser('energy')
    p_energy.add_argument('--work_dir', default='./3_adsorption')

    p_status = sub.add_parser('status')
    p_status.add_argument('--work_dir', default='./3_adsorption')

    args = parser.parse_args()

    if args.command == 'setup':
        setup_adsorption(args.slab_dir, args.work_dir, args.height,
                          args.relax_fraction)
    elif args.command == 'submit':
        submit_ads_jobs(args.work_dir, args.partition, args.ntasks)
    elif args.command == 'energy':
        calculate_adsorption_energies(args.work_dir)
    elif args.command == 'status':
        manifest_path = os.path.join(args.work_dir, 'manifest.json')
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            for entry in manifest:
                done = sum(1 for a in entry['ads_dirs']
                           if check_vasp_converged(a['dir'])[0])
                total = len(entry['ads_dirs'])
                print(f"  {entry['compound']:<30} {done}/{total} converged")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
