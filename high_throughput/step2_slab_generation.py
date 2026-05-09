#!/usr/bin/env python3
"""
Step 2: Generate slab models from relaxed bulk structures.

Uses surfaxe for non-dipolar terminations on low-index surfaces.
Falls back to pymatgen SlabGenerator if surfaxe fails.

Usage:
    # Setup + submit in one go (most common)
    python3 step2_slab_generation.py --bulk_dir ./1_bulk --work_dir ./2_slabs

    # Or step-by-step
    python3 step2_slab_generation.py setup --bulk_dir ./1_bulk --work_dir ./2_slabs
    python3 step2_slab_generation.py submit --work_dir ./2_slabs
    python3 step2_slab_generation.py status --work_dir ./2_slabs
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.incar_generator import generate_incar, write_incar, read_species_from_poscar
from utils.job_manager import submit_job, check_vasp_converged, parse_energy


def ensure_potcar(calc_dir):
    """Generate POTCAR using vaspkit if not already present."""
    potcar_path = os.path.join(calc_dir, 'POTCAR')
    if os.path.exists(potcar_path) and os.path.getsize(potcar_path) > 0:
        return True

    poscar_path = os.path.join(calc_dir, 'POSCAR')
    if not os.path.exists(poscar_path):
        return False

    try:
        subprocess.run(
            ['vaspkit', '-task', '103'],
            cwd=calc_dir,
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if os.path.exists(potcar_path) and os.path.getsize(potcar_path) > 0:
        return True

    print(f"    WARNING: POTCAR generation failed in {calc_dir}")
    return False


HKL_INDICES = [
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (1, 1, 0), (1, 0, 1), (0, 1, 1),
    (1, 1, 1)
]


def write_clean_poscar(slab_struct, filepath):
    """Write a pymatgen Structure as a clean POSCAR without site properties.

    pymatgen's slab.to(fmt='poscar') can produce broken POSCARs with
    oxidation states or site properties that create duplicate species
    (e.g., 24 types for 40 atoms). This writes a clean POSCAR with
    only element symbols.
    """
    from pymatgen.io.vasp import Poscar
    # Remove all site properties (oxidation states, magmoms, etc.)
    clean = slab_struct.copy()
    clean.remove_site_property('selective_dynamics') if 'selective_dynamics' in clean.site_properties else None
    # Remove oxidation states by converting to simple Structure
    from pymatgen.core import Structure, Lattice
    clean_struct = Structure(
        lattice=clean.lattice,
        species=[site.specie.element if hasattr(site.specie, 'element') else site.specie
                 for site in clean],
        coords=[site.frac_coords for site in clean],
    )
    poscar = Poscar(clean_struct, sort_structure=True)
    poscar.write_file(filepath)


def add_selective_dynamics(poscar_path, relax_fraction=0.25):
    """Add selective dynamics to a slab POSCAR (top+bottom relax_fraction relaxed)."""
    with open(poscar_path, 'r') as f:
        lines = f.readlines()

    comment = lines[0]
    scale = float(lines[1].strip())

    # Check if selective dynamics already present
    idx = 7
    has_sd = lines[idx].strip().lower().startswith('s')
    if has_sd:
        idx += 1
    coord_type = lines[idx].strip()
    idx += 1

    species = lines[5].split()
    counts = list(map(int, lines[6].split()))
    total = sum(counts)

    # Parse positions
    c_z = 1.0
    if coord_type.lower().startswith('d'):
        lattice_c = [float(x) for x in lines[4].split()]
        c_z = lattice_c[2] * scale

    positions_z = []
    coord_lines = []
    for i in range(total):
        parts = lines[idx + i].split()
        z = float(parts[2])
        if coord_type.lower().startswith('d'):
            z_cart = z * c_z
        else:
            z_cart = z
        positions_z.append(z_cart)
        coord_lines.append(f"  {parts[0]}  {parts[1]}  {parts[2]}")

    z_min = min(positions_z)
    z_max = max(positions_z)
    thickness = z_max - z_min
    z_top_cut = z_max - relax_fraction * thickness
    z_bot_cut = z_min + relax_fraction * thickness

    # Write back with selective dynamics
    with open(poscar_path, 'w') as f:
        f.write(comment)
        f.write(lines[1])
        for i in range(2, 5):
            f.write(lines[i])
        f.write(lines[5])
        f.write(lines[6])
        f.write('Selective dynamics\n')
        f.write(coord_type + '\n')
        for i in range(total):
            z = positions_z[i]
            if z >= z_top_cut or z <= z_bot_cut:
                flag = 'T T T'
            else:
                flag = 'F F F'
            f.write(f"{coord_lines[i]}  {flag}\n")


def _run_surfaxe(bulk_contcar, hkl_indices, thickness, vacuum, result_holder):
    """Worker function for surfaxe with timeout."""
    from surfaxe.generation import generate_slabs
    result_holder.append(generate_slabs(
        structure=bulk_contcar,
        hkl=hkl_indices,
        thicknesses=[thickness],
        vacuums=[vacuum],
        save_slabs=False,
        save_metadata=False,
        processes=1,
    ))


def generate_slabs_for_compound(bulk_contcar, compound_slab_dir, vacuum=20.0,
                                 thickness=20.0, timeout=120):
    """Generate slabs for one compound using surfaxe, fallback to pymatgen.

    Args:
        timeout: max seconds to wait for surfaxe per compound (default: 120).
    """
    slab_dirs = []
    all_slabs = []

    # Try surfaxe with timeout
    try:
        import multiprocessing
        manager = multiprocessing.Manager()
        result_holder = manager.list()

        proc = multiprocessing.Process(
            target=_run_surfaxe,
            args=(bulk_contcar, HKL_INDICES, thickness, vacuum, result_holder)
        )
        proc.start()
        proc.join(timeout=timeout)

        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            print(f"    surfaxe timed out after {timeout}s, falling back to pymatgen")
        elif result_holder:
            all_slabs = result_holder[0] or []
    except Exception as e:
        print(f"    surfaxe failed: {e}")

    # Fallback to pymatgen if surfaxe found nothing
    if not all_slabs:
        try:
            from pymatgen.core import Structure
            from pymatgen.core.surface import SlabGenerator
            struct = Structure.from_file(bulk_contcar)

            for hkl in HKL_INDICES:
                try:
                    sg = SlabGenerator(struct, hkl,
                                       min_slab_size=thickness,
                                       min_vacuum_size=vacuum,
                                       center_slab=True)
                    slabs = sg.get_slabs()
                    for j, slab in enumerate(slabs[:3]):
                        hkl_str = "".join(map(str, hkl))
                        folder = os.path.join(compound_slab_dir,
                                              f"surf_{hkl_str}_term_{j}")
                        os.makedirs(folder, exist_ok=True)
                        write_clean_poscar(slab, os.path.join(folder, 'POSCAR'))

                        m = slab.lattice.matrix
                        area = np.linalg.norm(np.cross(m[0], m[1]))

                        with open(os.path.join(folder, 'surface_info.log'), 'w') as f:
                            f.write(f"HKL Index: {hkl}\n")
                            f.write(f"Area (A^2): {area:.4f}\n")
                            f.write(f"Source: pymatgen SlabGenerator\n")

                        slab_dirs.append(folder)
                        print(f"    Generated (pymatgen): surf_{hkl_str}_term_{j}")
                except Exception as e:
                    print(f"    pymatgen failed for {hkl}: {e}")
        except ImportError:
            print("    ERROR: Neither surfaxe nor pymatgen available!")
            return []

        return slab_dirs

    # Process surfaxe results
    for i, slab_dict in enumerate(all_slabs):
        try:
            hkl_str = "".join(map(str, slab_dict['hkl']))
            label = slab_dict.get('label', i)
            folder = os.path.join(compound_slab_dir, f"surf_{hkl_str}_term_{label}")
            os.makedirs(folder, exist_ok=True)

            slab_struct = slab_dict['slab']
            write_clean_poscar(slab_struct, os.path.join(folder, 'POSCAR'))

            m = slab_struct.lattice.matrix
            area = np.linalg.norm(np.cross(m[0], m[1]))

            with open(os.path.join(folder, 'surface_info.log'), 'w') as f:
                f.write(f"HKL Index: {slab_dict['hkl']}\n")
                f.write(f"Tasker Type: {slab_dict.get('tasker', 'Unknown')}\n")
                f.write(f"Area (A^2): {area:.4f}\n")
                f.write(f"Source: surfaxe\n")

            slab_dirs.append(folder)
            print(f"    Generated (surfaxe): surf_{hkl_str}_term_{label}")
        except Exception as e:
            print(f"    Error processing slab {i}: {e}")

    return slab_dirs


def setup_slab_calculations(bulk_dir, work_dir, vacuum=20.0, thickness=20.0,
                             relax_fraction=0.25, timeout=120):
    """Set up slab generation and relaxation for all converged bulk calculations."""
    os.makedirs(work_dir, exist_ok=True)

    manifest_path = os.path.join(bulk_dir, 'manifest.json')
    if not os.path.exists(manifest_path):
        print(f"No manifest.json in {bulk_dir}. Run step1 first.")
        return

    with open(manifest_path) as f:
        bulk_manifest = json.load(f)

    slab_manifest = []
    total_slabs = 0

    for entry in bulk_manifest:
        bulk_calc_dir = entry['dir']
        compound = entry['compound']
        elements = entry.get('elements', '')

        # Check if bulk converged
        converged, msg = check_vasp_converged(bulk_calc_dir)
        if not converged:
            print(f"  {compound}: bulk not converged, skipping")
            continue

        bulk_contcar = os.path.join(bulk_calc_dir, 'CONTCAR')
        if not os.path.exists(bulk_contcar):
            print(f"  {compound}: no CONTCAR, skipping")
            continue

        compound_slab_dir = os.path.join(work_dir, elements, compound)
        print(f"\n  {compound}:")

        # Generate slabs
        slab_dirs = generate_slabs_for_compound(
            bulk_contcar, compound_slab_dir, vacuum, thickness,
            timeout=timeout,
        )

        if not slab_dirs:
            print(f"    No slabs generated!")
            continue

        # Add selective dynamics and INCAR to each slab
        for slab_dir in slab_dirs:
            poscar = os.path.join(slab_dir, 'POSCAR')
            add_selective_dynamics(poscar, relax_fraction)

            species, counts = read_species_from_poscar(poscar)
            params = generate_incar(species, counts, calc_type='slab_relax')
            write_incar(params, os.path.join(slab_dir, 'INCAR'))

            # Generate POTCAR via vaspkit
            ensure_potcar(slab_dir)

        slab_manifest.append({
            'compound': compound,
            'elements': elements,
            'bulk_dir': bulk_calc_dir,
            'slab_dirs': slab_dirs,
        })
        total_slabs += len(slab_dirs)

    # Save manifest
    manifest_path = os.path.join(work_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(slab_manifest, f, indent=2)
    print(f"\n{'=' * 70}")
    print(f"Total: {len(slab_manifest)} compounds, {total_slabs} slab calculations")


def submit_slab_jobs(work_dir, partition='cu', ntasks=64):
    """Submit all slab relaxation jobs."""
    manifest_path = os.path.join(work_dir, 'manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)

    submitted = 0
    for entry in manifest:
        for slab_dir in entry['slab_dirs']:
            converged, _ = check_vasp_converged(slab_dir)
            if converged:
                continue

            from utils.job_manager import read_saved_jobid, get_job_status
            jobid = read_saved_jobid(slab_dir)
            if jobid and get_job_status(jobid) in ('RUNNING', 'PENDING'):
                continue

            slab_name = os.path.basename(slab_dir)
            job_name = f"slb_{entry['compound'][:8]}_{slab_name[-6:]}"
            jobid = submit_job(slab_dir, job_name=job_name,
                               partition=partition, ntasks=ntasks)
            if jobid:
                print(f"  Submitted {entry['compound']}/{slab_name}: job {jobid}")
                submitted += 1

    print(f"\nSubmitted: {submitted} slab jobs")


def show_status(work_dir):
    """Show status of all slab calculations."""
    manifest_path = os.path.join(work_dir, 'manifest.json')
    if not os.path.exists(manifest_path):
        print("No manifest.json found. Run setup first.")
        return
    with open(manifest_path) as f:
        manifest = json.load(f)
    done = 0
    total = 0
    for entry in manifest:
        print(f"\n  {entry['compound']}:")
        for sd in entry['slab_dirs']:
            total += 1
            converged, msg = check_vasp_converged(sd)
            name = os.path.basename(sd)
            if converged:
                done += 1
                print(f"    {name:<25} DONE         {msg}")
            else:
                print(f"    {name:<25} PENDING      {msg}")
    print(f"\n  Total: {done}/{total} converged")


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Slab generation + relaxation.\n"
                    "Without subcommand: runs setup then submit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Top-level args (used when running without subcommand)
    parser.add_argument('--bulk_dir', default='./1_bulk')
    parser.add_argument('--work_dir', default='./2_slabs')
    parser.add_argument('--vacuum', type=float, default=20.0)
    parser.add_argument('--thickness', type=float, default=20.0)
    parser.add_argument('--relax_fraction', type=float, default=0.25)
    parser.add_argument('--timeout', type=int, default=120,
                        help="Timeout in seconds for surfaxe per compound (default: 120)")
    parser.add_argument('--partition', default='cu')
    parser.add_argument('--ntasks', type=int, default=64)

    sub = parser.add_subparsers(dest='command')

    p_setup = sub.add_parser('setup', help="Generate slabs only (no submit)")
    p_setup.add_argument('--bulk_dir', default='./1_bulk')
    p_setup.add_argument('--work_dir', default='./2_slabs')
    p_setup.add_argument('--vacuum', type=float, default=20.0)
    p_setup.add_argument('--thickness', type=float, default=20.0)
    p_setup.add_argument('--relax_fraction', type=float, default=0.25)
    p_setup.add_argument('--timeout', type=int, default=120)

    p_submit = sub.add_parser('submit', help="Submit SLURM jobs")
    p_submit.add_argument('--work_dir', default='./2_slabs')
    p_submit.add_argument('--partition', default='cu')
    p_submit.add_argument('--ntasks', type=int, default=64)

    p_status = sub.add_parser('status', help="Check calculation status")
    p_status.add_argument('--work_dir', default='./2_slabs')

    args = parser.parse_args()

    if args.command == 'setup':
        setup_slab_calculations(args.bulk_dir, args.work_dir,
                                 args.vacuum, args.thickness, args.relax_fraction,
                                 args.timeout)
    elif args.command == 'submit':
        submit_slab_jobs(args.work_dir, args.partition, args.ntasks)
    elif args.command == 'status':
        show_status(args.work_dir)
    else:
        # No subcommand: run setup + submit
        setup_slab_calculations(args.bulk_dir, args.work_dir,
                                 args.vacuum, args.thickness, args.relax_fraction,
                                 args.timeout)
        print(f"\n{'=' * 70}")
        print("Submitting slab relaxation jobs...")
        print("=" * 70)
        submit_slab_jobs(args.work_dir, args.partition, args.ntasks)


if __name__ == "__main__":
    main()
