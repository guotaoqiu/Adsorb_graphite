#!/usr/bin/env python3
"""
Step 1: Set up and submit bulk structure relaxation for all materials.

Scans structure files under the input directory, creates calculation
directories, generates element-aware INCAR (with U values, LMAXMIX),
and submits SLURM jobs.

Usage:
    # Set up all structures
    python3 step1_bulk_relax.py setup --input_dir /path/to/structures/ --work_dir ./1_bulk

    # Check status
    python3 step1_bulk_relax.py status --work_dir ./1_bulk

    # Resubmit failed jobs
    python3 step1_bulk_relax.py resubmit --work_dir ./1_bulk
"""

import os
import sys
import glob
import shutil
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.incar_generator import generate_incar, write_incar, read_species_from_poscar
from utils.job_manager import (
    submit_job, scan_status, check_vasp_converged, parse_energy
)


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


def find_structure_files(input_dir):
    """Find all .vasp structure files recursively."""
    vasp_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith('.vasp'):
                vasp_files.append(os.path.join(root, f))
    return sorted(vasp_files)


def get_compound_name(vasp_path):
    """Extract compound name from path or filename.
    e.g., '/path/C-Cu-O/mp-556202_Cu(CO)4.vasp' -> 'mp-556202_CuCO4'
    """
    basename = os.path.splitext(os.path.basename(vasp_path))[0]
    # Clean up parentheses for directory names
    clean = basename.replace('(', '').replace(')', '')
    return clean


def setup_bulk_calculations(input_dir, work_dir, partition='cu', ntasks=64):
    """Set up bulk relaxation calculations for all structures."""
    os.makedirs(work_dir, exist_ok=True)

    vasp_files = find_structure_files(input_dir)
    if not vasp_files:
        print(f"No .vasp files found in {input_dir}")
        return

    print(f"Found {len(vasp_files)} structure files")
    print("=" * 70)

    manifest = []

    for vasp_path in vasp_files:
        compound = get_compound_name(vasp_path)
        # Include element combo in path for organization
        parent = os.path.basename(os.path.dirname(vasp_path))
        calc_dir = os.path.join(work_dir, parent, compound)

        if os.path.exists(os.path.join(calc_dir, 'POSCAR')):
            print(f"  {compound}: already set up, skipping")
            manifest.append({'compound': compound, 'dir': calc_dir,
                             'source': vasp_path, 'elements': parent})
            continue

        os.makedirs(calc_dir, exist_ok=True)

        # Copy structure as POSCAR
        shutil.copy2(vasp_path, os.path.join(calc_dir, 'POSCAR'))

        # Read species for INCAR generation
        try:
            species, counts = read_species_from_poscar(os.path.join(calc_dir, 'POSCAR'))
        except Exception as e:
            print(f"  {compound}: ERROR reading POSCAR - {e}")
            continue

        # Generate INCAR
        params = generate_incar(species, counts, calc_type='bulk_relax')
        write_incar(params, os.path.join(calc_dir, 'INCAR'))

        # Generate POTCAR via vaspkit
        ensure_potcar(calc_dir)

        # U value info
        u_info = ""
        if 'LDAU' in params:
            u_info = f" [+U: {params.get('LDAUU', '')}]"

        print(f"  {compound}: {sum(counts)} atoms, species={species}, "
              f"LMAXMIX={params['LMAXMIX']}{u_info}")

        manifest.append({'compound': compound, 'dir': calc_dir,
                         'source': vasp_path, 'elements': parent,
                         'species': species, 'counts': counts})

    # Save manifest
    manifest_path = os.path.join(work_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {manifest_path} ({len(manifest)} compounds)")

    return manifest


def submit_bulk_jobs(work_dir, partition='cu', ntasks=64):
    """Submit SLURM jobs for all set-up bulk calculations."""
    manifest_path = os.path.join(work_dir, 'manifest.json')
    if not os.path.exists(manifest_path):
        print("No manifest.json found. Run 'setup' first.")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    submitted = 0
    skipped = 0
    for entry in manifest:
        calc_dir = entry['dir']

        if not os.path.exists(os.path.join(calc_dir, 'POSCAR')):
            continue

        # Skip if already converged
        converged, _ = check_vasp_converged(calc_dir)
        if converged:
            skipped += 1
            continue

        # Skip if job is running
        from utils.job_manager import read_saved_jobid, get_job_status
        jobid = read_saved_jobid(calc_dir)
        if jobid:
            status = get_job_status(jobid)
            if status in ('RUNNING', 'PENDING'):
                skipped += 1
                continue

        job_name = f"blk_{entry['compound'][:12]}"
        jobid = submit_job(calc_dir, job_name=job_name,
                           partition=partition, ntasks=ntasks)
        if jobid:
            print(f"  Submitted {entry['compound']}: job {jobid}")
            submitted += 1

    print(f"\nSubmitted: {submitted}, Skipped: {skipped}")


def show_status(work_dir):
    """Show status of all bulk calculations."""
    manifest_path = os.path.join(work_dir, 'manifest.json')
    if not os.path.exists(manifest_path):
        print("No manifest.json found. Run 'setup' first.")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    done = 0
    running = 0
    failed = 0
    pending = 0

    for entry in manifest:
        calc_dir = entry['dir']
        converged, msg = check_vasp_converged(calc_dir)
        compound = entry['compound']

        if converged:
            energy = parse_energy(calc_dir)
            print(f"  DONE     {compound:<35} E = {energy:.6f} eV")
            done += 1
        elif os.path.exists(os.path.join(calc_dir, 'OUTCAR')):
            from utils.job_manager import read_saved_jobid, get_job_status
            jobid = read_saved_jobid(calc_dir)
            status = get_job_status(jobid) if jobid else 'UNKNOWN'
            if status in ('RUNNING', 'PENDING'):
                print(f"  {status:<8} {compound:<35} job {jobid}")
                running += 1
            else:
                print(f"  FAILED   {compound:<35} {msg}")
                failed += 1
        else:
            print(f"  PENDING  {compound}")
            pending += 1

    print(f"\nTotal: {len(manifest)} | Done: {done} | Running: {running} | "
          f"Failed: {failed} | Pending: {pending}")


def main():
    parser = argparse.ArgumentParser(description="Step 1: Bulk structure relaxation")
    sub = parser.add_subparsers(dest='command')

    p_setup = sub.add_parser('setup', help="Set up calculation directories")
    p_setup.add_argument('--input_dir', required=True, help="Directory with .vasp structure files")
    p_setup.add_argument('--work_dir', default='./1_bulk', help="Working directory")
    # POTCAR is auto-generated by vaspkit -task 103 at submission time

    p_submit = sub.add_parser('submit', help="Submit SLURM jobs")
    p_submit.add_argument('--work_dir', default='./1_bulk')
    p_submit.add_argument('--partition', default='cu')
    p_submit.add_argument('--ntasks', type=int, default=64)

    p_status = sub.add_parser('status', help="Check calculation status")
    p_status.add_argument('--work_dir', default='./1_bulk')

    p_resub = sub.add_parser('resubmit', help="Resubmit failed jobs")
    p_resub.add_argument('--work_dir', default='./1_bulk')
    p_resub.add_argument('--partition', default='cu')
    p_resub.add_argument('--ntasks', type=int, default=64)

    args = parser.parse_args()

    if args.command == 'setup':
        setup_bulk_calculations(args.input_dir, args.work_dir)
    elif args.command == 'submit':
        submit_bulk_jobs(args.work_dir, args.partition, args.ntasks)
    elif args.command == 'status':
        show_status(args.work_dir)
    elif args.command == 'resubmit':
        submit_bulk_jobs(args.work_dir, args.partition, args.ntasks)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
