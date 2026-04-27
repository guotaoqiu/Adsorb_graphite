#!/usr/bin/env python3
"""
Master orchestrator for graphitization catalyst screening.

Runs the full pipeline:
  Step 1: Bulk relaxation
  Step 2: Slab generation + relaxation
  Step 3: Surface energy calculation
  Step 4: Adsorption generation + relaxation + energy calculation

Each step can be run independently or as part of the pipeline.
The 'auto' command monitors job completion and advances to the next step.

Usage:
    # Full pipeline setup
    python3 run_screening.py setup --input_dir /path/to/structures

    # Submit all ready jobs
    python3 run_screening.py submit

    # Check overall status
    python3 run_screening.py status

    # Run individual steps
    python3 run_screening.py step1 setup --input_dir /path/to/structures
    python3 run_screening.py step1 submit
    python3 run_screening.py step2 setup
    python3 run_screening.py step3
    python3 run_screening.py step4 setup
    python3 run_screening.py step4 energy

    # Advance: check what's done and set up the next step
    python3 run_screening.py advance
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_setup(args):
    """Set up the full pipeline."""
    from step1_bulk_relax import setup_bulk_calculations
    print("=" * 70)
    print("STEP 1: Setting up bulk relaxations")
    print("=" * 70)
    setup_bulk_calculations(args.input_dir, args.bulk_dir, args.potcar_dir)

    print(f"\nNext: run 'python3 run_screening.py submit' to submit jobs")


def cmd_submit(args):
    """Submit all ready jobs across all steps."""
    from step1_bulk_relax import submit_bulk_jobs
    from step2_slab_generation import submit_slab_jobs
    from step4_adsorption import submit_ads_jobs

    if os.path.exists(os.path.join(args.bulk_dir, 'manifest.json')):
        print("── Step 1: Bulk jobs ──")
        submit_bulk_jobs(args.bulk_dir, args.partition, args.ntasks)

    if os.path.exists(os.path.join(args.slab_dir, 'manifest.json')):
        print("\n── Step 2: Slab jobs ──")
        submit_slab_jobs(args.slab_dir, args.partition, args.ntasks)

    if os.path.exists(os.path.join(args.ads_dir, 'manifest.json')):
        print("\n── Step 4: Adsorption jobs ──")
        submit_ads_jobs(args.ads_dir, args.partition, args.ntasks)


def cmd_status(args):
    """Show status across all steps."""
    from utils.job_manager import check_vasp_converged

    print("=" * 70)
    print("SCREENING STATUS")
    print("=" * 70)

    # Step 1
    bulk_manifest = os.path.join(args.bulk_dir, 'manifest.json')
    if os.path.exists(bulk_manifest):
        with open(bulk_manifest) as f:
            m = json.load(f)
        done = sum(1 for e in m if check_vasp_converged(e['dir'])[0])
        print(f"\n  Step 1 (Bulk):        {done}/{len(m)} converged")
    else:
        print(f"\n  Step 1 (Bulk):        not set up")

    # Step 2
    slab_manifest = os.path.join(args.slab_dir, 'manifest.json')
    if os.path.exists(slab_manifest):
        with open(slab_manifest) as f:
            m = json.load(f)
        total = sum(len(e['slab_dirs']) for e in m)
        done = sum(1 for e in m for sd in e['slab_dirs']
                   if check_vasp_converged(sd)[0])
        print(f"  Step 2 (Slabs):       {done}/{total} converged")
    else:
        print(f"  Step 2 (Slabs):       not set up")

    # Step 3
    se_path = os.path.join(args.slab_dir, 'surface_energies.json')
    if os.path.exists(se_path):
        with open(se_path) as f:
            se = json.load(f)
        print(f"  Step 3 (Surf Energy): {len(se)} compounds analyzed")
    else:
        print(f"  Step 3 (Surf Energy): not calculated")

    # Step 4
    ads_manifest = os.path.join(args.ads_dir, 'manifest.json')
    if os.path.exists(ads_manifest):
        with open(ads_manifest) as f:
            m = json.load(f)
        total = sum(len(e['ads_dirs']) for e in m)
        done = sum(1 for e in m for a in e['ads_dirs']
                   if check_vasp_converged(a['dir'])[0])
        print(f"  Step 4 (Adsorption):  {done}/{total} converged")
    else:
        print(f"  Step 4 (Adsorption):  not set up")

    # Final results
    ae_path = os.path.join(args.ads_dir, 'adsorption_energies.json')
    if os.path.exists(ae_path):
        with open(ae_path) as f:
            ae = json.load(f)
        print(f"\n  Final results:        {len(ae)} compounds with adsorption energies")


def cmd_advance(args):
    """Check completed steps and advance to the next one."""
    from utils.job_manager import check_vasp_converged

    # Check Step 1
    bulk_manifest = os.path.join(args.bulk_dir, 'manifest.json')
    if not os.path.exists(bulk_manifest):
        print("Step 1 not set up. Run: python3 run_screening.py setup --input_dir /path/to/structures")
        return

    with open(bulk_manifest) as f:
        bulk_m = json.load(f)
    bulk_done = sum(1 for e in bulk_m if check_vasp_converged(e['dir'])[0])

    if bulk_done == 0:
        print(f"Step 1: 0/{len(bulk_m)} done. Submit jobs: python3 run_screening.py submit")
        return

    # Check if Step 2 needs setup
    slab_manifest = os.path.join(args.slab_dir, 'manifest.json')
    if not os.path.exists(slab_manifest) and bulk_done > 0:
        print(f"Step 1: {bulk_done}/{len(bulk_m)} done. Setting up Step 2...")
        from step2_slab_generation import setup_slab_calculations
        setup_slab_calculations(args.bulk_dir, args.slab_dir)
        print("\nNext: python3 run_screening.py submit")
        return

    # Check Step 2 completion
    if os.path.exists(slab_manifest):
        with open(slab_manifest) as f:
            slab_m = json.load(f)
        slab_total = sum(len(e['slab_dirs']) for e in slab_m)
        slab_done = sum(1 for e in slab_m for sd in e['slab_dirs']
                        if check_vasp_converged(sd)[0])

        if slab_done < slab_total and slab_done > 0:
            # Check if Step 3 has been run
            se_path = os.path.join(args.slab_dir, 'surface_energies.json')
            if not os.path.exists(se_path):
                print(f"Step 2: {slab_done}/{slab_total} done. Running Step 3...")
                from step3_surface_energy import calculate_surface_energies
                calculate_surface_energies(args.slab_dir, args.bulk_dir)

        # Check Step 3 -> Step 4
        se_path = os.path.join(args.slab_dir, 'surface_energies.json')
        ads_manifest = os.path.join(args.ads_dir, 'manifest.json')
        if os.path.exists(se_path) and not os.path.exists(ads_manifest):
            print(f"Step 3 done. Setting up Step 4 (adsorption)...")
            from step4_adsorption import setup_adsorption
            setup_adsorption(args.slab_dir, args.ads_dir)
            print("\nNext: python3 run_screening.py submit")
            return

    # Check Step 4 completion
    ads_manifest = os.path.join(args.ads_dir, 'manifest.json')
    if os.path.exists(ads_manifest):
        with open(ads_manifest) as f:
            ads_m = json.load(f)
        ads_total = sum(len(e['ads_dirs']) for e in ads_m)
        ads_done = sum(1 for e in ads_m for a in e['ads_dirs']
                       if check_vasp_converged(a['dir'])[0])

        if ads_done > 0:
            print(f"Step 4: {ads_done}/{ads_total} done. Calculating energies...")
            from step4_adsorption import calculate_adsorption_energies
            calculate_adsorption_energies(args.ads_dir)

    print("\nRun 'python3 run_screening.py status' for overview")


def main():
    parser = argparse.ArgumentParser(
        description="Graphitization catalyst screening orchestrator"
    )
    parser.add_argument('--bulk_dir', default='./1_bulk')
    parser.add_argument('--slab_dir', default='./2_slabs')
    parser.add_argument('--ads_dir', default='./3_adsorption')
    parser.add_argument('--partition', default='cu')
    parser.add_argument('--ntasks', type=int, default=64)
    parser.add_argument('--potcar_dir', default=None)

    sub = parser.add_subparsers(dest='command')

    p_setup = sub.add_parser('setup', help="Set up Step 1 (bulk relaxation)")
    p_setup.add_argument('--input_dir', required=True)

    sub.add_parser('submit', help="Submit all ready jobs")
    sub.add_parser('status', help="Show pipeline status")
    sub.add_parser('advance', help="Advance to next step based on completion")

    # Step-specific subcommands
    p_s1 = sub.add_parser('step1', help="Step 1 operations")
    p_s1.add_argument('action', choices=['setup', 'submit', 'status', 'resubmit'])
    p_s1.add_argument('--input_dir', default=None)

    p_s2 = sub.add_parser('step2', help="Step 2 operations")
    p_s2.add_argument('action', choices=['setup', 'submit', 'status'])

    p_s3 = sub.add_parser('step3', help="Step 3: calculate surface energies")

    p_s4 = sub.add_parser('step4', help="Step 4 operations")
    p_s4.add_argument('action', choices=['setup', 'submit', 'status', 'energy'])

    args = parser.parse_args()

    if args.command == 'setup':
        cmd_setup(args)
    elif args.command == 'submit':
        cmd_submit(args)
    elif args.command == 'status':
        cmd_status(args)
    elif args.command == 'advance':
        cmd_advance(args)
    elif args.command == 'step1':
        from step1_bulk_relax import (setup_bulk_calculations, submit_bulk_jobs,
                                       show_status)
        if args.action == 'setup':
            if not args.input_dir:
                print("--input_dir required for step1 setup")
                return
            setup_bulk_calculations(args.input_dir, args.bulk_dir, args.potcar_dir)
        elif args.action in ('submit', 'resubmit'):
            submit_bulk_jobs(args.bulk_dir, args.partition, args.ntasks)
        elif args.action == 'status':
            show_status(args.bulk_dir)
    elif args.command == 'step2':
        from step2_slab_generation import (setup_slab_calculations,
                                            submit_slab_jobs)
        if args.action == 'setup':
            setup_slab_calculations(args.bulk_dir, args.slab_dir)
        elif args.action == 'submit':
            submit_slab_jobs(args.slab_dir, args.partition, args.ntasks)
    elif args.command == 'step3':
        from step3_surface_energy import calculate_surface_energies
        calculate_surface_energies(args.slab_dir, args.bulk_dir)
    elif args.command == 'step4':
        from step4_adsorption import (setup_adsorption, submit_ads_jobs,
                                       calculate_adsorption_energies)
        if args.action == 'setup':
            setup_adsorption(args.slab_dir, args.ads_dir)
        elif args.action == 'submit':
            submit_ads_jobs(args.ads_dir, args.partition, args.ntasks)
        elif args.action == 'energy':
            calculate_adsorption_energies(args.ads_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
