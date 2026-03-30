#!/usr/bin/env python3
"""
Master workflow orchestration for graphite adsorption on carbide surfaces.

This script ties together all steps of the high-throughput workflow:

  Step 1: Bulk optimization (via mpjob - external)
  Step 2: Slab generation (surfaxe)
  Step 3: Surface energy calculation -> identify most stable surface
  Step 4: Generate adsorption configurations (ASE)
  Step 5: Calculate adsorption energies

Usage:
    # Check status of all calculations
    python3 run_workflow.py status --compound CaBC2

    # Run Step 3: calculate surface energies after slab relaxations finish
    python3 run_workflow.py surface_energy --compound CaBC2 --bulk_outcar bulk/OUTCAR --bulk_natoms 16

    # Run Step 4: generate adsorption configs on most stable surface
    python3 run_workflow.py gen_adsorption --compound CaBC2 --surface surf_001_term_0 --adsorbate all

    # Run Step 5: calculate adsorption energies after all ads calcs finish
    python3 run_workflow.py ads_energy --compound CaBC2 --surface surf_001_term_0 --c_energy -1.36

    # Full pipeline summary
    python3 run_workflow.py summary --compound CaBC2
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.vasp_utils import (
    check_vasp_done, parse_outcar_energy, read_poscar_atoms,
    get_surface_area, scan_job_status
)


COMPOUNDS = ['CaBC2', 'YBC2', 'LaBC2', 'CeBC2', 'PrBC2']


def cmd_status(args):
    """Check status of all calculations for a compound."""
    compound = args.compound
    base = os.path.join(args.workdir, compound)

    if not os.path.isdir(base):
        print(f"Directory not found: {base}")
        return

    print(f"{'=' * 70}")
    print(f"STATUS: {compound}")
    print(f"{'=' * 70}")

    # Check bulk
    bulk_dir = os.path.join(base, 'bulk')
    if os.path.isdir(bulk_dir):
        done, msg = check_vasp_done(bulk_dir)
        print(f"\n[Bulk] {msg}")
    else:
        print(f"\n[Bulk] Not set up")

    # Check surfaces
    surf_dirs = sorted(glob.glob(os.path.join(base, 'surf_*')))
    if surf_dirs:
        print(f"\n[Surfaces] {len(surf_dirs)} surface directories")
        for sd in surf_dirs:
            if os.path.exists(os.path.join(sd, 'OUTCAR')):
                done, msg = check_vasp_done(sd)
                status = "DONE" if done else "INCOMPLETE"
            elif os.path.exists(os.path.join(sd, 'POSCAR')):
                status, msg = "PENDING", "Waiting for submission"
            else:
                status, msg = "EMPTY", "No input files"
            print(f"  {os.path.basename(sd):<30} {status:<12} {msg}")
    else:
        print(f"\n[Surfaces] Not generated yet")

    # Check adsorption
    for sd in surf_dirs:
        ads_dirs = sorted(glob.glob(os.path.join(sd, 'ads_*')))
        if ads_dirs:
            print(f"\n[Adsorption on {os.path.basename(sd)}] {len(ads_dirs)} configurations")
            for ad in ads_dirs[:5]:  # Show first 5
                if os.path.exists(os.path.join(ad, 'OUTCAR')):
                    done, msg = check_vasp_done(ad)
                    status = "DONE" if done else "INCOMPLETE"
                else:
                    status = "PENDING"
                    msg = ""
                print(f"    {os.path.basename(ad):<35} {status:<12} {msg}")
            if len(ads_dirs) > 5:
                print(f"    ... and {len(ads_dirs) - 5} more")


def cmd_surface_energy(args):
    """Calculate surface energies and identify the most stable surface."""
    from Step3_surface_energy.calc_surface_energy import process_surface, parse_outcar_energy

    compound = args.compound
    base = os.path.join(args.workdir, compound)

    # Get bulk energy
    if args.bulk_outcar:
        e_bulk = parse_outcar_energy(args.bulk_outcar)
    elif args.bulk_energy:
        e_bulk = args.bulk_energy
    else:
        # Try to find bulk OUTCAR
        bulk_outcar = os.path.join(base, 'bulk', 'OUTCAR')
        if os.path.exists(bulk_outcar):
            e_bulk = parse_outcar_energy(bulk_outcar)
        else:
            print("ERROR: No bulk energy provided. Use --bulk_outcar or --bulk_energy")
            return

    e_bulk_per_atom = e_bulk / args.bulk_natoms
    print(f"Bulk energy per atom: {e_bulk_per_atom:.6f} eV")

    # Process all surfaces
    surf_dirs = sorted(glob.glob(os.path.join(base, 'surf_*')))
    results = []
    for sd in surf_dirs:
        try:
            result = process_surface(sd, e_bulk_per_atom)
            results.append(result)
        except Exception as e:
            print(f"  ERROR in {sd}: {e}")

    if results:
        results.sort(key=lambda x: x['gamma_J_m2'])
        output_file = os.path.join(base, 'surface_energies.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
        print(f"Most stable: {results[0]['directory']} (gamma = {results[0]['gamma_J_m2']:.4f} J/m^2)")


def cmd_gen_adsorption(args):
    """Generate adsorption configurations on a specific surface."""
    from Step4_adsorption_sites.generate_adsorption import process_slab

    compound = args.compound
    base = os.path.join(args.workdir, compound)

    if args.surface:
        slab_path = os.path.join(base, args.surface, 'CONTCAR')
        if not os.path.exists(slab_path):
            slab_path = os.path.join(base, args.surface, 'POSCAR')
    else:
        # Use most stable surface from surface_energies.json
        se_file = os.path.join(base, 'surface_energies.json')
        if os.path.exists(se_file):
            with open(se_file) as f:
                se = json.load(f)
            best_dir = se[0]['directory']
            slab_path = os.path.join(best_dir, 'CONTCAR')
            print(f"Using most stable surface: {best_dir}")
        else:
            print("ERROR: No --surface specified and no surface_energies.json found")
            return

    incar_template = os.path.join(os.path.dirname(__file__), 'templates', 'INCAR_relax_template')

    process_slab(
        slab_path, args.adsorbate,
        height=args.height,
        chain_length=args.chain_length,
        ring_size=args.ring_size,
        max_strain=args.max_strain,
        incar_template=incar_template if os.path.exists(incar_template) else None,
        rotation_angles=args.rotation_angles,
    )


def cmd_ads_energy(args):
    """Calculate adsorption energies."""
    from Step5_adsorption_energy.calc_adsorption_energy import (
        get_energy, count_adsorbate_C, get_natoms_and_species,
        calc_adsorption_energy, get_surface_area
    )

    compound = args.compound
    base = os.path.join(args.workdir, compound)
    surf_dir = os.path.join(base, args.surface) if args.surface else base

    # Clean slab energy
    if args.slab_energy:
        e_slab = args.slab_energy
    else:
        e_slab = get_energy(surf_dir)

    slab_species, slab_counts = get_natoms_and_species(surf_dir)

    print(f"Clean slab: E = {e_slab:.6f} eV")
    print(f"C reference: {args.c_energy:.6f} eV/atom")

    ads_dirs = sorted(glob.glob(os.path.join(surf_dir, 'ads_*')))
    results = []

    for ad in ads_dirs:
        try:
            e_total = get_energy(ad)
            n_C = count_adsorbate_C(ad, slab_species, slab_counts)
            area = get_surface_area(ad) if args.per_area else None
            e_ads = calc_adsorption_energy(e_total, e_slab, n_C, args.c_energy,
                                            area, args.per_area)
            results.append({
                'config': os.path.basename(ad),
                'e_ads': e_ads,
                'n_C': n_C,
            })
            print(f"  {os.path.basename(ad):<40} n_C={n_C:>3}  E_ads={e_ads:>10.4f} eV")
        except Exception as e:
            print(f"  {os.path.basename(ad):<40} ERROR: {e}")

    if results:
        results.sort(key=lambda x: x['e_ads'])
        output_file = os.path.join(surf_dir, 'adsorption_energies.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nMost stable: {results[0]['config']} (E_ads = {results[0]['e_ads']:.4f} eV)")


def cmd_summary(args):
    """Print a full pipeline summary for all compounds."""
    compounds = [args.compound] if args.compound else COMPOUNDS

    print(f"{'=' * 80}")
    print(f"WORKFLOW SUMMARY - Graphite Adsorption on Carbide Surfaces")
    print(f"{'=' * 80}")

    for compound in compounds:
        base = os.path.join(args.workdir, compound)
        if not os.path.isdir(base):
            continue

        print(f"\n{'─' * 40}")
        print(f"  {compound}")
        print(f"{'─' * 40}")

        # Surface energies
        se_file = os.path.join(base, 'surface_energies.json')
        if os.path.exists(se_file):
            with open(se_file) as f:
                se = json.load(f)
            print(f"  Most stable surface: {se[0]['hkl']} (gamma = {se[0]['gamma_J_m2']:.4f} J/m^2)")
        else:
            print(f"  Surface energies: not calculated")

        # Adsorption energies
        for surf_dir in sorted(glob.glob(os.path.join(base, 'surf_*'))):
            ae_file = os.path.join(surf_dir, 'adsorption_energies.json')
            if os.path.exists(ae_file):
                with open(ae_file) as f:
                    ae = json.load(f)
                print(f"  Adsorption on {os.path.basename(surf_dir)}:")
                for r in ae[:3]:
                    print(f"    {r['config']:<35} E_ads = {r['e_ads']:.4f} eV")


def main():
    parser = argparse.ArgumentParser(
        description="Workflow orchestration for graphite adsorption on carbide surfaces."
    )
    parser.add_argument('--workdir', default='.', help="Working directory containing compound folders")

    sub = parser.add_subparsers(dest='command')

    # status
    p_status = sub.add_parser('status', help="Check calculation status")
    p_status.add_argument('--compound', required=True)

    # surface_energy
    p_se = sub.add_parser('surface_energy', help="Calculate surface energies")
    p_se.add_argument('--compound', required=True)
    p_se.add_argument('--bulk_outcar', default=None)
    p_se.add_argument('--bulk_energy', type=float, default=None)
    p_se.add_argument('--bulk_natoms', type=int, required=True)

    # gen_adsorption
    p_ga = sub.add_parser('gen_adsorption', help="Generate adsorption configurations")
    p_ga.add_argument('--compound', required=True)
    p_ga.add_argument('--surface', default=None, help="Surface directory name (e.g. surf_001_term_0)")
    p_ga.add_argument('--adsorbate', required=True,
                      choices=['single_C', 'C_chain_v', 'C_chain_h', 'C_ring', 'C_ring_v', 'graphene', 'all'])
    p_ga.add_argument('--height', type=float, default=2.0)
    p_ga.add_argument('--chain_length', type=int, default=3)
    p_ga.add_argument('--ring_size', type=int, default=6)
    p_ga.add_argument('--max_strain', type=float, default=5.0)
    p_ga.add_argument('--rotation_angles', type=float, nargs='*', default=None)

    # ads_energy
    p_ae = sub.add_parser('ads_energy', help="Calculate adsorption energies")
    p_ae.add_argument('--compound', required=True)
    p_ae.add_argument('--surface', default=None)
    p_ae.add_argument('--slab_energy', type=float, default=None)
    p_ae.add_argument('--c_energy', type=float, required=True)
    p_ae.add_argument('--per_area', action='store_true')

    # summary
    p_sum = sub.add_parser('summary', help="Full pipeline summary")
    p_sum.add_argument('--compound', default=None)

    args = parser.parse_args()

    if args.command == 'status':
        cmd_status(args)
    elif args.command == 'surface_energy':
        cmd_surface_energy(args)
    elif args.command == 'gen_adsorption':
        cmd_gen_adsorption(args)
    elif args.command == 'ads_energy':
        cmd_ads_energy(args)
    elif args.command == 'summary':
        cmd_summary(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
