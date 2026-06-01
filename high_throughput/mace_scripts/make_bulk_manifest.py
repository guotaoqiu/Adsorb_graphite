#!/usr/bin/env python3
"""
Generate manifest.json for MACE bulk results by scanning the directory tree.

Bulk structure: mace_dir / element_combo / compound / (POSCAR, CONTCAR, energy.json)

Usage:
    python3 make_bulk_manifest.py --mace_dir ./1_bulk_mace
"""

import os
import sys
import json
import glob
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Generate manifest.json for MACE bulk results."
    )
    parser.add_argument('--mace_dir', required=True)

    args = parser.parse_args()

    manifest = []

    # Structure: mace_dir / element_combo / compound /
    for elem_dir in sorted(glob.glob(os.path.join(args.mace_dir, '*'))):
        if not os.path.isdir(elem_dir):
            continue
        elements = os.path.basename(elem_dir)

        for comp_dir in sorted(glob.glob(os.path.join(elem_dir, '*'))):
            if not os.path.isdir(comp_dir):
                continue

            # Check if this dir has results
            has_energy = os.path.exists(os.path.join(comp_dir, 'energy.json'))
            has_contcar = os.path.exists(os.path.join(comp_dir, 'CONTCAR'))

            if not (has_energy or has_contcar):
                continue

            compound = os.path.basename(comp_dir)
            manifest.append({
                'compound': compound,
                'dir': comp_dir,
                'elements': elements,
            })

    output_path = os.path.join(args.mace_dir, 'manifest.json')
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest saved: {output_path}")
    print(f"  Compounds: {len(manifest)}")


if __name__ == "__main__":
    main()
