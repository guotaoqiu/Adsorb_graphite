#!/usr/bin/env python3
"""
Generate manifest.json for MACE slab results by scanning the directory tree.

Usage:
    python3 make_mace_manifest.py --mace_dir ./2_slabs_mace --bulk_dir ./1_bulk
    python3 make_mace_manifest.py --mace_dir ./2_slabs_mace --from_manifest ./2_slabs/manifest.json
"""

import os
import sys
import json
import glob
import argparse


def scan_mace_dirs(mace_dir):
    """Scan MACE output tree and build manifest from directory structure."""
    manifest = []

    # Structure: mace_dir / element_combo / compound / surf_*
    element_dirs = sorted(glob.glob(os.path.join(mace_dir, '*')))

    for elem_dir in element_dirs:
        if not os.path.isdir(elem_dir):
            continue
        elements = os.path.basename(elem_dir)

        compound_dirs = sorted(glob.glob(os.path.join(elem_dir, '*')))
        for comp_dir in compound_dirs:
            if not os.path.isdir(comp_dir):
                continue
            compound = os.path.basename(comp_dir)

            slab_dirs = sorted(glob.glob(os.path.join(comp_dir, 'surf_*')))
            slab_dirs = [d for d in slab_dirs if os.path.isdir(d)]

            if not slab_dirs:
                continue

            # Check how many have converged
            converged = 0
            for sd in slab_dirs:
                ej = os.path.join(sd, 'energy.json')
                if os.path.exists(ej):
                    with open(ej) as f:
                        info = json.load(f)
                    if info.get('converged', False):
                        converged += 1

            manifest.append({
                'compound': compound,
                'elements': elements,
                'bulk_dir': '',  # will be filled if --bulk_dir provided
                'slab_dirs': slab_dirs,
                'n_converged': converged,
                'n_total': len(slab_dirs),
            })

    return manifest


def from_existing_manifest(src_manifest_path, mace_dir, src_dir):
    """Create MACE manifest by remapping paths from existing DFT manifest."""
    with open(src_manifest_path) as f:
        src = json.load(f)

    manifest = []
    for entry in src:
        new_entry = dict(entry)
        # Remap slab_dirs
        new_slab_dirs = []
        for sd in entry['slab_dirs']:
            rel = os.path.relpath(sd, src_dir)
            new_sd = os.path.join(mace_dir, rel)
            if os.path.isdir(new_sd):
                new_slab_dirs.append(new_sd)

        if new_slab_dirs:
            new_entry['slab_dirs'] = new_slab_dirs
            manifest.append(new_entry)

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Generate manifest.json for MACE slab results."
    )
    parser.add_argument('--mace_dir', required=True, help="MACE results directory")
    parser.add_argument('--bulk_dir', default='./1_bulk',
                        help="Bulk calculation directory")
    parser.add_argument('--from_manifest', default=None,
                        help="Copy structure from existing DFT manifest.json")
    parser.add_argument('--dft_dir', default='./2_slabs',
                        help="DFT slab directory (for path remapping with --from_manifest)")

    args = parser.parse_args()

    if args.from_manifest and os.path.exists(args.from_manifest):
        print(f"Remapping from: {args.from_manifest}")
        manifest = from_existing_manifest(args.from_manifest, args.mace_dir, args.dft_dir)
    else:
        print(f"Scanning: {args.mace_dir}")
        manifest = scan_mace_dirs(args.mace_dir)

    # Fill in bulk_dir from bulk manifest
    bulk_manifest_path = os.path.join(args.bulk_dir, 'manifest.json')
    if os.path.exists(bulk_manifest_path):
        with open(bulk_manifest_path) as f:
            bulk_manifest = json.load(f)
        bulk_lookup = {e['compound']: e['dir'] for e in bulk_manifest}
        for entry in manifest:
            entry['bulk_dir'] = bulk_lookup.get(entry['compound'], '')

    # Save
    output_path = os.path.join(args.mace_dir, 'manifest.json')
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    total_slabs = sum(len(e['slab_dirs']) for e in manifest)
    total_converged = sum(e.get('n_converged', 0) for e in manifest)
    print(f"\nManifest saved: {output_path}")
    print(f"  Compounds: {len(manifest)}")
    print(f"  Slab dirs: {total_slabs}")
    if total_converged:
        print(f"  Converged: {total_converged}/{total_slabs}")


if __name__ == "__main__":
    main()
