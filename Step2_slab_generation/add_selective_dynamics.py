#!/usr/bin/env python3
"""
Add selective dynamics to POSCAR slab files.

Strategy: relax atoms in the top and bottom portions of the slab by
Cartesian thickness (Angstrom or percentage), fix the middle region.
Default: 25% per side -> 25/50/25 split (works well for ~20 Ang slabs).

Single file mode:
    python3 add_selective_dynamics.py POSCAR              # 25% default
    python3 add_selective_dynamics.py POSCAR 5.0          # 5 Ang per side
    python3 add_selective_dynamics.py POSCAR 25%          # 25% per side

Batch mode (process all surf_*/POSCAR):
    python3 add_selective_dynamics.py --batch             # 25% default
    python3 add_selective_dynamics.py --batch 5.0         # 5 Ang per side
    python3 add_selective_dynamics.py --batch 25%         # 25% per side
    python3 add_selective_dynamics.py --batch 25% --pattern "surf_*/POSCAR"
    python3 add_selective_dynamics.py --batch 25% --overwrite   # overwrite POSCAR in place
"""

import sys
import glob
import os
import argparse
import numpy as np


def read_poscar(filepath="POSCAR"):
    with open(filepath, "r") as f:
        lines = f.readlines()

    comment = lines[0].strip()
    scale = float(lines[1].strip())
    lattice = []
    lattice_vectors = []
    for i in range(2, 5):
        lattice.append(lines[i].strip())
        lattice_vectors.append([float(x) for x in lines[i].split()])

    species = lines[5].split()
    counts = list(map(int, lines[6].split()))
    total_atoms = sum(counts)

    idx = 7
    coord_type = lines[idx].strip()

    positions = []
    labels = []
    for i in range(idx + 1, idx + 1 + total_atoms):
        parts = lines[i].split()
        positions.append([float(parts[0]), float(parts[1]), float(parts[2])])
        label = " ".join(parts[3:]) if len(parts) > 3 else ""
        labels.append(label)

    return {
        "comment": comment,
        "scale": scale,
        "lattice": lattice,
        "lattice_vectors": lattice_vectors,
        "species": species,
        "counts": counts,
        "coord_type": coord_type,
        "positions": positions,
        "labels": labels,
        "total_atoms": total_atoms,
    }


def write_poscar_selective(data, flags, filepath="POSCAR_selective"):
    with open(filepath, "w") as f:
        f.write(data["comment"] + "\n")
        f.write(f"  {data['scale']}\n")
        for lat in data["lattice"]:
            f.write(f"  {lat}\n")
        f.write("  " + "  ".join(data["species"]) + "\n")
        f.write("  " + "  ".join(map(str, data["counts"])) + "\n")
        f.write("Selective dynamics\n")
        f.write(data["coord_type"] + "\n")
        for i, pos in enumerate(data["positions"]):
            flag_str = flags[i]
            label = f"  {data['labels'][i]}" if data["labels"][i] else ""
            f.write(
                f"  {pos[0]:.16f}  {pos[1]:.16f}  {pos[2]:.16f}  {flag_str}{label}\n"
            )


def parse_relax_input(relax_input, slab_thickness):
    if relax_input.endswith("%"):
        pct = float(relax_input.strip("%"))
        relax_thickness = slab_thickness * pct / 100.0
        return relax_thickness, pct
    else:
        relax_thickness = float(relax_input)
        pct = relax_thickness / slab_thickness * 100.0
        return relax_thickness, pct


def process_single(poscar_path, relax_input, output_path, verbose=True):
    data = read_poscar(poscar_path)

    c_z = data["lattice_vectors"][2][2] * data["scale"]
    z_cart = np.array([p[2] * c_z for p in data["positions"]])

    z_min = z_cart.min()
    z_max = z_cart.max()
    slab_thickness = z_max - z_min

    relax_thickness, pct = parse_relax_input(relax_input, slab_thickness)

    bot_cutoff = z_min + relax_thickness
    top_cutoff = z_max - relax_thickness
    fixed_thickness = top_cutoff - bot_cutoff

    flags = []
    for z in z_cart:
        if z <= bot_cutoff or z >= top_cutoff:
            flags.append("T T T")
        else:
            flags.append("F F F")

    relaxed = flags.count("T T T")
    fixed = flags.count("F F F")

    if verbose:
        print(f"  Slab thickness:  {slab_thickness:.2f} Ang")
        print(f"  Relax per side:  {relax_thickness:.2f} Ang ({pct:.1f}%)")
        print(f"  Fixed middle:    {fixed_thickness:.2f} Ang ({fixed_thickness/slab_thickness*100:.1f}%)")
        print(f"  Atoms:           {relaxed} relaxed / {fixed} fixed / {data['total_atoms']} total")

        sorted_idx = np.argsort(z_cart)
        species_per_atom = []
        for sp, cnt in zip(data["species"], data["counts"]):
            species_per_atom.extend([sp] * cnt)

        print(f"\n  {'#':>4}  {'Elem':>5}  {'z(Ang)':>10}  {'Flag':>6}  Label")
        print("  " + "-" * 46)
        for idx in sorted_idx:
            elem = species_per_atom[idx]
            z = z_cart[idx]
            flag = flags[idx]
            label = data["labels"][idx]
            marker = ""
            if abs(z - bot_cutoff) < 0.5 or abs(z - top_cutoff) < 0.5:
                marker = " <-- near boundary"
            print(f"  {idx:>4}  {elem:>5}  {z:>10.4f}  {flag:>6}  {label}{marker}")

    write_poscar_selective(data, flags, output_path)

    return {
        "path": poscar_path,
        "output": output_path,
        "total": data["total_atoms"],
        "relaxed": relaxed,
        "fixed": fixed,
        "slab_thickness": slab_thickness,
        "relax_thickness": relax_thickness,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Add selective dynamics to POSCAR slab files."
    )
    parser.add_argument(
        "poscar", nargs="?", default="POSCAR",
        help="Input POSCAR file (single mode, default: POSCAR)"
    )
    parser.add_argument(
        "relax", nargs="?", default="25%%",
        help="Relax thickness per side: e.g. '5.0' (Ang) or '25%%' (default: 25%%)"
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Batch mode: process all matching POSCAR files"
    )
    parser.add_argument(
        "--pattern", default="surf_*/POSCAR",
        help="Glob pattern for batch mode (default: surf_*/POSCAR)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite original POSCAR (backup saved as POSCAR_orig)"
    )

    args = parser.parse_args()
    relax_input = args.relax.replace("%%", "%")

    if args.batch:
        poscar_files = sorted(glob.glob(args.pattern))

        if not poscar_files:
            print(f"No files found matching pattern: {args.pattern}")
            print("Make sure you're in the parent directory containing surf_* folders.")
            sys.exit(1)

        print(f"Batch mode: found {len(poscar_files)} POSCAR files")
        print(f"Relax setting: {relax_input} per side")
        print(f"Overwrite: {'Yes (backup as POSCAR_orig)' if args.overwrite else 'No (save as POSCAR_selective)'}")
        print("=" * 70)

        summaries = []
        for poscar_path in poscar_files:
            folder = os.path.dirname(poscar_path)
            print(f"\n[{folder}]")

            if args.overwrite:
                backup_path = os.path.join(folder, "POSCAR_orig")
                if not os.path.exists(backup_path):
                    import shutil
                    shutil.copy2(poscar_path, backup_path)
                    print(f"  Backup: {backup_path}")
                output_path = poscar_path
            else:
                output_path = os.path.join(folder, "POSCAR_selective")

            try:
                summary = process_single(
                    poscar_path, relax_input, output_path, verbose=False
                )
                summaries.append(summary)
                print(f"  Slab: {summary['slab_thickness']:.2f} Ang | "
                      f"Relax: {summary['relax_thickness']:.2f} Ang/side | "
                      f"Atoms: {summary['relaxed']}T/{summary['fixed']}F/{summary['total']} total | "
                      f"-> {output_path}")
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        print("\n" + "=" * 70)
        print("BATCH SUMMARY")
        print("=" * 70)
        print(f"  {'Folder':<30}  {'Thick':>6}  {'Relax':>6}  {'T':>4}  {'F':>4}  {'Tot':>4}")
        print("  " + "-" * 62)
        for s in summaries:
            folder = os.path.dirname(s["path"])
            print(f"  {folder:<30}  {s['slab_thickness']:>5.1f}A  {s['relax_thickness']:>5.1f}A  "
                  f"{s['relaxed']:>4}  {s['fixed']:>4}  {s['total']:>4}")
        print(f"\n  Processed {len(summaries)}/{len(poscar_files)} files successfully.")

    else:
        print(f"Reading: {args.poscar}\n")

        if args.overwrite:
            backup_path = args.poscar + "_orig"
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(args.poscar, backup_path)
                print(f"Backup saved: {backup_path}")
            output_path = args.poscar
        else:
            output_path = "POSCAR_selective"

        summary = process_single(args.poscar, relax_input, output_path, verbose=True)
        print(f"\nWritten to: {output_path}")


if __name__ == "__main__":
    main()
