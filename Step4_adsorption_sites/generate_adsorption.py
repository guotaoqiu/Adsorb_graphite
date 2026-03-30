#!/usr/bin/env python3
"""
Generate adsorption configurations for carbon species on slab surfaces using ASE.

Supported adsorbates:
  - single_C:    Single carbon atom
  - C_chain_v:   Carbon chain (vertical, perpendicular to surface)
  - C_chain_h:   Carbon chain (horizontal, parallel to surface)
  - C_ring:      Carbon ring (C6, benzene-like, horizontal)
  - C_ring_v:    Carbon ring (C6, vertical/tilted)
  - graphene:    Graphene monolayer (lattice-matched)

Usage:
    # Single carbon on all adsorption sites
    python3 generate_adsorption.py --slab CONTCAR --adsorbate single_C

    # Carbon chain (3 atoms, vertical) at height 2.0 Ang
    python3 generate_adsorption.py --slab CONTCAR --adsorbate C_chain_v --chain_length 3 --height 2.0

    # Graphene on slab (automatic lattice matching)
    python3 generate_adsorption.py --slab CONTCAR --adsorbate graphene --max_strain 5.0

    # All adsorbates on all sites
    python3 generate_adsorption.py --slab CONTCAR --adsorbate all

    # Batch mode: process multiple slab directories
    python3 generate_adsorption.py --batch --pattern "surf_*/CONTCAR" --adsorbate single_C
"""

import os
import argparse
import glob
import itertools
import numpy as np
from pathlib import Path

from ase.io import read, write
from ase import Atoms
from ase.build import add_adsorbate, molecule
from ase.build.surface import surface


def find_adsorption_sites(slab, symm_reduce=0.01):
    """
    Find unique adsorption sites on a slab using ASE's built-in analysis.

    Returns dict with site types: 'ontop', 'bridge', 'hollow', 'fourhold'
    Each entry is a list of (x, y) fractional positions.
    """
    from ase.geometry.analysis import Analysis

    # Get surface atoms (top layer)
    positions = slab.get_positions()
    z_coords = positions[:, 2]
    z_max = z_coords.max()
    # Surface atoms: within 1.5 Ang of the topmost atom
    surface_mask = z_coords >= (z_max - 1.5)
    surface_indices = np.where(surface_mask)[0]
    surface_pos = positions[surface_indices]

    sites = {}

    # Ontop sites: directly above each surface atom
    ontop_sites = []
    for i, idx in enumerate(surface_indices):
        pos = positions[idx]
        ontop_sites.append((pos[0], pos[1]))
    sites['ontop'] = _reduce_sites(ontop_sites, slab.cell, symm_reduce)

    # Bridge sites: midpoints between neighboring surface atoms
    bridge_sites = []
    for i in range(len(surface_indices)):
        for j in range(i + 1, len(surface_indices)):
            p1 = positions[surface_indices[i]]
            p2 = positions[surface_indices[j]]
            # Use minimum image convention for distance
            diff = p2[:2] - p1[:2]
            # Apply PBC
            frac1 = np.linalg.solve(slab.cell[:2, :2].T, diff)
            frac1 = frac1 - np.round(frac1)
            diff_pbc = slab.cell[:2, :2].T @ frac1
            dist = np.linalg.norm(diff_pbc)

            if dist < 4.0:  # Neighbors within 4 Ang
                mid = p1[:2] + diff_pbc / 2
                bridge_sites.append((mid[0], mid[1]))
    sites['bridge'] = _reduce_sites(bridge_sites, slab.cell, symm_reduce)

    # Hollow sites: centroids of triangles formed by 3 neighboring surface atoms
    hollow_sites = []
    for i in range(len(surface_indices)):
        for j in range(i + 1, len(surface_indices)):
            for k in range(j + 1, len(surface_indices)):
                pts = positions[surface_indices[[i, j, k]]]
                # Check all pairs are neighbors
                dists = []
                for a, b in [(0, 1), (0, 2), (1, 2)]:
                    diff = pts[b, :2] - pts[a, :2]
                    frac = np.linalg.solve(slab.cell[:2, :2].T, diff)
                    frac = frac - np.round(frac)
                    d = np.linalg.norm(slab.cell[:2, :2].T @ frac)
                    dists.append(d)
                if all(d < 4.0 for d in dists):
                    centroid = pts.mean(axis=0)
                    hollow_sites.append((centroid[0], centroid[1]))
    sites['hollow'] = _reduce_sites(hollow_sites, slab.cell, symm_reduce)

    return sites


def _reduce_sites(sites, cell, tol):
    """Remove symmetry-equivalent sites within tolerance."""
    if not sites:
        return []
    unique = [sites[0]]
    cell_2d = cell[:2, :2]
    for s in sites[1:]:
        is_dup = False
        for u in unique:
            diff = np.array(s) - np.array(u)
            frac = np.linalg.solve(cell_2d.T, diff)
            frac = frac - np.round(frac)
            cart_diff = cell_2d.T @ frac
            if np.linalg.norm(cart_diff) < tol:
                is_dup = True
                break
        if not is_dup:
            unique.append(s)
    return unique


def make_single_C():
    """Single carbon atom adsorbate."""
    return Atoms('C', positions=[(0, 0, 0)])


def make_C_chain(n=3, bond_length=1.3, vertical=True):
    """
    Carbon chain adsorbate.
    vertical=True: chain perpendicular to surface (z-direction)
    vertical=False: chain parallel to surface (x-direction)
    """
    positions = []
    for i in range(n):
        if vertical:
            positions.append([0, 0, i * bond_length])
        else:
            positions.append([i * bond_length, 0, 0])
    return Atoms(f'C{n}', positions=positions)


def make_C_ring(n=6, bond_length=1.4, vertical=False):
    """
    Carbon ring (C_n) adsorbate.
    vertical=False: ring parallel to surface (flat)
    vertical=True: ring perpendicular to surface (edge-on)
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = bond_length / (2 * np.sin(np.pi / n))

    positions = []
    for theta in angles:
        if vertical:
            # Ring in xz plane
            x = radius * np.cos(theta)
            y = 0
            z = radius * np.sin(theta)
        else:
            # Ring in xy plane (flat on surface)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = 0
        positions.append([x, y, z])

    return Atoms(f'C{n}', positions=positions)


def make_graphene_layer(slab, max_strain=5.0, rotation_angles=None):
    """
    Create a graphene layer matched to the slab surface.

    Strategy:
    1. Generate graphene with its natural lattice constant (2.46 Ang)
    2. Find the best supercell that matches the slab's a,b vectors
    3. Strain the graphene to match exactly

    Returns list of (graphene_atoms, strain_info) for each good match.
    """
    a_graphene = 2.46  # Graphene lattice constant in Angstrom

    # Graphene unit cell vectors
    g_a1 = np.array([a_graphene, 0, 0])
    g_a2 = np.array([a_graphene * 0.5, a_graphene * np.sqrt(3) / 2, 0])

    # Slab surface vectors
    s_a1 = slab.cell[0][:2]
    s_a2 = slab.cell[1][:2]

    # Search for graphene supercell that best matches slab
    best_matches = []
    max_n = 8  # Max supercell size to search

    for n1 in range(-max_n, max_n + 1):
        for n2 in range(-max_n, max_n + 1):
            if n1 == 0 and n2 == 0:
                continue
            for m1 in range(-max_n, max_n + 1):
                for m2 in range(-max_n, max_n + 1):
                    if m1 == 0 and m2 == 0:
                        continue

                    # Candidate graphene supercell vectors
                    ga = n1 * g_a1[:2] + n2 * g_a2[:2]
                    gb = m1 * g_a1[:2] + m2 * g_a2[:2]

                    # Check if these match the slab vectors
                    strain_a = np.linalg.norm(ga - s_a1) / np.linalg.norm(s_a1) * 100
                    strain_b = np.linalg.norm(gb - s_a2) / np.linalg.norm(s_a2) * 100

                    # Also check the angle
                    cos_g = np.dot(ga, gb) / (np.linalg.norm(ga) * np.linalg.norm(gb))
                    cos_s = np.dot(s_a1, s_a2) / (np.linalg.norm(s_a1) * np.linalg.norm(s_a2))
                    angle_diff = abs(np.arccos(np.clip(cos_g, -1, 1)) - np.arccos(np.clip(cos_s, -1, 1)))
                    angle_diff_deg = np.degrees(angle_diff)

                    avg_strain = (strain_a + strain_b) / 2
                    if avg_strain < max_strain and angle_diff_deg < 5.0:
                        # Compute number of graphene atoms
                        det = abs(n1 * m2 - n2 * m1)
                        if det > 0:
                            n_atoms = 2 * det  # 2 atoms per graphene unit cell
                            best_matches.append({
                                'n1': n1, 'n2': n2, 'm1': m1, 'm2': m2,
                                'strain_a': strain_a, 'strain_b': strain_b,
                                'avg_strain': avg_strain, 'angle_diff': angle_diff_deg,
                                'n_atoms': n_atoms, 'det': det,
                            })

    if not best_matches:
        print(f"  WARNING: No graphene match found within {max_strain}% strain")
        return []

    # Sort by strain, prefer smaller cells
    best_matches.sort(key=lambda x: (x['avg_strain'], x['n_atoms']))

    # Take top matches (up to 3)
    results = []
    for match in best_matches[:3]:
        # Build graphene supercell
        graphene_positions = []
        # Basis atoms in graphene unit cell (fractional)
        basis = [
            np.array([0.0, 0.0]),
            np.array([1.0 / 3.0, 1.0 / 3.0]),
        ]

        for i in range(abs(match['n1']) + abs(match['n2']) + abs(match['m1']) + abs(match['m2']) + 2):
            for j in range(abs(match['n1']) + abs(match['n2']) + abs(match['m1']) + abs(match['m2']) + 2):
                for b in basis:
                    # Position in graphene unit cell coordinates
                    pos_g = (i + b[0]) * g_a1[:2] + (j + b[1]) * g_a2[:2]

                    # Convert to slab fractional coordinates
                    try:
                        cell_2d = np.array([slab.cell[0][:2], slab.cell[1][:2]])
                        frac = np.linalg.solve(cell_2d.T, pos_g)
                        if 0 <= frac[0] < 1 - 1e-6 and 0 <= frac[1] < 1 - 1e-6:
                            graphene_positions.append([pos_g[0], pos_g[1], 0.0])
                    except np.linalg.LinAlgError:
                        continue

        if len(graphene_positions) == 0:
            continue

        graphene = Atoms(
            f'C{len(graphene_positions)}',
            positions=graphene_positions,
            cell=slab.cell.copy(),
            pbc=[True, True, False]
        )

        results.append((graphene, match))

    # Also add rotated graphene if requested
    if rotation_angles:
        base_graphene = results[0][0] if results else None
        if base_graphene:
            for angle in rotation_angles:
                rotated = base_graphene.copy()
                # Rotate around z-axis through center of mass
                com = rotated.get_center_of_mass()
                rotated.translate(-com)
                cos_a = np.cos(np.radians(angle))
                sin_a = np.sin(np.radians(angle))
                rot_matrix = np.array([[cos_a, -sin_a, 0],
                                        [sin_a, cos_a, 0],
                                        [0, 0, 1]])
                rotated.positions = rotated.positions @ rot_matrix.T
                rotated.translate(com)
                info = results[0][1].copy()
                info['rotation'] = angle
                results.append((rotated, info))

    return results


def generate_adsorption_configs(slab, adsorbate_type, height=2.0,
                                 chain_length=3, ring_size=6,
                                 max_strain=5.0, rotation_angles=None):
    """
    Generate all adsorption configurations for a given adsorbate type.

    Returns list of (slab_with_adsorbate, config_name) tuples.
    """
    configs = []

    if adsorbate_type == 'graphene':
        # Graphene needs special handling - no site enumeration
        matches = make_graphene_layer(slab, max_strain, rotation_angles)
        for graphene, match_info in matches:
            # Place graphene above the slab
            slab_copy = slab.copy()
            z_top = slab_copy.positions[:, 2].max()

            graphene_copy = graphene.copy()
            graphene_copy.positions[:, 2] = z_top + height

            # Combine slab + graphene
            combined = slab_copy + graphene_copy
            combined.cell = slab_copy.cell
            combined.pbc = slab_copy.pbc

            rot_str = f"_rot{match_info.get('rotation', 0)}" if 'rotation' in match_info else ""
            name = f"graphene_strain{match_info['avg_strain']:.1f}pct{rot_str}"
            configs.append((combined, name))

        return configs

    # For molecular adsorbates, enumerate adsorption sites
    sites = find_adsorption_sites(slab)

    adsorbate_makers = {
        'single_C': [('C', make_single_C())],
        'C_chain_v': [(f'Cchain{chain_length}v', make_C_chain(chain_length, vertical=True))],
        'C_chain_h': [(f'Cchain{chain_length}h', make_C_chain(chain_length, vertical=False))],
        'C_ring': [(f'Cring{ring_size}', make_C_ring(ring_size, vertical=False))],
        'C_ring_v': [(f'Cring{ring_size}v', make_C_ring(ring_size, vertical=True))],
    }

    if adsorbate_type == 'all':
        adsorbates = []
        for ads_list in adsorbate_makers.values():
            adsorbates.extend(ads_list)
    elif adsorbate_type in adsorbate_makers:
        adsorbates = adsorbate_makers[adsorbate_type]
    else:
        raise ValueError(f"Unknown adsorbate type: {adsorbate_type}. "
                         f"Choose from: {list(adsorbate_makers.keys()) + ['graphene', 'all']}")

    for ads_name, ads_mol in adsorbates:
        for site_type, site_positions in sites.items():
            for i, (x, y) in enumerate(site_positions):
                slab_copy = slab.copy()

                # Place adsorbate
                ads_copy = ads_mol.copy()
                # Shift adsorbate so its bottom is at 'height' above surface
                z_top = slab_copy.positions[:, 2].max()
                z_bottom_ads = ads_copy.positions[:, 2].min()
                ads_copy.positions[:, 2] += (z_top + height - z_bottom_ads)
                # Center on the adsorption site
                com_xy = ads_copy.get_center_of_mass()[:2]
                ads_copy.positions[:, 0] += x - com_xy[0]
                ads_copy.positions[:, 1] += y - com_xy[1]

                combined = slab_copy + ads_copy
                combined.cell = slab_copy.cell
                combined.pbc = slab_copy.pbc

                config_name = f"{ads_name}_{site_type}_{i}"
                configs.append((combined, config_name))

    return configs


def setup_vasp_inputs(output_dir, slab_with_ads, incar_template=None):
    """Write POSCAR and optionally copy INCAR template."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    write(os.path.join(output_dir, 'POSCAR'), slab_with_ads, format='vasp')

    if incar_template and os.path.exists(incar_template):
        import shutil
        shutil.copy2(incar_template, os.path.join(output_dir, 'INCAR'))


def process_slab(slab_path, adsorbate_type, height=2.0, chain_length=3,
                 ring_size=6, max_strain=5.0, incar_template=None,
                 rotation_angles=None):
    """Process a single slab file and generate all adsorption configurations."""
    print(f"\nProcessing: {slab_path}")

    slab = read(slab_path, format='vasp')
    parent_dir = os.path.dirname(slab_path) or '.'

    configs = generate_adsorption_configs(
        slab, adsorbate_type, height=height,
        chain_length=chain_length, ring_size=ring_size,
        max_strain=max_strain, rotation_angles=rotation_angles
    )

    if not configs:
        print("  No configurations generated!")
        return []

    results = []
    for combined, config_name in configs:
        output_dir = os.path.join(parent_dir, f"ads_{config_name}")
        setup_vasp_inputs(output_dir, combined, incar_template)
        n_ads = len(combined) - len(slab)
        results.append({
            'config': config_name,
            'directory': output_dir,
            'n_slab_atoms': len(slab),
            'n_adsorbate_atoms': n_ads,
            'n_total_atoms': len(combined),
        })
        print(f"  Generated: {output_dir} ({n_ads} adsorbate atoms, {len(combined)} total)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate adsorption configurations using ASE."
    )
    parser.add_argument('--slab', default='CONTCAR', help="Slab structure file (default: CONTCAR)")
    parser.add_argument('--adsorbate', required=True,
                        choices=['single_C', 'C_chain_v', 'C_chain_h', 'C_ring', 'C_ring_v', 'graphene', 'all'],
                        help="Adsorbate type")
    parser.add_argument('--height', type=float, default=2.0, help="Adsorption height above surface (Ang, default: 2.0)")
    parser.add_argument('--chain_length', type=int, default=3, help="Number of C atoms in chain (default: 3)")
    parser.add_argument('--ring_size', type=int, default=6, help="Number of C atoms in ring (default: 6)")
    parser.add_argument('--max_strain', type=float, default=5.0, help="Max strain %% for graphene matching (default: 5.0)")
    parser.add_argument('--rotation_angles', type=float, nargs='*', default=None,
                        help="Additional rotation angles (degrees) for graphene")
    parser.add_argument('--incar', default=None, help="INCAR template file to copy into each directory")
    parser.add_argument('--batch', action='store_true', help="Batch mode")
    parser.add_argument('--pattern', default='surf_*/CONTCAR', help="Glob pattern for batch mode")

    args = parser.parse_args()

    if args.batch:
        slab_files = sorted(glob.glob(args.pattern))
        if not slab_files:
            print(f"No files matching: {args.pattern}")
            return

        print(f"Batch mode: {len(slab_files)} slab files")
        print(f"Adsorbate: {args.adsorbate}")
        print("=" * 70)

        all_results = []
        for slab_path in slab_files:
            try:
                results = process_slab(
                    slab_path, args.adsorbate, height=args.height,
                    chain_length=args.chain_length, ring_size=args.ring_size,
                    max_strain=args.max_strain, incar_template=args.incar,
                    rotation_angles=args.rotation_angles
                )
                all_results.extend(results)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        print(f"\n{'=' * 70}")
        print(f"Total configurations generated: {len(all_results)}")
    else:
        process_slab(
            args.slab, args.adsorbate, height=args.height,
            chain_length=args.chain_length, ring_size=args.ring_size,
            max_strain=args.max_strain, incar_template=args.incar,
            rotation_angles=args.rotation_angles
        )


if __name__ == "__main__":
    main()
