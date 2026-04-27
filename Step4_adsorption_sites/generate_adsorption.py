#!/usr/bin/env python3
"""
Generate adsorption configurations for carbon species on slab surfaces.

Uses pymatgen's AdsorbateSiteFinder for proper symmetry-based unique site
identification (via spglib), so only symmetrically inequivalent sites are
generated. Typically yields 3-10 sites per slab, not hundreds.

Features:
  - Selective dynamics: top 25% of slab + all adsorbate atoms relaxed (T T T),
    bottom 75% fixed (F F F). Configurable via --relax_fraction.
  - Auto-supercell: if the ab-plane is too small for the adsorbate, automatically
    creates a 2x2x1 (or NxMx1) supercell so periodic images don't interact.
    Also generates the matching clean supercell slab for energy comparison.
  - Minimum image distance criterion: default 8 Ang between adsorbate images.

Supported adsorbates:
  - single_C:    Single carbon atom
  - C_chain_v:   Carbon chain (vertical, perpendicular to surface)
  - C_chain_h:   Carbon chain (horizontal, parallel to surface)
  - C_ring:      Carbon ring (C6, benzene-like, horizontal)
  - C_ring_v:    Carbon ring (C6, vertical/tilted)
  - graphene:    Graphene monolayer (lattice-matched)

Usage:
    python3 generate_adsorption.py --slab CONTCAR --adsorbate single_C
    python3 generate_adsorption.py --slab CONTCAR --adsorbate C_ring --min_image_dist 8.0
    python3 generate_adsorption.py --slab CONTCAR --adsorbate all --relax_fraction 0.25
    python3 generate_adsorption.py --slab CONTCAR --adsorbate graphene --max_strain 5.0
    python3 generate_adsorption.py --batch --pattern "surf_*/CONTCAR" --adsorbate single_C
"""

import os
import argparse
import glob
import numpy as np
from pathlib import Path

from ase.io import read, write
from ase import Atoms
from ase.build import make_supercell

from pymatgen.core import Structure
from pymatgen.analysis.adsorption import AdsorbateSiteFinder


# ─── Slab symmetry detection ───

def detect_slab_asymmetry(slab_ase, layer_tol=1.0):
    """
    Detect whether a slab has asymmetric (dipolar) top and bottom surfaces
    by comparing the chemical composition of the top and bottom layers.

    Args:
        slab_ase: ASE Atoms object of the slab
        layer_tol: thickness (Ang) to define "surface layer"

    Returns:
        (is_asymmetric, top_comp, bot_comp, details_str)
    """
    positions = slab_ase.get_positions()
    symbols = slab_ase.get_chemical_symbols()
    z = positions[:, 2]
    z_min, z_max = z.min(), z.max()

    # Top layer: atoms within layer_tol of the highest z
    top_mask = z >= (z_max - layer_tol)
    bot_mask = z <= (z_min + layer_tol)

    # Count species in each layer
    top_comp = {}
    for i in np.where(top_mask)[0]:
        s = symbols[i]
        top_comp[s] = top_comp.get(s, 0) + 1

    bot_comp = {}
    for i in np.where(bot_mask)[0]:
        s = symbols[i]
        bot_comp[s] = bot_comp.get(s, 0) + 1

    # Normalize to fractions for comparison (handles different layer sizes)
    def normalize(comp):
        total = sum(comp.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in sorted(comp.items())}

    top_frac = normalize(top_comp)
    bot_frac = normalize(bot_comp)

    # Compare: are they the same within tolerance?
    all_species = set(list(top_frac.keys()) + list(bot_frac.keys()))
    max_diff = 0.0
    for sp in all_species:
        diff = abs(top_frac.get(sp, 0) - bot_frac.get(sp, 0))
        max_diff = max(max_diff, diff)

    is_asymmetric = max_diff > 0.15  # >15% composition difference = asymmetric

    top_str = " ".join(f"{k}:{v}" for k, v in sorted(top_comp.items()))
    bot_str = " ".join(f"{k}:{v}" for k, v in sorted(bot_comp.items()))
    details = f"Top=[{top_str}] Bot=[{bot_str}] max_diff={max_diff:.0%}"

    return is_asymmetric, top_comp, bot_comp, details


# ─── Adsorbate size estimates (diameter in xy-plane, Angstrom) ───

def get_adsorbate_diameter(adsorbate_type, chain_length=3, ring_size=6,
                           bond_length_chain=1.3, bond_length_ring=1.4):
    """Estimate the xy-plane diameter of an adsorbate for supercell check."""
    if adsorbate_type == 'single_C':
        return 0.0  # point adsorbate
    elif adsorbate_type == 'C_chain_v':
        return 0.0  # vertical chain has no xy extent
    elif adsorbate_type == 'C_chain_h':
        return (chain_length - 1) * bond_length_chain
    elif adsorbate_type in ('C_ring', 'C_ring_v'):
        radius = bond_length_ring / (2 * np.sin(np.pi / ring_size))
        return 2 * radius
    elif adsorbate_type == 'graphene':
        return 0.0  # graphene fills the cell by definition
    return 0.0


def check_supercell_needed(slab_ase, adsorbate_diameter, min_image_dist=8.0):
    """
    Check if supercell is needed so that periodic images of the adsorbate
    are at least min_image_dist apart.

    Returns (na, nb) supercell multipliers.
    """
    if adsorbate_diameter <= 0:
        return 1, 1

    a_vec = slab_ase.cell[0][:2]
    b_vec = slab_ase.cell[1][:2]
    a_len = np.linalg.norm(a_vec)
    b_len = np.linalg.norm(b_vec)

    # Minimum image distance = cell_length - adsorbate_diameter
    # We need: cell_length - adsorbate_diameter >= min_image_dist
    # So: cell_length >= adsorbate_diameter + min_image_dist
    required = adsorbate_diameter + min_image_dist

    na = max(1, int(np.ceil(required / a_len)))
    nb = max(1, int(np.ceil(required / b_len)))

    return na, nb


def make_supercell_slab(slab_ase, na, nb):
    """Create an NxMx1 supercell of the slab."""
    P = np.array([[na, 0, 0],
                  [0, nb, 0],
                  [0, 0, 1]])
    return make_supercell(slab_ase, P)


# ─── Selective dynamics ───

def apply_selective_dynamics(combined, n_slab_atoms, relax_fraction=0.25,
                             both_sides=False):
    """
    Apply selective dynamics to a slab+adsorbate structure.

    Strategy:
    - All adsorbate atoms (indices >= n_slab_atoms): T T T (relaxed)
    - Top relax_fraction of slab atoms (by z-coordinate): T T T
    - If both_sides: also relax bottom relax_fraction
    - Rest of slab atoms: F F F (fixed)

    Returns list of flags ['T T T' or 'F F F'] for each atom.
    """
    positions = combined.get_positions()

    # Slab atom z-coordinates (only slab, not adsorbate)
    slab_z = positions[:n_slab_atoms, 2]
    z_min = slab_z.min()
    z_max = slab_z.max()
    slab_thickness = z_max - z_min

    # Top relax_fraction of the slab
    z_top_cutoff = z_max - relax_fraction * slab_thickness
    # Bottom cutoff (only used for both_sides)
    z_bot_cutoff = z_min + relax_fraction * slab_thickness

    flags = []
    n_relaxed_slab = 0
    n_fixed_slab = 0
    for i in range(len(combined)):
        if i >= n_slab_atoms:
            # Adsorbate atom - always relax
            flags.append('T T T')
        elif positions[i, 2] >= z_top_cutoff:
            # Top part of slab - relax
            flags.append('T T T')
            n_relaxed_slab += 1
        elif both_sides and positions[i, 2] <= z_bot_cutoff:
            # Bottom part of slab - relax (asymmetric slab)
            flags.append('T T T')
            n_relaxed_slab += 1
        else:
            # Bulk region - fix
            flags.append('F F F')
            n_fixed_slab += 1

    n_ads = len(combined) - n_slab_atoms
    print(f"    Selective dynamics: {n_relaxed_slab} slab(T) + {n_ads} ads(T) + "
          f"{n_fixed_slab} slab(F) = {len(combined)} total")

    return flags


def write_poscar_selective(atoms, flags, filepath):
    """Write POSCAR with selective dynamics flags.

    Handles the case where the input atoms were read from a CONTCAR that
    already had selective dynamics - strips old flags before adding new ones.
    """
    # First write a normal POSCAR via ASE, then inject selective dynamics
    write(filepath + '.tmp', atoms, format='vasp')

    with open(filepath + '.tmp', 'r') as f:
        lines = f.readlines()

    # Parse header: lines 0-6 are always comment, scale, 3x lattice, species, counts
    comment = lines[0]
    scale = lines[1]
    lattice = lines[2:5]
    species_line = lines[5]
    counts_line = lines[6]

    counts = list(map(int, counts_line.split()))
    total = sum(counts)

    # Detect if ASE already wrote "Selective dynamics" line
    idx = 7
    if lines[idx].strip().lower().startswith('s'):
        # Skip the existing "Selective dynamics" line
        idx += 1

    coord_line = lines[idx]  # "Direct" or "Cartesian"
    idx += 1  # Now idx points to first coordinate line

    with open(filepath, 'w') as f:
        f.write(comment)
        f.write(scale)
        for lat in lattice:
            f.write(lat)
        f.write(species_line)
        f.write(counts_line)
        f.write('Selective dynamics\n')
        f.write(coord_line)
        for i in range(total):
            parts = lines[idx + i].split()
            # Take only the first 3 values (x, y, z), discard any old T/F flags
            x, y, z = parts[0], parts[1], parts[2]
            f.write(f'  {x}  {y}  {z}  {flags[i]}\n')

    os.remove(filepath + '.tmp')


# ─── Adsorption site finding ───

def find_adsorption_sites(slab_path, height=2.0, symm_reduce=0.01,
                          near_reduce=0.5, no_obtuse_hollow=True):
    """
    Find symmetrically unique adsorption sites using pymatgen.

    Returns dict with 'ontop', 'bridge', 'hollow' -> list of cartesian (x, y, z).
    """
    slab_pmg = Structure.from_file(slab_path)
    asf = AdsorbateSiteFinder(slab_pmg)

    coords = asf.find_adsorption_sites(
        distance=height,
        symm_reduce=symm_reduce,
        near_reduce=near_reduce,
        no_obtuse_hollow=no_obtuse_hollow,
    )

    sites = {
        'ontop': coords.get('ontop', []),
        'bridge': coords.get('bridge', []),
        'hollow': coords.get('hollow', []),
    }

    total = sum(len(v) for v in sites.values())
    print(f"  Unique adsorption sites: {total} "
          f"(ontop={len(sites['ontop'])}, bridge={len(sites['bridge'])}, "
          f"hollow={len(sites['hollow'])})")

    return sites


def find_bottom_adsorption_sites(slab_path, height=2.0, symm_reduce=0.01,
                                  near_reduce=0.5, no_obtuse_hollow=True):
    """
    Find adsorption sites on the BOTTOM surface of an asymmetric slab.

    Strategy: flip the slab (invert z), find sites on the new "top",
    then flip the site coordinates back.
    """
    from pymatgen.core import Structure as PmgStructure
    slab_pmg = PmgStructure.from_file(slab_path)

    # Flip: z -> 1-z (in fractional coordinates)
    flipped = slab_pmg.copy()
    for i, site in enumerate(flipped):
        new_frac = list(site.frac_coords)
        new_frac[2] = 1.0 - new_frac[2]
        flipped.replace(i, site.species, new_frac)

    asf = AdsorbateSiteFinder(flipped)
    coords = asf.find_adsorption_sites(
        distance=height,
        symm_reduce=symm_reduce,
        near_reduce=near_reduce,
        no_obtuse_hollow=no_obtuse_hollow,
    )

    # Flip site coordinates back: the "top" of flipped slab = bottom of original
    # Convert cartesian sites: z -> c_z - z (where c_z is the c lattice height)
    c_z = slab_pmg.lattice.matrix[2][2]
    sites = {}
    for site_type in ['ontop', 'bridge', 'hollow']:
        flipped_coords = coords.get(site_type, [])
        original_coords = []
        for xyz in flipped_coords:
            original_coords.append([xyz[0], xyz[1], c_z - xyz[2]])
        sites[site_type] = original_coords

    total = sum(len(v) for v in sites.values())
    print(f"  Bottom surface sites: {total} "
          f"(ontop={len(sites['ontop'])}, bridge={len(sites['bridge'])}, "
          f"hollow={len(sites['hollow'])})")

    return sites


# ─── Adsorbate builders ───

def make_single_C():
    return Atoms('C', positions=[(0, 0, 0)])


def make_C_chain(n=3, bond_length=1.3, vertical=True):
    positions = []
    for i in range(n):
        if vertical:
            positions.append([0, 0, i * bond_length])
        else:
            positions.append([i * bond_length, 0, 0])
    return Atoms(f'C{n}', positions=positions)


def make_C_ring(n=6, bond_length=1.4, vertical=False):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = bond_length / (2 * np.sin(np.pi / n))
    positions = []
    for theta in angles:
        if vertical:
            positions.append([radius * np.cos(theta), 0, radius * np.sin(theta)])
        else:
            positions.append([radius * np.cos(theta), radius * np.sin(theta), 0])
    return Atoms(f'C{n}', positions=positions)


def make_graphene_layer(slab, max_strain=5.0, rotation_angles=None):
    """Create a graphene layer matched to the slab surface."""
    a_graphene = 2.46
    g_a1 = np.array([a_graphene, 0, 0])
    g_a2 = np.array([a_graphene * 0.5, a_graphene * np.sqrt(3) / 2, 0])

    s_a1 = slab.cell[0][:2]
    s_a2 = slab.cell[1][:2]

    best_matches = []
    max_n = 8

    for n1 in range(-max_n, max_n + 1):
        for n2 in range(-max_n, max_n + 1):
            if n1 == 0 and n2 == 0:
                continue
            for m1 in range(-max_n, max_n + 1):
                for m2 in range(-max_n, max_n + 1):
                    if m1 == 0 and m2 == 0:
                        continue

                    ga = n1 * g_a1[:2] + n2 * g_a2[:2]
                    gb = m1 * g_a1[:2] + m2 * g_a2[:2]

                    strain_a = np.linalg.norm(ga - s_a1) / np.linalg.norm(s_a1) * 100
                    strain_b = np.linalg.norm(gb - s_a2) / np.linalg.norm(s_a2) * 100

                    cos_g = np.dot(ga, gb) / (np.linalg.norm(ga) * np.linalg.norm(gb))
                    cos_s = np.dot(s_a1, s_a2) / (np.linalg.norm(s_a1) * np.linalg.norm(s_a2))
                    angle_diff = abs(np.arccos(np.clip(cos_g, -1, 1)) -
                                     np.arccos(np.clip(cos_s, -1, 1)))
                    angle_diff_deg = np.degrees(angle_diff)

                    avg_strain = (strain_a + strain_b) / 2
                    if avg_strain < max_strain and angle_diff_deg < 5.0:
                        det = abs(n1 * m2 - n2 * m1)
                        if det > 0:
                            best_matches.append({
                                'n1': n1, 'n2': n2, 'm1': m1, 'm2': m2,
                                'strain_a': strain_a, 'strain_b': strain_b,
                                'avg_strain': avg_strain, 'angle_diff': angle_diff_deg,
                                'n_atoms': 2 * det, 'det': det,
                            })

    if not best_matches:
        print(f"  WARNING: No graphene match found within {max_strain}% strain")
        return []

    best_matches.sort(key=lambda x: (x['avg_strain'], x['n_atoms']))

    results = []
    for match in best_matches[:3]:
        graphene_positions = []
        basis = [np.array([0.0, 0.0]), np.array([1.0 / 3.0, 1.0 / 3.0])]

        search_range = abs(match['n1']) + abs(match['n2']) + abs(match['m1']) + abs(match['m2']) + 2
        for i in range(search_range):
            for j in range(search_range):
                for b in basis:
                    pos_g = (i + b[0]) * g_a1[:2] + (j + b[1]) * g_a2[:2]
                    try:
                        cell_2d = np.array([slab.cell[0][:2], slab.cell[1][:2]])
                        frac = np.linalg.solve(cell_2d.T, pos_g)
                        if 0 <= frac[0] < 1 - 1e-6 and 0 <= frac[1] < 1 - 1e-6:
                            graphene_positions.append([pos_g[0], pos_g[1], 0.0])
                    except np.linalg.LinAlgError:
                        continue

        if not graphene_positions:
            continue

        graphene = Atoms(
            f'C{len(graphene_positions)}',
            positions=graphene_positions,
            cell=slab.cell.copy(),
            pbc=[True, True, False]
        )
        results.append((graphene, match))

    if rotation_angles and results:
        base_graphene = results[0][0]
        for angle in rotation_angles:
            rotated = base_graphene.copy()
            com = rotated.get_center_of_mass()
            rotated.translate(-com)
            c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            rotated.positions = rotated.positions @ rot.T
            rotated.translate(com)
            info = results[0][1].copy()
            info['rotation'] = angle
            results.append((rotated, info))

    return results


# ─── Main logic ───

def generate_adsorption_configs(slab_ase, slab_path, adsorbate_type, height=2.0,
                                chain_length=3, ring_size=6,
                                max_strain=5.0, rotation_angles=None,
                                min_image_dist=8.0, both_sides=False):
    """
    Generate all adsorption configurations for a given adsorbate type.

    If the ab-plane is too small for the adsorbate (periodic images < min_image_dist),
    automatically creates a supercell.

    Returns list of (slab_with_adsorbate, config_name, n_slab_atoms, supercell_info) tuples.
    """
    configs = []

    # ── Check if supercell is needed ──
    ads_diameter = get_adsorbate_diameter(adsorbate_type, chain_length, ring_size)
    na, nb = check_supercell_needed(slab_ase, ads_diameter, min_image_dist)
    supercell_tag = ""

    if (na > 1 or nb > 1) and adsorbate_type != 'graphene':
        a_len = np.linalg.norm(slab_ase.cell[0][:2])
        b_len = np.linalg.norm(slab_ase.cell[1][:2])
        print(f"  Cell too small for {adsorbate_type}: a={a_len:.2f}, b={b_len:.2f} Ang")
        print(f"  Adsorbate diameter ~{ads_diameter:.2f} Ang, "
              f"need {ads_diameter + min_image_dist:.1f} Ang between images")
        print(f"  Creating {na}x{nb}x1 supercell...")

        slab_ase = make_supercell_slab(slab_ase, na, nb)
        supercell_tag = f"_{na}x{nb}"

        a_len_new = np.linalg.norm(slab_ase.cell[0][:2])
        b_len_new = np.linalg.norm(slab_ase.cell[1][:2])
        print(f"  New cell: a={a_len_new:.2f}, b={b_len_new:.2f} Ang, "
              f"{len(slab_ase)} atoms")

        # Write supercell POSCAR for the pymatgen site finder
        slab_path_super = slab_path + f'.super{na}x{nb}'
        write(slab_path_super, slab_ase, format='vasp')
        slab_path = slab_path_super

    n_slab = len(slab_ase)

    if adsorbate_type == 'graphene':
        matches = make_graphene_layer(slab_ase, max_strain, rotation_angles)
        for graphene, match_info in matches:
            slab_copy = slab_ase.copy()
            z_top = slab_copy.positions[:, 2].max()

            graphene_copy = graphene.copy()
            graphene_copy.positions[:, 2] = z_top + height

            combined = slab_copy + graphene_copy
            combined.cell = slab_copy.cell
            combined.pbc = slab_copy.pbc

            rot_str = f"_rot{match_info.get('rotation', 0)}" if 'rotation' in match_info else ""
            name = f"graphene_strain{match_info['avg_strain']:.1f}pct{rot_str}"
            configs.append((combined, name, n_slab, f"{na}x{nb}"))

        return configs

    # For molecular adsorbates, find unique sites via pymatgen
    print(f"  Finding TOP surface sites...")
    top_sites = find_adsorption_sites(slab_path, height=height)

    # For asymmetric (dipolar) slabs, also find bottom surface sites
    bottom_sites = None
    if both_sides:
        print(f"  Finding BOTTOM surface sites (asymmetric slab)...")
        bottom_sites = find_bottom_adsorption_sites(slab_path, height=height)

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

    # Build list of (sites_dict, side_label, placement_mode)
    sides = [('top', top_sites)]
    if bottom_sites:
        sides.append(('bot', bottom_sites))

    for side_label, sites in sides:
        for ads_name, ads_mol in adsorbates:
            for site_type, site_coords_list in sites.items():
                for i, site_xyz in enumerate(site_coords_list):
                    slab_copy = slab_ase.copy()
                    ads_copy = ads_mol.copy()

                    if side_label == 'top':
                        # Place above the top surface
                        z_bottom_ads = ads_copy.positions[:, 2].min()
                        ads_copy.positions[:, 2] += (site_xyz[2] - z_bottom_ads)
                    else:
                        # Place below the bottom surface
                        # Flip adsorbate so it hangs down, then position at site_xyz[2]
                        z_top_ads = ads_copy.positions[:, 2].max()
                        ads_copy.positions[:, 2] = site_xyz[2] - (ads_copy.positions[:, 2] - ads_copy.positions[:, 2].min())
                        # Alternatively: mirror z relative to bottom
                        ads_copy.positions[:, 2] = 2 * site_xyz[2] - ads_copy.positions[:, 2]

                    com_xy = ads_copy.get_center_of_mass()[:2]
                    ads_copy.positions[:, 0] += site_xyz[0] - com_xy[0]
                    ads_copy.positions[:, 1] += site_xyz[1] - com_xy[1]

                    combined = slab_copy + ads_copy
                    combined.cell = slab_copy.cell
                    combined.pbc = slab_copy.pbc

                    side_tag = f"_{side_label}" if both_sides else ""
                    config_name = f"{ads_name}{supercell_tag}{side_tag}_{site_type}_{i}"
                    configs.append((combined, config_name, n_slab, f"{na}x{nb}"))

    return configs


def setup_vasp_inputs(output_dir, combined, n_slab_atoms, relax_fraction,
                      incar_template=None, both_sides=False):
    """Write POSCAR with selective dynamics and optionally copy INCAR."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    flags = apply_selective_dynamics(combined, n_slab_atoms, relax_fraction,
                                     both_sides=both_sides)
    write_poscar_selective(combined, flags, os.path.join(output_dir, 'POSCAR'))

    if incar_template and os.path.exists(incar_template):
        import shutil
        shutil.copy2(incar_template, os.path.join(output_dir, 'INCAR'))


def process_slab(slab_path, adsorbate_type, height=2.0, chain_length=3,
                 ring_size=6, max_strain=5.0, incar_template=None,
                 rotation_angles=None, min_image_dist=8.0, relax_fraction=0.25,
                 both_sides=False):
    """Process a single slab file and generate all adsorption configurations."""
    print(f"\nProcessing: {slab_path}")

    slab_ase = read(slab_path, format='vasp')
    parent_dir = os.path.dirname(slab_path) or '.'

    a_len = np.linalg.norm(slab_ase.cell[0][:2])
    b_len = np.linalg.norm(slab_ase.cell[1][:2])
    print(f"  Slab: {len(slab_ase)} atoms, a={a_len:.2f} b={b_len:.2f} Ang")

    # Auto-detect slab asymmetry if not explicitly set
    is_asym, top_comp, bot_comp, asym_details = detect_slab_asymmetry(slab_ase)
    print(f"  Surface analysis: {asym_details}")
    if both_sides == 'auto':
        both_sides = is_asym
        if is_asym:
            print(f"  AUTO-DETECTED: asymmetric (dipolar) slab → generating both sides")
        else:
            print(f"  AUTO-DETECTED: symmetric slab → top surface only")
    elif both_sides == 'yes' or both_sides is True:
        both_sides = True
        print(f"  FORCED: generating adsorption on both sides")
    else:
        both_sides = False

    configs = generate_adsorption_configs(
        slab_ase, slab_path, adsorbate_type, height=height,
        chain_length=chain_length, ring_size=ring_size,
        max_strain=max_strain, rotation_angles=rotation_angles,
        min_image_dist=min_image_dist, both_sides=both_sides,
    )

    if not configs:
        print("  No configurations generated!")
        return []

    # Record supercell info so calc_adsorption_energy.py can scale E(slab) properly
    _, _, _, sc_info = configs[0]
    sc_na, sc_nb = map(int, sc_info.split('x'))
    if sc_na > 1 or sc_nb > 1:
        import json
        sc_meta = {
            'supercell': [sc_na, sc_nb, 1],
            'scale_factor': sc_na * sc_nb,
            'note': 'E(supercell slab) = scale_factor * E(1x1 slab). '
                    'No separate supercell slab calculation needed.',
        }
        meta_path = os.path.join(parent_dir, f'supercell_{sc_na}x{sc_nb}_info.json')
        with open(meta_path, 'w') as f:
            json.dump(sc_meta, f, indent=2)
        print(f"  Supercell info saved: {meta_path}")
        print(f"  E_ads = E(slab+ads) - {sc_na*sc_nb} * E(1x1 slab) - n_C * E_C_ref")

    results = []
    for combined, config_name, n_slab, sc_info in configs:
        output_dir = os.path.join(parent_dir, f"ads_{config_name}")
        setup_vasp_inputs(output_dir, combined, n_slab, relax_fraction,
                          incar_template, both_sides=both_sides)
        n_ads = len(combined) - n_slab
        results.append({
            'config': config_name,
            'directory': output_dir,
            'n_slab_atoms': n_slab,
            'n_adsorbate_atoms': n_ads,
            'n_total_atoms': len(combined),
            'supercell': sc_info,
        })
        print(f"  Generated: {output_dir} ({n_ads} ads + {n_slab} slab = {len(combined)} total)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate adsorption configurations with symmetry-unique sites, "
                    "selective dynamics, and auto-supercell."
    )
    parser.add_argument('--slab', default='CONTCAR',
                        help="Slab structure file (default: CONTCAR)")
    parser.add_argument('--adsorbate', required=True,
                        choices=['single_C', 'C_chain_v', 'C_chain_h',
                                 'C_ring', 'C_ring_v', 'graphene', 'all'],
                        help="Adsorbate type")
    parser.add_argument('--height', type=float, default=2.5,
                        help="Adsorption height above surface (Ang, default: 2.5). "
                             "Too small (<1.8) can cause VASP FEXCF errors.")
    parser.add_argument('--chain_length', type=int, default=3,
                        help="Number of C atoms in chain (default: 3)")
    parser.add_argument('--ring_size', type=int, default=6,
                        help="Number of C atoms in ring (default: 6)")
    parser.add_argument('--max_strain', type=float, default=5.0,
                        help="Max strain %% for graphene matching (default: 5.0)")
    parser.add_argument('--rotation_angles', type=float, nargs='*', default=None,
                        help="Additional rotation angles (degrees) for graphene")
    parser.add_argument('--relax_fraction', type=float, default=0.25,
                        help="Fraction of slab top to relax (default: 0.25 = 25%%)")
    parser.add_argument('--min_image_dist', type=float, default=8.0,
                        help="Minimum distance between periodic images of adsorbate (Ang, "
                             "default: 8.0). Triggers supercell if too small.")
    parser.add_argument('--both_sides', default='auto',
                        choices=['auto', 'yes', 'no'],
                        help="Generate adsorption on both surfaces. "
                             "'auto' (default): detect asymmetry automatically. "
                             "'yes': force both sides. 'no': top only.")
    parser.add_argument('--incar', default=None,
                        help="INCAR template file to copy into each directory")
    parser.add_argument('--batch', action='store_true', help="Batch mode")
    parser.add_argument('--pattern', default='surf_*/CONTCAR',
                        help="Glob pattern for batch mode")

    args = parser.parse_args()

    if args.batch:
        slab_files = sorted(glob.glob(args.pattern))
        if not slab_files:
            print(f"No files matching: {args.pattern}")
            return

        print(f"Batch mode: {len(slab_files)} slab files")
        print(f"Adsorbate: {args.adsorbate}")
        print(f"Relax fraction: {args.relax_fraction:.0%}")
        print(f"Min image distance: {args.min_image_dist} Ang")
        print("=" * 70)

        all_results = []
        for slab_path in slab_files:
            try:
                results = process_slab(
                    slab_path, args.adsorbate, height=args.height,
                    chain_length=args.chain_length, ring_size=args.ring_size,
                    max_strain=args.max_strain, incar_template=args.incar,
                    rotation_angles=args.rotation_angles,
                    min_image_dist=args.min_image_dist,
                    relax_fraction=args.relax_fraction,
                    both_sides=args.both_sides,
                )
                all_results.extend(results)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n{'=' * 70}")
        print(f"Total configurations generated: {len(all_results)}")
    else:
        process_slab(
            args.slab, args.adsorbate, height=args.height,
            chain_length=args.chain_length, ring_size=args.ring_size,
            max_strain=args.max_strain, incar_template=args.incar,
            rotation_angles=args.rotation_angles,
            min_image_dist=args.min_image_dist,
            relax_fraction=args.relax_fraction,
            both_sides=args.both_sides,
        )


if __name__ == "__main__":
    main()
