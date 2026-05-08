#!/usr/bin/env python3
"""
INCAR generator with Materials Project-compatible settings.

Handles:
- U value assignment based on elements (PBE+U from MP)
- LMAXMIX auto-detection (4 for d-block, 6 for f-block)
- KSPACING from bandgap
- ISMEAR/SIGMA from bandgap
- Magnetic moment initialization
"""

import os
import re
import numpy as np


# ─── U values from Materials Project (PBE+U) ───

MP_U_VALUES = {
    'Co': 3.32, 'Cr': 3.7, 'Fe': 5.3, 'Mn': 3.9,
    'Mo': 4.38, 'Ni': 6.2, 'V': 3.25, 'W': 6.2,
    'Ce': 4.5, 'Pr': 5.5, 'Nd': 5.5, 'Pm': 6.0,
    'Sm': 6.0, 'Eu': 6.0, 'Gd': 6.0, 'Tb': 6.5,
    'Dy': 6.5, 'Ho': 6.5, 'Er': 6.5, 'Tm': 6.5,
    'Yb': 6.0, 'U': 4.5, 'Pu': 4.5,
}

# Elements with d electrons (need LMAXMIX=4)
D_BLOCK = {
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
}

# Elements with f electrons (need LMAXMIX=6)
F_BLOCK = {
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
    'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
}

# MP-recommended POTCAR variants
MP_POTCAR_MAP = {
    'H': 'H', 'He': 'He', 'Li': 'Li_sv', 'Be': 'Be_sv', 'B': 'B',
    'C': 'C', 'N': 'N', 'O': 'O', 'F': 'F', 'Ne': 'Ne',
    'Na': 'Na_pv', 'Mg': 'Mg_pv', 'Al': 'Al', 'Si': 'Si', 'P': 'P',
    'S': 'S', 'Cl': 'Cl', 'Ar': 'Ar', 'K': 'K_sv', 'Ca': 'Ca_sv',
    'Sc': 'Sc_sv', 'Ti': 'Ti_pv', 'V': 'V_sv', 'Cr': 'Cr_pv',
    'Mn': 'Mn_pv', 'Fe': 'Fe_pv', 'Co': 'Co', 'Ni': 'Ni_pv',
    'Cu': 'Cu_pv', 'Zn': 'Zn', 'Ga': 'Ga_d', 'Ge': 'Ge_d',
    'As': 'As', 'Se': 'Se', 'Br': 'Br', 'Kr': 'Kr',
    'Rb': 'Rb_sv', 'Sr': 'Sr_sv', 'Y': 'Y_sv', 'Zr': 'Zr_sv',
    'Nb': 'Nb_sv', 'Mo': 'Mo_pv', 'Tc': 'Tc_pv', 'Ru': 'Ru_pv',
    'Rh': 'Rh_pv', 'Pd': 'Pd', 'Ag': 'Ag', 'Cd': 'Cd',
    'In': 'In_d', 'Sn': 'Sn_d', 'Sb': 'Sb', 'Te': 'Te',
    'I': 'I', 'Xe': 'Xe', 'Cs': 'Cs_sv', 'Ba': 'Ba_sv',
    'La': 'La', 'Ce': 'Ce', 'Pr': 'Pr_3', 'Nd': 'Nd_3',
    'Pm': 'Pm_3', 'Sm': 'Sm_3', 'Eu': 'Eu', 'Gd': 'Gd',
    'Tb': 'Tb_3', 'Dy': 'Dy_3', 'Ho': 'Ho_3', 'Er': 'Er_3',
    'Tm': 'Tm_3', 'Yb': 'Yb_2', 'Lu': 'Lu_3',
    'Hf': 'Hf_pv', 'Ta': 'Ta_pv', 'W': 'W_pv', 'Re': 'Re_pv',
    'Os': 'Os_pv', 'Ir': 'Ir', 'Pt': 'Pt', 'Au': 'Au',
    'Hg': 'Hg', 'Tl': 'Tl_d', 'Pb': 'Pb_d', 'Bi': 'Bi',
    'Po': 'Po_d', 'At': 'At', 'Rn': 'Rn',
}

# Default magnetic moments per element
DEFAULT_MAGMOMS = {
    'Ce': 5.0, 'Co': 0.6, 'Cr': 5.0, 'Dy': 5.0, 'Er': 3.0,
    'Eu': 10.0, 'Fe': 5.0, 'Gd': 7.0, 'Ho': 4.0, 'Mn': 5.0,
    'Mo': 5.0, 'Nd': 3.0, 'Ni': 5.0, 'Pm': 4.0, 'Pr': 2.0,
    'Sm': 5.0, 'Tb': 6.0, 'Tm': 2.0, 'U': 5.0, 'V': 5.0,
    'W': 5.0, 'Yb': 1.0,
}


def get_lmaxmix(elements):
    """Get LMAXMIX based on elements present."""
    elems = set(elements)
    if elems & F_BLOCK:
        return 6
    elif elems & D_BLOCK:
        return 4
    return 2


def should_add_u(elements):
    """Determine if +U correction should be applied (MP logic)."""
    elems = set(elements)
    u_elements = elems & set(MP_U_VALUES.keys())
    if not u_elements:
        return False
    # MP rule: only add U if O or F present, or if lanthanide/actinide
    if ('O' in elems) or ('F' in elems):
        return True
    if u_elements & F_BLOCK:
        return True
    return False


def get_u_params(species_order):
    """
    Generate LDAU* parameters for VASP INCAR.

    Args:
        species_order: list of element symbols in POSCAR order

    Returns:
        dict with LDAUU, LDAUJ, LDAUL, LDAUTYPE, LDAU, LMAXMIX
    """
    ldauu = []
    ldauj = []
    ldaul = []

    for sp in species_order:
        if sp in MP_U_VALUES:
            ldauu.append(MP_U_VALUES[sp])
            ldauj.append(0.0)
            if sp in F_BLOCK:
                ldaul.append(3)  # f orbitals
            else:
                ldaul.append(2)  # d orbitals
        else:
            ldauu.append(0.0)
            ldauj.append(0.0)
            ldaul.append(-1)

    return {
        'LDAU': True,
        'LDAUTYPE': 2,
        'LDAUU': ' '.join(f'{v:.2f}' for v in ldauu),
        'LDAUJ': ' '.join(f'{v:.2f}' for v in ldauj),
        'LDAUL': ' '.join(str(v) for v in ldaul),
    }


def get_magmom_string(species, counts):
    """Generate MAGMOM string for INCAR."""
    parts = []
    for sp, cnt in zip(species, counts):
        mag = DEFAULT_MAGMOMS.get(sp, 0.6)
        parts.append(f'{cnt}*{mag}')
    return ' '.join(parts)


def get_kspacing(bandgap=None):
    """Get KSPACING based on bandgap (MP formula)."""
    if bandgap is None:
        return 0.22
    elif bandgap <= 1e-4:
        return 0.22
    else:
        rmin = max(1.5, 25.22 - 2.87 * bandgap)
        kspacing = 2 * np.pi * 1.0265 / (rmin - 1.0183)
        return min(kspacing, 0.44)


def read_species_from_poscar(poscar_path):
    """Read species list and counts from POSCAR/CONTCAR."""
    with open(poscar_path, 'r') as f:
        lines = f.readlines()
    species = lines[5].split()
    counts = list(map(int, lines[6].split()))
    return species, counts


def generate_incar(species, counts, calc_type='bulk_relax', bandgap=None,
                   custom_overrides=None):
    """
    Generate a complete INCAR dict for a given calculation type.

    Args:
        species: list of element symbols in POSCAR order
        counts: list of atom counts per species
        calc_type: 'bulk_relax', 'slab_relax', 'ads_prerelax', 'ads_refine', 'static'
        bandgap: bandgap in eV (None = unknown, assume metallic)
        custom_overrides: dict of additional INCAR parameters

    Returns:
        dict of INCAR parameters
    """
    elements = []
    for sp, cnt in zip(species, counts):
        elements.extend([sp] * cnt)

    lmaxmix = get_lmaxmix(species)
    magmom = get_magmom_string(species, counts)
    kspacing = get_kspacing(bandgap)

    # Base settings (common to all)
    params = {
        'ALGO': 'Fast',
        'EDIFF': 1e-4,
        'ENCUT': 520,
        'PREC': 'Normal',
        'ISMEAR': 0,
        'SIGMA': 0.2,
        'ISPIN': 2,
        'NELM': 120,
        'NELMIN': 4,
        'LREAL': 'Auto',
        'LASPH': True,
        'LMAXMIX': lmaxmix,
        'LMIXTAU': True,
        'MAGMOM': magmom,
        'KSPACING': max(kspacing, 0.25),
        'KPAR': 4,
        'NCORE': 16,
        'LWAVE': False,
        'LCHARG': False,
        'LAECHG': False,
        'LORBIT': 0,
        'LVTOT': False,
    }

    # Calculation-type-specific settings
    if calc_type == 'bulk_relax':
        params.update({
            'IBRION': 2,
            'ISIF': 3,
            'NSW': 200,
            'EDIFFG': -0.05,
        })

    elif calc_type == 'slab_relax':
        params.update({
            'IBRION': 2,
            'ISIF': 2,
            'NSW': 150,
            'EDIFFG': -0.05,
            'LDIPOL': True,
            'IDIPOL': 3,
        })

    elif calc_type in ('ads_prerelax', 'ads_relax'):
        params.update({
            'IBRION': 2,
            'ISIF': 2,
            'NSW': 150,
            'EDIFFG': -0.05,
            'LDIPOL': True,
            'IDIPOL': 3,
        })

    elif calc_type == 'static':
        params.update({
            'ENCUT': 680,
            'EDIFF': 1e-5,
            'PREC': 'Accurate',
            'IBRION': -1,
            'NSW': 0,
            'NELM': 200,
            'LCHARG': True,
            'LAECHG': True,
            'LORBIT': 11,
            'LVTOT': True,
            'LVHAR': True,
            'NEDOS': 2000,
        })

    # Add U correction if needed
    if should_add_u(species):
        u_params = get_u_params(species)
        params.update(u_params)

    # Custom overrides
    if custom_overrides:
        params.update(custom_overrides)

    return params


def write_incar(params, filepath):
    """Write INCAR parameters to file."""
    with open(filepath, 'w') as f:
        # Write LDAU params together for readability
        ldau_keys = {'LDAU', 'LDAUTYPE', 'LDAUU', 'LDAUJ', 'LDAUL'}
        regular = {k: v for k, v in params.items() if k not in ldau_keys and v is not None}
        ldau = {k: v for k, v in params.items() if k in ldau_keys and v is not None}

        for key in sorted(regular.keys()):
            val = regular[key]
            if isinstance(val, bool):
                val = '.TRUE.' if val else '.FALSE.'
            f.write(f'{key} = {val}\n')

        if ldau:
            f.write('\n# DFT+U settings\n')
            for key in ['LDAU', 'LDAUTYPE', 'LDAUL', 'LDAUU', 'LDAUJ']:
                if key in ldau:
                    val = ldau[key]
                    if isinstance(val, bool):
                        val = '.TRUE.' if val else '.FALSE.'
                    f.write(f'{key} = {val}\n')
