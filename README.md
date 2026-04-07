# Adsorb_graphite

High-throughput DFT workflow for calculating the adsorption energy of carbon
species (single C atom, C chain, C ring, graphene) on metal carbide surfaces.

## Overview

This project screens carbide surfaces for graphite nucleation by computing
adsorption energies of progressively larger carbon species. The workflow is:

```
Step 1: Bulk optimization
    │
Step 2: Slab generation → surface relaxation
    │
Step 3: Surface energy calculation → identify most stable surface
    │
Step 4: Generate adsorption configurations (C atom / C3 chain / C6 ring / graphene)
    │
Step 5: Adsorption energy calculation
    ├── 5a. Relaxation + static calculation
    ├── 5b. Find lowest-energy config
    ├── 5c. Frequency calculation (for ZPE)
    └── 5d. ZPE-corrected adsorption energy
```

**Target compounds:** CaBC2, YBC2, LaBC2, CeBC2, PrBC2
(highest C-content carbides from Materials Project phase diagrams)

---

## Project Structure

```
Adsorb_graphite/
├── Step1_bulk_optimization/       # (external: mpjob)
├── Step2_slab_generation/
│   ├── generate_slab.py           # surfaxe-based slab generator
│   └── add_selective_dynamics.py  # T/F flags for slab relaxation
├── Step3_surface_energy/
│   └── calc_surface_energy.py     # γ = (E_slab - N·E_bulk/atom) / 2A
├── Step4_adsorption_sites/
│   └── generate_adsorption.py     # pymatgen AdsorbateSiteFinder + auto-supercell
├── Step5_adsorption_energy/
│   ├── calc_adsorption_energy.py  # E_ads with per-adsorbate references
│   ├── setup_frequency.py         # prepare VASP freq calc (IBRION=5)
│   └── parse_frequency.py         # parse OUTCAR frequencies → ZPE
├── templates/
│   ├── INCAR_relax_template       # surface/adsorption relaxation
│   ├── INCAR_freq_template        # frequency calculation
│   └── reference_energies.json    # gas-phase reference energies
├── utils/
│   └── vasp_utils.py              # shared VASP parsing utilities
└── run_workflow.py                # master orchestrator
```

---

## Step-by-Step Guide

### Step 1: Bulk Structure Optimization

Use MP-compatible settings via mpjob:

```bash
mpjob *.vasp -t u_relax_static -p cu -n 8 -m native -c LHYPERFINE=F
```

### Step 2: Slab Generation

#### 2a. Generate non-equivalent surfaces

```bash
cd Step2/CaBC2/
cp /path/to/bulk/CONTCAR .
python3 generate_slab.py
# Generates: surf_001_term_0/, surf_010_term_0/, surf_110_term_1/, ...
```

#### 2b. Add selective dynamics for surface relaxation

```bash
# Top+bottom 25% relaxed, middle 50% fixed
python3 add_selective_dynamics.py --batch --overwrite

# Custom: 5 Ang per side
python3 add_selective_dynamics.py --batch 5.0 --overwrite
```

#### 2c. Submit VASP surface relaxation jobs

Use the INCAR from `templates/INCAR_relax_template` (ISIF=2, NSW=100).

### Step 3: Surface Energy Calculation

After surface relaxations complete:

```bash
# Point to bulk calculation directory (auto-reads energy + atom count)
python3 calc_surface_energy.py --batch --bulk_dir /path/to/bulk/

# Or using OUTCAR directly
python3 calc_surface_energy.py --batch --bulk_outcar /path/to/bulk/OUTCAR

# Output: surface_energies.json (sorted by gamma)
# Reports the most stable surface for the next step
```

The formula is:

```
γ = (E_slab - N_slab × E_bulk_per_atom) / (2 × A)
```

where `N_slab` and `A` are auto-read from the slab's CONTCAR/POSCAR.

**Note on dipolar (asymmetric) slabs:**

If surfaxe cannot find non-dipolar terminations for a given Miller index (e.g., 001),
you may need to create the slab manually. The standard surface energy formula assumes
two equivalent surfaces and gives an **average** gamma. For asymmetric slabs:
- The reported gamma is the average of top and bottom surface energies
- Unphysically large gamma values (e.g., >10 J/m^2) indicate an unstable polar termination
- Reasonable gamma values are typically 0.5-5 J/m^2

### Step 4: Generate Adsorption Configurations

Run on the most stable surface from Step 3:

```bash
cd surf_001_term_0/   # most stable surface

# Single C atom on all symmetry-unique sites
python3 generate_adsorption.py --slab CONTCAR --adsorbate single_C

# C3 chain (vertical)
python3 generate_adsorption.py --slab CONTCAR --adsorbate C_chain_v

# C6 ring (flat on surface)
python3 generate_adsorption.py --slab CONTCAR --adsorbate C_ring

# Graphene (lattice-matched)
python3 generate_adsorption.py --slab CONTCAR --adsorbate graphene

# Everything at once
python3 generate_adsorption.py --slab CONTCAR --adsorbate all
```

**Key features:**

- **Symmetry-unique sites only** (pymatgen AdsorbateSiteFinder, typically 3-10 sites)
- **Selective dynamics** included: top 25% of slab + all adsorbate atoms relaxed,
  bottom 75% fixed. Configurable via `--relax_fraction 0.30`.
- **Auto-detect asymmetric slabs** (`--both_sides auto`, default): compares chemical
  composition of top vs bottom surface layers. If different (>15% composition
  difference), automatically generates adsorption on both surfaces. Examples:
  - CaBC2 (010) surfaxe: symmetric → top only
  - CaBC2 (001) manual: Ca-top vs BC-bottom → both sides
  - CaBC2 (111) term_2: Ca-rich top vs B-rich bottom → both sides
  Override with `--both_sides yes` (force both) or `--both_sides no` (top only).
- **Auto-supercell**: if ab-plane is too small for the adsorbate (e.g., C_ring on
  a 4×5 Å surface), automatically creates NxMx1 supercell so periodic images
  are >8 Å apart. Configurable via `--min_image_dist 10.0`.

### Step 5: Adsorption Energy Calculation

This is a multi-stage process:

#### 5a. Relaxation → Static Calculation

Submit VASP jobs for each `ads_*/POSCAR`. The INCAR from `templates/INCAR_relax_template`
already has the right settings (ISIF=2, NSW=100, selective dynamics respected).

After relaxation, optionally run a static calculation (NSW=0) for more accurate energy.

#### 5b. Calculate Adsorption Energy (before ZPE)

```bash
# Simplest: defaults are baked in from your gas-phase calculations
python3 calc_adsorption_energy.py --batch --slab_dir .

# Or specify energies explicitly
python3 calc_adsorption_energy.py --batch --slab_dir . \
    --e_single_C -1.32064672 \
    --e_C_chain -19.02083563 \
    --e_C_ring -42.82686827

# Or load from JSON
python3 calc_adsorption_energy.py --batch --slab_dir . \
    --ref_json /path/to/reference_energies.json
```

The formula uses the **total gas-phase energy** of each adsorbate species:

```
E_ads(C)  = E(slab+C)  - E(slab) - E(C_atom)        [ref: -1.3206 eV]
E_ads(C3) = E(slab+C3) - E(slab) - E(C3_chain)      [ref: -19.0208 eV]
E_ads(C6) = E(slab+C6) - E(slab) - E(C6_ring)       [ref: -42.8269 eV]
```

> **Important:** Do NOT use `n_C × E(graphite_per_atom)`. Each adsorbate type has
> its own gas-phase reference. See `templates/reference_energies.json`.

The adsorbate type is **auto-detected** from the directory name
(`ads_C_ontop_0` → single_C, `ads_Cchain3v_bridge_1` → C_chain, etc.).

For supercell configurations, slab energy is automatically scaled:
`E(NxM slab) = N×M × E(1x1 slab)`.

#### 5c. Find Lowest Energy → Set Up Frequency Calculation

After Step 5b identifies the most stable adsorption structure:

```bash
# Set up freq calc for the best config only
python3 setup_frequency.py --best_only \
    --ads_energies adsorption_energies.json --slab_dir .

# Or for all configs
python3 setup_frequency.py --batch --slab_dir .

# Include top surface layer atoms in frequency calc
python3 setup_frequency.py --best_only --slab_dir . --relax_surface_layers 1
```

This creates `ads_*/freq/` directories with:
- **POSCAR**: selective dynamics where only adsorbate atoms (+ optional surface layer)
  are `T T T`, bulk slab is `F F F`
- **INCAR**: IBRION=5 (finite differences), NFREE=2, EDIFF=1e-7, LREAL=False

Graphene configurations are **automatically skipped** (no ZPE needed).

Submit the VASP frequency jobs in each `ads_*/freq/` directory.

#### 5d. Parse Frequencies → ZPE-Corrected Adsorption Energy

After frequency calculations finish:

```bash
# Parse all frequency OUTCARs
python3 parse_frequency.py --batch --pattern "ads_*/freq/OUTCAR"
# Output: zpe_results.json
```

Then recalculate adsorption energy with ZPE:

```bash
python3 calc_adsorption_energy.py --batch --slab_dir . --zpe
# Automatically reads freq/OUTCAR in each ads_* directory
```

The ZPE-corrected formula:

```
E_ads = [E(slab+ads) + ZPE(slab+ads)] - [E(slab) + ZPE(slab)] - [E_ref + ZPE(ref)]

where:
  ZPE(slab+ads) = from frequency calculation (parsed from OUTCAR)
  ZPE(slab)     = 0 (negligible for bulk slab, default)
  ZPE(ref)      = 0 for single C atom (no vibrational modes)
                  from gas-phase freq calc for C3/C6
```

**Notes on frequency calculations:**
- Imaginary frequencies indicate a transition state, not a minimum. The script
  flags these as warnings.
- Only **real** frequencies contribute to ZPE: `ZPE = 0.5 × Σ(ℏω_i)`
- Single C atom has zero vibrational modes → ZPE = 0

---

## Reference Energies

Gas-phase reference energies from own VASP calculations (PBE, same POTCAR/ENCUT):

| Species | Formula | E_total (eV) | Source |
|---------|---------|-------------|--------|
| Single C atom | C₁ | -1.32064672 | C atom in 15×15×15 Å box |
| C chain | C₃ | -19.02083563 | Linear C₃ chain in box |
| C ring | C₆ | -42.82686827 | Benzene-like C₆ ring in box |
| Graphite | C₄ | -39.78722576 | AB stacking, a=2.47, c=7.34 Å |

Stored in `templates/reference_energies.json`. Load via `--ref_json`.

---

## INCAR Templates

### Surface/Adsorption Relaxation (`INCAR_relax_template`)
- ISIF=2 (cell fixed), IBRION=2, NSW=100
- ENCUT=680, EDIFF=1e-5, KSPACING=0.15
- LDIPOL=True, IDIPOL=3, LVHAR=True (dipole correction for slab)
- ISPIN=2, LMAXMIX=6

### Frequency Calculation (`INCAR_freq_template`)
- IBRION=5 (symmetric finite differences), NFREE=2
- POTIM=0.015 (displacement step)
- EDIFF=1e-7 (tighter convergence), LREAL=False (reciprocal space)
- NSW=1, ISIF=0
- LDIPOL=True, IDIPOL=3

---

## Dependencies

- **Python 3.8+**
- **ASE** (ase >= 3.22): atomic structure manipulation, POSCAR I/O
- **pymatgen** (>= 2022): `AdsorbateSiteFinder`, `Structure.from_file`
- **surfaxe**: slab generation (Step 2 only)
- **numpy**
- **VASP** (on HPC): DFT calculations

Install Python dependencies:
```bash
pip install ase pymatgen surfaxe numpy
```

---

## Quick Reference

```bash
# Full workflow for one compound (e.g., CaBC2)

# Step 1: bulk optimization (external)
mpjob CaBC2.vasp -t u_relax_static -p cu -n 8 -m native -c LHYPERFINE=F

# Step 2: generate slabs
python3 generate_slab.py
python3 add_selective_dynamics.py --batch --overwrite
# → submit VASP jobs for each surf_*/

# Step 3: surface energy (after slab relaxations finish)
python3 calc_surface_energy.py --batch --bulk_dir ../bulk/
# → identifies most stable surface

# Step 4: generate adsorption configs (on most stable surface)
cd surf_001_term_0/
python3 generate_adsorption.py --slab CONTCAR --adsorbate all
# → submit VASP jobs for each ads_*/

# Step 5a: adsorption energy (after relaxations finish)
python3 calc_adsorption_energy.py --batch --slab_dir .
# → adsorption_energies.json

# Step 5b: frequency calc (on best config, skip graphene)
python3 setup_frequency.py --best_only --ads_energies adsorption_energies.json --slab_dir .
# → submit VASP job in ads_*/freq/

# Step 5c: ZPE-corrected adsorption energy (after freq calc finishes)
python3 parse_frequency.py --batch
python3 calc_adsorption_energy.py --batch --slab_dir . --zpe
```
