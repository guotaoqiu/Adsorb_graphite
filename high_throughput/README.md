# High-Throughput Graphitization Catalyst Screening

A fully automated DFT pipeline that screens carbon-rich materials to find the
best graphitization catalysts. For every input structure, it computes the
adsorption energy of three carbon species (single C atom, C₃ chain, C₆ ring)
on the most stable low-index surface and ranks all candidates.

---

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Quick Start](#quick-start)
- [Detailed Workflow](#detailed-workflow)
  - [Step 1: Bulk Relaxation](#step-1-bulk-relaxation)
  - [Step 2: Slab Generation + Relaxation](#step-2-slab-generation--relaxation)
  - [Step 3: Surface Energy Ranking](#step-3-surface-energy-ranking)
  - [Step 4: Adsorption Screening](#step-4-adsorption-screening)
- [INCAR Settings Explained](#incar-settings-explained)
- [Output Files](#output-files)
- [Adding New Compounds](#adding-new-compounds)
- [Troubleshooting](#troubleshooting)
- [Reference Energies](#reference-energies)

---

## Pipeline Overview

```
Input: /path/to/structures/*.vasp
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Bulk Relaxation (ISIF=3)                           │
│  • Auto-generate INCAR per compound                          │
│  • Apply +U if needed (MP rules: O/F or lanthanide/actinide)│
│  • LMAXMIX = 6 (f-block), 4 (d-block), 2 (otherwise)        │
│  • MAGMOM initialized per element                            │
│  Output: relaxed bulk CONTCAR + total energy                 │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Slab Generation + Relaxation (ISIF=2)              │
│  • Generate 7 low-index surfaces via surfaxe                 │
│    (fallback: pymatgen if surfaxe fails)                     │
│  • Add selective dynamics (top/bottom 25% relaxed)           │
│  • Slab thickness = 20 Å, vacuum = 20 Å                      │
│  • Add LDIPOL, IDIPOL=3 for slab dipole correction           │
│  Output: relaxed slabs + total energies per surface          │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Surface Energy Ranking                              │
│  • γ = (E_slab - N × E_bulk/atom) / (2A)                     │
│  • Rank all surfaces per compound                            │
│  • Identify most stable surface for adsorption studies       │
│  Output: surface_energies.json                               │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Adsorption Screening (on most stable surface)      │
│  • Generate symmetry-unique sites via pymatgen               │
│  • Auto-supercell if cell is too small for adsorbate         │
│  • Place single C, C₃ chain, C₆ ring at each site            │
│  • Selective dynamics: relax top 25% + adsorbate             │
│  • Two-phase relaxation (prerelax → refine)                  │
│  • E_ads = E(slab+ads) - E(slab) - E(gas-phase ref)          │
│  Output: adsorption_energies.json (final ranking)            │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### One-time setup

```bash
cd /home/qiugt/work/20260324_Graphite/high_throughput/
```

### The "3-command loop"

```bash
# (1) Set up Step 1 (bulk relaxations) — run once
python3 run_screening.py setup --input_dir /path/to/structures/

# (2) Submit ready jobs — run every time you need to submit
python3 run_screening.py submit

# (3) Advance pipeline — run every time jobs finish
python3 run_screening.py advance
```

Repeat steps (2) and (3) until all four stages complete.

### Check status anytime

```bash
python3 run_screening.py status
```

Example output:
```
======================================================================
SCREENING STATUS
======================================================================
  Step 1 (Bulk):        38/45 converged
  Step 2 (Slabs):       126/210 converged
  Step 3 (Surf Energy): 18 compounds analyzed
  Step 4 (Adsorption):  84/162 converged

  Final results:        12 compounds with adsorption energies
```

---

## Detailed Workflow

### Step 1: Bulk Relaxation

**Purpose:** Get the equilibrium structure and energy of each bulk compound.

```bash
# Set up calculation directories
python3 step1_bulk_relax.py setup \
    --input_dir /home/qiugt/.../structures_carbon_rich_combinations/ \
    --work_dir ./1_bulk \
    --potcar_dir /software/vasp/potpaw_PBE/   # optional

# Submit jobs
python3 step1_bulk_relax.py submit --work_dir ./1_bulk

# Check status
python3 step1_bulk_relax.py status --work_dir ./1_bulk

# Resubmit failed jobs
python3 step1_bulk_relax.py resubmit --work_dir ./1_bulk
```

**Directory structure created:**

```
1_bulk/
├── manifest.json                   # registry of all compounds
├── C-Cu-O/
│   └── mp-556202_CuCO4/
│       ├── POSCAR                  # input structure
│       ├── INCAR                   # auto-generated, element-aware
│       ├── POTCAR                  # MP-recommended variants
│       ├── KPOINTS                 # auto from KSPACING
│       ├── sub_vasp.sh             # SLURM submission script
│       ├── slurm_jobid             # tracking
│       ├── OUTCAR, CONTCAR, ...    # VASP output
│       └── vasp.out, vasp.err
└── C-Fe-O/
    └── ...
```

**Key INCAR settings for bulk relaxation:**

| Setting | Value | Reason |
|---------|-------|--------|
| ISIF | 3 | Full cell + ion relaxation |
| IBRION | 2 | Conjugate gradient |
| NSW | 200 | Max ionic steps |
| EDIFFG | -0.05 | Force convergence (eV/Å) |
| ENCUT | 680 | High accuracy for energy |
| ENAUG | 1360 | Augmentation cutoff (2×ENCUT) |
| LAECHG | True | All-electron charge for Bader |
| ALGO | All | Robust SCF for difficult systems |

### Step 2: Slab Generation + Relaxation

**Purpose:** Create non-dipolar slabs for 7 low-index surfaces and relax them.

```bash
python3 step2_slab_generation.py setup \
    --bulk_dir ./1_bulk \
    --work_dir ./2_slabs \
    --vacuum 20.0 \
    --thickness 20.0 \
    --relax_fraction 0.25

python3 step2_slab_generation.py submit --work_dir ./2_slabs
python3 step2_slab_generation.py status --work_dir ./2_slabs
```

**Surfaces generated:** (100), (010), (001), (110), (101), (011), (111).

**Tools used:**
- **surfaxe** (primary): finds non-dipolar terminations only.
- **pymatgen SlabGenerator** (fallback): used when surfaxe finds no valid
  terminations (e.g., for highly polar materials).

**Selective dynamics applied to POSCAR:** top 25% and bottom 25% relaxed
(`T T T`), middle 50% fixed (`F F F`).

**Directory structure:**

```
2_slabs/
├── manifest.json
├── surface_energies.json           # created by Step 3
└── C-Cu-O/
    └── mp-556202_CuCO4/
        ├── surf_001_term_0/        # one calc per surface
        │   ├── POSCAR (with selective dynamics)
        │   ├── INCAR (slab settings)
        │   ├── surface_info.log    # HKL, area, Tasker type
        │   └── ...
        ├── surf_010_term_1/
        ├── surf_111_term_0/
        └── ...
```

### Step 3: Surface Energy Ranking

**Purpose:** Compute γ for all surfaces, find the most stable one per compound.

```bash
python3 step3_surface_energy.py \
    --slab_dir ./2_slabs \
    --bulk_dir ./1_bulk
```

**Formula:**

```
γ = (E_slab - N_slab × E_bulk_per_atom) / (2 × A)
```

where:
- E_slab from OUTCAR of the slab calculation
- N_slab from CONTCAR (atom count)
- A = |a × b| computed from CONTCAR
- E_bulk_per_atom from Step 1 OUTCAR ÷ atom count

Result is converted from eV/Å² to J/m² using 1 eV/Å² = 16.0218 J/m².

**Output (surface_energies.json):**

```json
{
  "mp-556202_CuCO4": {
    "bulk_e_per_atom": -7.342,
    "surfaces": [
      {"name": "surf_001_term_0", "hkl": "(0, 0, 1)", "gamma_J_m2": 0.523, ...},
      {"name": "surf_111_term_0", "hkl": "(1, 1, 1)", "gamma_J_m2": 0.831, ...}
    ],
    "most_stable": {"name": "surf_001_term_0", "gamma_J_m2": 0.523, ...}
  }
}
```

### Step 4: Adsorption Screening

**Purpose:** Place 3 types of carbon adsorbates on each compound's most stable
surface, relax, and compute adsorption energies.

```bash
python3 step4_adsorption.py setup \
    --slab_dir ./2_slabs \
    --work_dir ./3_adsorption \
    --height 2.5 \
    --relax_fraction 0.25

python3 step4_adsorption.py submit --work_dir ./3_adsorption
python3 step4_adsorption.py energy --work_dir ./3_adsorption
```

**Adsorbates:**

| Type | Formula | Description |
|------|---------|-------------|
| `single_C` | C | Single carbon atom |
| `C_chain_v` | C₃ | Linear chain perpendicular to surface |
| `C_ring` | C₆ | Benzene-like ring flat on surface |

**Site finding:** Uses `pymatgen.analysis.adsorption.AdsorbateSiteFinder` which
applies spglib symmetry to identify only inequivalent ontop/bridge/hollow sites.
Typically yields 3-10 unique sites per surface (not 100+).

**Auto-supercell:** If the slab's ab-plane is too small for the adsorbate
(e.g., C₆ ring needs ~2.8 Å diameter + 8 Å spacing → minimum cell of ~10.8 Å),
the script automatically creates an N×M×1 supercell. Slab energy is then scaled:

```
E(N×M slab) = N × M × E(1×1 slab)   # no need to re-relax the supercell
E_ads = E(N×M slab + adsorbate) - N × M × E(1×1 slab) - E_ref(adsorbate)
```

**Adsorption energy formula:**

```
E_ads(C)  = E(slab+C)  - E(slab) - E(C_atom)         [E_ref = -1.3206 eV]
E_ads(C₃) = E(slab+C₃) - E(slab) - E(C₃_chain_gas)   [E_ref = -19.0208 eV]
E_ads(C₆) = E(slab+C₆) - E(slab) - E(C₆_ring_gas)    [E_ref = -42.8269 eV]
```

The script auto-detects the adsorbate type from the directory name and uses
the matching reference energy. **Do not** use `n_C × E(graphite)/atom` as
reference — see [Reference Energies](#reference-energies).

**Output (adsorption_energies.json):**

```json
{
  "mp-556202_CuCO4": [
    {"config": "C_ontop_0", "adsorbate_type": "single_C",
     "e_adsorption": -3.45, "supercell_factor": 1, ...},
    {"config": "Cring6_2x2_hollow_0", "adsorbate_type": "C_ring",
     "e_adsorption": -7.21, "supercell_factor": 4, ...}
  ]
}
```

Final summary table prints the best adsorption energy per compound per
adsorbate type:

```
==========================================================================================
SCREENING SUMMARY
==========================================================================================
  Compound                   Best C        Best C3       Best C6
  ─────────────────────────────────────────────────────────────────
  mp-556202_CuCO4            -3.452 eV     -8.213 eV     -12.721 eV
  mp-227_NiC                 -4.186 eV     -9.892 eV     -14.103 eV
  ...
```

---

## INCAR Settings Explained

The `utils/incar_generator.py` produces element-aware INCARs. Here's what it
does for each calculation type:

### Element-aware logic (applied to all calc types)

**+U correction (Materials Project rules):**

```python
if (any element ∈ U_table) AND (O ∈ structure OR F ∈ structure
                                 OR any element ∈ lanthanide/actinide):
    apply LDAU=True with MP-recommended U values
```

U values used:

| Element | U (eV) | | Element | U (eV) | | Element | U (eV) |
|---------|--------|-|---------|--------|-|---------|--------|
| V       | 3.25   | | Mn      | 3.9    | | Pr      | 5.5    |
| Cr      | 3.7    | | Fe      | 5.3    | | Nd      | 5.5    |
| Mo      | 4.38   | | Co      | 3.32   | | Sm      | 6.0    |
| W       | 6.2    | | Ni      | 6.2    | | Eu/Gd   | 6.0    |
| U       | 4.5    | | Ce      | 4.5    | | Tb-Tm   | 6.5    |

**LMAXMIX:** auto-set based on highest-orbital element:
- f-block elements (lanthanides/actinides) → **LMAXMIX = 6**
- d-block elements (transition metals) → **LMAXMIX = 4**
- otherwise → **LMAXMIX = 2** (default)

**MAGMOM:** auto-initialized from per-element defaults
(Fe=5.0, Co=0.6, Mn=5.0, Ce=5.0, Gd=7.0, others=0.6).

**KSPACING:** 0.22 Å⁻¹ (Materials Project default).

### Calculation-type-specific settings

| Setting | bulk_relax | slab_relax | ads_prerelax | ads_refine | static |
|---------|-----------|-----------|--------------|------------|--------|
| ISIF    | 3 (full)  | 2 (ions)  | 2            | 2          | -      |
| IBRION  | 2         | 2         | 2            | 1 (RMM-DIIS)| -1   |
| NSW     | 200       | 200       | 200          | 100        | 0      |
| EDIFF   | 1e-5      | 1e-5      | 1e-4         | 1e-5       | 1e-5   |
| EDIFFG  | -0.05     | -0.03     | -0.05        | -0.02      | -      |
| ENCUT   | 680       | 680       | 520 (fast)   | 680        | 680    |
| PREC    | Accurate  | Accurate  | Normal       | Accurate   | Accurate|
| KSPACING| 0.22      | 0.22      | ≥0.20        | 0.22       | 0.22   |
| LDIPOL  | -         | True      | True         | True       | True   |
| LCHARG  | True      | False     | False        | True       | True   |
| LAECHG  | True      | False     | False        | False      | True   |

**Why two phases for adsorption?** Adsorption configurations often start
with the adsorbate ~2.5 Å from the surface. Phase 1 (`ads_prerelax`) uses
cheaper settings (lower ENCUT, PREC=Normal, looser EDIFF) to handle the
large initial atomic motion quickly. Phase 2 (`ads_refine`) then polishes
the structure with full accuracy. Combined, this is **3-5× faster** than
a single high-accuracy run from the same starting structure.

---

## Output Files

### After Step 1 (per compound)

| File | Purpose |
|------|---------|
| `OUTCAR` | Total energy, forces, magnetic moments |
| `CONTCAR` | Relaxed structure (input for Step 2) |
| `CHGCAR` | Charge density (LCHARG=True) |
| `AECCAR0/1/2` | All-electron charge for Bader (LAECHG=True) |
| `WAVECAR` | Wavefunctions (LWAVE=False, not produced) |

### After Step 2 (per slab)

Smaller output (LCHARG=False, LAECHG=False to save disk space):
- `OUTCAR`, `CONTCAR`, `OSZICAR`, `vasprun.xml`
- `surface_info.log` — HKL, area, Tasker type, source

### After Step 3

`2_slabs/surface_energies.json` — sorted γ for each surface, identifies
most stable surface per compound.

### After Step 4

| File | Purpose |
|------|---------|
| `3_adsorption/manifest.json` | Maps compounds → adsorption configs |
| `3_adsorption/<compound>/slab_info.json` | Reference slab energy and source |
| `3_adsorption/<compound>/CONTCAR_slab` | Relaxed slab (for reference) |
| `3_adsorption/<compound>/ads_*/OUTCAR, CONTCAR` | Per-config relaxation result |
| `3_adsorption/<compound>/supercell_NxM_info.json` | Scale factor (if supercell used) |
| `3_adsorption/adsorption_energies.json` | Final ranking |

---

## Adding New Compounds

To add new structures after the initial run:

```bash
# 1. Drop new .vasp files into your structure directory
cp new_compound.vasp /path/to/structures/X-Y-Z/

# 2. Re-run setup (skips already-done compounds)
python3 run_screening.py setup --input_dir /path/to/structures/

# 3. Submit and advance as usual
python3 run_screening.py submit
python3 run_screening.py advance
```

The `manifest.json` files use the directory existence as a "done marker"
for setup, so re-running is safe and idempotent.

---

## Troubleshooting

### "FEXCF error: exchange-correlation table too small"

**Cause:** Atoms are too close together in the initial POSCAR.

**Fix:** Increase adsorbate height in Step 4 setup:

```bash
python3 step4_adsorption.py setup --slab_dir ./2_slabs --height 3.0
```

### "Job not converged within 100 ionic steps"

**Cause:** Adsorbate reconstruction needs more steps.

**Fix:** The pipeline already uses NSW=200 for prerelax. For specific configs,
manually copy CONTCAR → POSCAR and resubmit.

### "Bulk did not converge electronically (200 NELM steps)"

**Cause:** Difficult metallic/transition-metal SCF.

**Fix:** Manually edit INCAR for that compound:

```
ALGO = All        # already default for bulk
AMIX = 0.1        # smaller mixing
BMIX = 0.0001
NELM = 300
```

### "Slab calculation gives unphysical γ (>10 J/m²)"

**Cause:** Polar/dipolar termination from pymatgen fallback.

**Fix:** Filter out polar terminations or accept that surface as unstable.
The most stable surface (lowest γ) is what matters for Step 4.

### "surfaxe found 0 surfaces for some compound"

**Cause:** All terminations are dipolar (e.g., LiCoO₂-like materials).

**Fix:** The pipeline automatically falls back to pymatgen SlabGenerator,
which produces surfaces but they may need manual screening.

### "Asymmetric/dipolar slab — should I use --both_sides?"

The Step 4 script auto-detects asymmetric slabs by comparing top vs bottom
surface composition. If they differ by >15%, both sides are generated
(`ads_*_top_*` and `ads_*_bot_*`). No manual flag needed.

---

## Reference Energies

The reference energies for adsorption energy calculations come from your
own gas-phase VASP calculations using the **same POTCAR/INCAR settings**:

| Species | Formula | Total E (eV) | n_C | E per C |
|---------|---------|-------------|-----|---------|
| Single C atom | C₁ | -1.3206 | 1 | -1.3206 |
| C chain | C₃ | -19.0208 | 3 | -6.3403 |
| C ring | C₆ | -42.8269 | 6 | -7.1378 |
| Graphite | C₄ | -39.7872 | 4 | -9.9468 |

**⚠ Do NOT use scaled graphite energy as reference** for chain/ring adsorption.
The proper thermodynamic reference is the gas-phase species itself, not
n_C atoms taken from graphite. Mixing references would give incorrect E_ads.

To override defaults, edit `step4_adsorption.py`:

```python
REF_ENERGIES = {
    'single_C': -1.32064672,    # change here if you re-compute
    'C_chain': -19.02083563,
    'C_ring': -42.82686827,
}
```

---

## Project Structure

```
high_throughput/
├── run_screening.py          # Master orchestrator
├── step1_bulk_relax.py       # Bulk relaxation
├── step2_slab_generation.py  # Slab generation + relaxation
├── step3_surface_energy.py   # Surface energy ranking
├── step4_adsorption.py       # Adsorption screening
├── utils/
│   ├── __init__.py
│   ├── incar_generator.py    # Element-aware INCAR
│   └── job_manager.py        # SLURM submit + monitor
├── README.md                 # This file
│
└── (created at runtime)
    ├── 1_bulk/               # Bulk calculations
    ├── 2_slabs/              # Slab calculations + γ
    └── 3_adsorption/         # Adsorption calculations + E_ads
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python  | ≥3.8    | -       |
| numpy   | -       | array ops |
| ASE     | ≥3.22   | structure I/O, supercells |
| pymatgen| ≥2022   | site finder, Structure |
| surfaxe | ≥1.0    | non-dipolar slab generation |
| VASP    | 6.x     | DFT calculations (HPC) |
| SLURM   | -       | job scheduling |

Install Python deps:
```bash
pip install numpy ase pymatgen surfaxe
```

---

## SLURM Configuration

Default SLURM submission (override via CLI flags `--partition`, `--ntasks`):

```bash
#!/bin/bash
#SBATCH --job-name=<auto-generated>
#SBATCH --output=vasp.out
#SBATCH --error=vasp.err
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -p cu
#SBATCH --mem=12000mb

source /software/intel2020/.../compilervars.sh intel64
source /software/intel2020/mkl/.../mklvars.sh intel64
source /software/intel2020/impi/.../mpivars.sh
export PATH=/software/vasp.6.4.0/bin:$PATH
mpirun ... vasp_std
```

Edit `utils/job_manager.py` (`SLURM_TEMPLATE`) to change paths or modules.

---

## Workflow Diagram (data flow)

```
*.vasp ─┐
        ├──► Step 1 ──► CONTCAR_bulk + E_bulk
        │              (per compound)
        │                      │
        │                      ▼
        └──► Step 2 ──► slab POSCARs ──► slab OUTCARs + E_slab
                       (7 per compound)        │
                                               ▼
                                       Step 3: γ ranking
                                               │
                                               ▼
                                       most stable surface
                                               │
                                               ▼
                                  Step 4: adsorption configs
                                       single C / C₃ / C₆
                                               │
                                               ▼
                                  Step 4: adsorption OUTCARs
                                               │
                                               ▼
                              E_ads = E(s+a) - E(s) - E_ref
                                               │
                                               ▼
                                  adsorption_energies.json
                                       (final ranking)
```
