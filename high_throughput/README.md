# High-Throughput Graphitization Catalyst Screening

Automated DFT pipeline to screen carbon-rich carbides for graphite nucleation
by computing adsorption energies of C atom, C₃ chain, and C₆ ring on the
most stable surface of each material.

## Pipeline Overview

```
structures/*.vasp → Step 1 → Step 2 → Step 3 → Step 4 → E_ads ranking
                     bulk     slabs    γ rank   adsorb
                     relax    gen+     surface   C/C₃/C₆
                     (ISIF=3) relax    energy    relax+energy
```

## Quick Start

```bash
cd high_throughput/

# 1. Set up bulk relaxations
python3 run_screening.py setup --input_dir /path/to/structures/

# 2. Submit all bulk jobs
python3 run_screening.py submit

# 3. After bulk jobs finish, advance pipeline
python3 run_screening.py advance    # auto-detects completion, sets up Step 2

# 4. Submit slab jobs
python3 run_screening.py submit

# 5. Keep advancing as jobs finish
python3 run_screening.py advance    # runs Step 3, sets up Step 4
python3 run_screening.py submit     # submit adsorption jobs
python3 run_screening.py advance    # calculates final adsorption energies

# Check status anytime
python3 run_screening.py status
```

## Step-by-Step Details

### Step 1: Bulk Relaxation

```bash
python3 step1_bulk_relax.py setup --input_dir /path/to/structures/ --work_dir ./1_bulk
python3 step1_bulk_relax.py submit --work_dir ./1_bulk
python3 step1_bulk_relax.py status --work_dir ./1_bulk
```

**INCAR settings** (auto-generated per compound):
- ALGO=All, ISIF=3 (full cell relaxation), NSW=200
- ENCUT=680, EDIFF=1e-5, ENAUG=1360
- **U values**: auto-applied from Materials Project (PBE+U) when element has
  d/f electrons AND O/F present (e.g., Fe₃O₄ gets U_Fe=5.3)
- **LMAXMIX**: auto-set (4 for d-block, 6 for f-block, 2 otherwise)
- **KSPACING**: 0.22 (MP default for unknown bandgap)
- **MAGMOM**: initialized per element (Fe=5.0, Co=0.6, etc.)

### Step 2: Slab Generation + Relaxation

```bash
python3 step2_slab_generation.py setup --bulk_dir ./1_bulk --work_dir ./2_slabs
python3 step2_slab_generation.py submit --work_dir ./2_slabs
```

- Generates slabs for 7 low-index surfaces: (100), (010), (001), (110), (101), (011), (111)
- Uses **surfaxe** (non-dipolar terminations), falls back to **pymatgen** if surfaxe fails
- Adds selective dynamics: top/bottom 25% relaxed, middle 50% fixed
- Vacuum = 20 Å, slab thickness = 20 Å

### Step 3: Surface Energy Ranking

```bash
python3 step3_surface_energy.py --slab_dir ./2_slabs --bulk_dir ./1_bulk
```

- γ = (E_slab - N × E_bulk/atom) / (2A)
- Identifies the most stable surface per compound for Step 4

### Step 4: Adsorption Screening

```bash
python3 step4_adsorption.py setup --slab_dir ./2_slabs --work_dir ./3_adsorption
python3 step4_adsorption.py submit --work_dir ./3_adsorption
python3 step4_adsorption.py energy --work_dir ./3_adsorption
```

- Generates adsorption configs for **single C**, **C₃ chain** (vertical), **C₆ ring** (flat)
- Uses **pymatgen AdsorbateSiteFinder** for symmetry-unique sites only
- **Auto-supercell** when cell is too small for adsorbate
- **Selective dynamics**: top 25% slab + adsorbate relaxed
- Adsorption energy: E_ads = E(slab+ads) - E(slab) - E_ref

**Gas-phase reference energies**:

| Adsorbate | Formula | E_ref (eV) |
|-----------|---------|-----------|
| Single C  | C₁      | -1.3206   |
| C chain   | C₃      | -19.0208  |
| C ring    | C₆      | -42.8269  |

## Directory Structure

```
high_throughput/
├── run_screening.py          # Master orchestrator
├── step1_bulk_relax.py       # Bulk relaxation
├── step2_slab_generation.py  # Slab generation + relaxation
├── step3_surface_energy.py   # Surface energy calculation
├── step4_adsorption.py       # Adsorption screening
├── utils/
│   ├── incar_generator.py    # Element-aware INCAR (U, LMAXMIX, MAGMOM)
│   └── job_manager.py        # SLURM submission + monitoring
│
│  Working directories (created at runtime):
├── 1_bulk/                   # Bulk relaxation calculations
│   ├── manifest.json
│   └── C-Cu-O/
│       └── mp-556202_CuCO4/
│           ├── POSCAR, INCAR, POTCAR
│           └── OUTCAR, CONTCAR (after convergence)
├── 2_slabs/                  # Slab calculations
│   ├── manifest.json
│   ├── surface_energies.json
│   └── C-Cu-O/
│       └── mp-556202_CuCO4/
│           ├── surf_001_term_0/
│           ├── surf_010_term_1/
│           └── ...
└── 3_adsorption/             # Adsorption calculations
    ├── manifest.json
    ├── adsorption_energies.json
    └── mp-556202_CuCO4/
        ├── slab_info.json
        ├── ads_C_ontop_0/
        ├── ads_Cchain3v_bridge_0/
        └── ads_Cring6_hollow_0/
```

## SLURM Configuration

Default job settings (override via CLI):
- Partition: `cu`
- Tasks: 64
- Nodes: 1
- Memory: 12 GB

## Dependencies

- Python 3.8+, numpy, ASE, pymatgen, surfaxe
- VASP 6.x (on HPC)
- SLURM scheduler
