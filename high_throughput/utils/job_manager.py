#!/usr/bin/env python3
"""
SLURM job submission, monitoring, and dependency management.
"""

import os
import re
import subprocess
import time
import json
from pathlib import Path


SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=vasp.out
#SBATCH --error=vasp.err
#SBATCH -N {nodes}
#SBATCH -n {ntasks}
#SBATCH -p {partition}
#SBATCH --mem={mem}
{dependency_line}
srun hostname >./hostfile
echo $SLURM_NTASKS
source /software/intel2020/compilers_and_libraries_2020/linux/bin/compilervars.sh intel64
source /software/intel2020/mkl/bin/mklvars.sh intel64
source /software/intel2020/impi/2019.8.254/intel64/bin/mpivars.sh
export PATH=/software/vasp.6.4.0/bin:$PATH
mpirun -genv I_MPI_FABRICS=shm:ofi -machinefile hostfile -np $SLURM_NTASKS vasp_std
"""


def write_submit_script(calc_dir, job_name='vasp', nodes=1, ntasks=64,
                        partition='cu', mem='12000mb', dependency_jobid=None):
    """Write SLURM submission script."""
    dep_line = ''
    if dependency_jobid:
        dep_line = f'#SBATCH --dependency=afterok:{dependency_jobid}'

    script = SLURM_TEMPLATE.format(
        job_name=job_name,
        nodes=nodes,
        ntasks=ntasks,
        partition=partition,
        mem=mem,
        dependency_line=dep_line,
    )

    script_path = os.path.join(calc_dir, 'sub_vasp.sh')
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    return script_path


def ensure_potcar(calc_dir):
    """Generate POTCAR using vaspkit if not already present."""
    potcar_path = os.path.join(calc_dir, 'POTCAR')
    if os.path.exists(potcar_path) and os.path.getsize(potcar_path) > 0:
        return True

    poscar_path = os.path.join(calc_dir, 'POSCAR')
    if not os.path.exists(poscar_path):
        print(f"  WARNING: No POSCAR in {calc_dir}, cannot generate POTCAR")
        return False

    result = subprocess.run(
        ['vaspkit', '-task', '103'],
        cwd=calc_dir,
        capture_output=True, text=True,
        timeout=30,
    )

    if os.path.exists(potcar_path) and os.path.getsize(potcar_path) > 0:
        return True

    print(f"  WARNING: vaspkit -task 103 failed in {calc_dir}: {result.stderr.strip()}")
    return False


def submit_job(calc_dir, job_name='vasp', dependency_jobid=None,
               nodes=1, ntasks=64, partition='cu', mem='12000mb'):
    """Submit a SLURM job and return the job ID. Generates POTCAR first via vaspkit."""
    # Ensure POTCAR exists before submitting
    if not ensure_potcar(calc_dir):
        print(f"  ERROR: No POTCAR for {calc_dir}, skipping submission")
        return None

    write_submit_script(calc_dir, job_name, nodes, ntasks, partition, mem,
                        dependency_jobid)

    result = subprocess.run(
        ['sbatch', 'sub_vasp.sh'],
        cwd=calc_dir,
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"  ERROR submitting {calc_dir}: {result.stderr.strip()}")
        return None

    # Parse job ID from "Submitted batch job 12345"
    match = re.search(r'Submitted batch job (\d+)', result.stdout)
    if match:
        job_id = int(match.group(1))
        # Save job ID for tracking
        with open(os.path.join(calc_dir, 'slurm_jobid'), 'w') as f:
            f.write(str(job_id))
        return job_id

    print(f"  Could not parse job ID from: {result.stdout.strip()}")
    return None


def get_job_status(job_id):
    """Check SLURM job status. Returns 'RUNNING', 'PENDING', 'COMPLETED', 'FAILED', or 'UNKNOWN'."""
    try:
        result = subprocess.run(
            ['sacct', '-j', str(job_id), '--format=State', '--noheader', '-P'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            states = result.stdout.strip().split('\n')
            # Take the first (main job) state
            state = states[0].strip()
            if state in ('RUNNING', 'PENDING', 'COMPLETED', 'FAILED',
                         'CANCELLED', 'TIMEOUT', 'NODE_FAIL'):
                return state
            return 'UNKNOWN'
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: check squeue
    try:
        result = subprocess.run(
            ['squeue', '-j', str(job_id), '--noheader', '-o', '%T'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split('\n')[0]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return 'UNKNOWN'


def check_vasp_converged(calc_dir):
    """Check if calculation converged (supports both VASP OUTCAR and MACE energy.json)."""
    # Check MACE energy.json first
    energy_json = os.path.join(calc_dir, 'energy.json')
    if os.path.exists(energy_json):
        with open(energy_json) as f:
            info = json.load(f)
        energy = info.get('energy_sigma0')
        converged = info.get('converged', False)
        if energy is not None:
            if converged:
                return True, f'Converged (E={energy:.6f}, MACE)'
            else:
                return False, f'Not converged (E={energy:.4f}, MACE fmax={info.get("forces_max", "?")})'

    # Fall back to VASP OUTCAR
    outcar = os.path.join(calc_dir, 'OUTCAR')
    if not os.path.exists(outcar):
        return False, 'No OUTCAR'

    energy = None
    converged = False
    with open(outcar, 'r') as f:
        for line in f:
            if 'energy(sigma->0)' in line:
                match = re.search(r'energy\(sigma->0\)\s*=\s*([-\d.]+)', line)
                if match:
                    energy = float(match.group(1))
            if 'reached required accuracy' in line:
                converged = True

    if energy is None:
        return False, 'No energy parsed'
    if not converged:
        return False, f'Not converged (E={energy:.4f})'
    return True, f'Converged (E={energy:.6f})'


def parse_energy(calc_dir):
    """Parse final energy from energy.json (MACE) or OUTCAR (VASP)."""
    # Check MACE first
    energy_json = os.path.join(calc_dir, 'energy.json')
    if os.path.exists(energy_json):
        with open(energy_json) as f:
            info = json.load(f)
        return info.get('energy_sigma0')

    # Fall back to VASP OUTCAR
    outcar = os.path.join(calc_dir, 'OUTCAR')
    energy = None
    if os.path.exists(outcar):
        with open(outcar, 'r') as f:
            for line in f:
                if 'energy(sigma->0)' in line:
                    match = re.search(r'energy\(sigma->0\)\s*=\s*([-\d.]+)', line)
                    if match:
                        energy = float(match.group(1))
    return energy


def read_saved_jobid(calc_dir):
    """Read saved SLURM job ID from a calculation directory."""
    jobid_file = os.path.join(calc_dir, 'slurm_jobid')
    if os.path.exists(jobid_file):
        with open(jobid_file) as f:
            return int(f.read().strip())
    return None


def scan_status(base_dir, pattern='*'):
    """Scan calculation directories and report status."""
    import glob
    dirs = sorted(glob.glob(os.path.join(base_dir, pattern)))
    dirs = [d for d in dirs if os.path.isdir(d)]

    results = {'done': [], 'running': [], 'pending': [], 'failed': [], 'not_setup': []}

    for d in dirs:
        if not os.path.exists(os.path.join(d, 'POSCAR')):
            results['not_setup'].append(d)
            continue

        converged, msg = check_vasp_converged(d)
        if converged:
            results['done'].append((d, msg))
            continue

        jobid = read_saved_jobid(d)
        if jobid:
            status = get_job_status(jobid)
            if status in ('RUNNING', 'PENDING'):
                results['running'].append((d, f'{status} (job {jobid})'))
                continue
            elif status in ('FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL'):
                results['failed'].append((d, f'{status} (job {jobid}): {msg}'))
                continue

        if os.path.exists(os.path.join(d, 'OUTCAR')):
            results['failed'].append((d, msg))
        else:
            results['pending'].append(d)

    return results
