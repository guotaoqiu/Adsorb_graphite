#!/usr/bin/env python3
"""
Parse VASP frequency calculation OUTCAR to extract vibrational frequencies
and calculate Zero-Point Energy (ZPE).

ZPE = 0.5 * sum(hbar * omega_i)  for all REAL frequencies

VASP OUTCAR contains lines like:
   1 f  =   95.123456 THz   597.789012 2PiTHz  3172.345678 cm-1   393.234567 meV
   2 f/i=   12.345678 THz    77.567890 2PiTHz   411.890123 cm-1    51.067890 meV

"f" = real frequency, "f/i" = imaginary frequency (ignored for ZPE).

Usage:
    # Parse single OUTCAR
    python3 parse_frequency.py --outcar ads_C_ontop_0/freq/OUTCAR

    # Batch: parse all freq/OUTCAR files
    python3 parse_frequency.py --batch --pattern "ads_*/freq/OUTCAR"

    # Output ZPE summary as JSON
    python3 parse_frequency.py --batch --output zpe_results.json
"""

import os
import re
import json
import glob
import argparse


def parse_frequencies(outcar_path):
    """
    Parse vibrational frequencies from VASP OUTCAR.

    Returns:
        real_freqs: list of real frequencies in meV
        imag_freqs: list of imaginary frequencies in meV
        all_data: list of dicts with full frequency info
    """
    real_freqs = []
    imag_freqs = []
    all_data = []

    with open(outcar_path, 'r') as f:
        for line in f:
            # Match real frequency line
            match_real = re.match(
                r'\s*(\d+)\s+f\s+=\s+([\d.]+)\s+THz\s+([\d.]+)\s+2PiTHz\s+'
                r'([\d.]+)\s+cm-1\s+([\d.]+)\s+meV',
                line
            )
            if match_real:
                idx = int(match_real.group(1))
                thz = float(match_real.group(2))
                cm1 = float(match_real.group(4))
                mev = float(match_real.group(5))
                real_freqs.append(mev)
                all_data.append({
                    'index': idx, 'type': 'real',
                    'THz': thz, 'cm-1': cm1, 'meV': mev
                })
                continue

            # Match imaginary frequency line
            match_imag = re.match(
                r'\s*(\d+)\s+f/i\s*=\s+([\d.]+)\s+THz\s+([\d.]+)\s+2PiTHz\s+'
                r'([\d.]+)\s+cm-1\s+([\d.]+)\s+meV',
                line
            )
            if match_imag:
                idx = int(match_imag.group(1))
                thz = float(match_imag.group(2))
                cm1 = float(match_imag.group(4))
                mev = float(match_imag.group(5))
                imag_freqs.append(mev)
                all_data.append({
                    'index': idx, 'type': 'imaginary',
                    'THz': thz, 'cm-1': cm1, 'meV': mev
                })

    return real_freqs, imag_freqs, all_data


def calc_zpe(real_freqs_mev):
    """
    Calculate Zero-Point Energy from real frequencies.

    ZPE = 0.5 * sum(freq_i)  where freq_i in meV
    Returns ZPE in eV.
    """
    zpe_mev = 0.5 * sum(real_freqs_mev)
    zpe_ev = zpe_mev / 1000.0
    return zpe_ev


def process_outcar(outcar_path, verbose=True):
    """Process a single OUTCAR and return ZPE info."""
    real_freqs, imag_freqs, all_data = parse_frequencies(outcar_path)

    if not all_data:
        raise ValueError(f"No frequencies found in {outcar_path}")

    zpe_ev = calc_zpe(real_freqs)

    result = {
        'outcar': outcar_path,
        'n_real': len(real_freqs),
        'n_imaginary': len(imag_freqs),
        'n_total': len(all_data),
        'zpe_meV': zpe_ev * 1000,
        'zpe_eV': zpe_ev,
        'real_freqs_meV': real_freqs,
        'imag_freqs_meV': imag_freqs,
    }

    if verbose:
        print(f"\n  OUTCAR: {outcar_path}")
        print(f"  Frequencies: {len(real_freqs)} real + {len(imag_freqs)} imaginary "
              f"= {len(all_data)} total")

        if imag_freqs:
            print(f"  WARNING: {len(imag_freqs)} imaginary frequency(ies) found!")
            for d in all_data:
                if d['type'] == 'imaginary':
                    print(f"    Mode {d['index']}: {d['cm-1']:.1f} cm-1 ({d['meV']:.2f} meV) [imaginary]")

        print(f"\n  Real frequencies (cm-1):")
        for d in all_data:
            if d['type'] == 'real':
                marker = ""
                if d['cm-1'] < 50:
                    marker = "  (low - possible translation/rotation)"
                print(f"    Mode {d['index']:>3}: {d['cm-1']:>10.2f} cm-1  ({d['meV']:>8.3f} meV){marker}")

        print(f"\n  ZPE = {zpe_ev * 1000:.3f} meV = {zpe_ev:.6f} eV")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Parse VASP frequency OUTCAR and calculate ZPE."
    )
    parser.add_argument('--outcar', default=None, help="Single OUTCAR file")
    parser.add_argument('--batch', action='store_true', help="Process multiple OUTCARs")
    parser.add_argument('--pattern', default='ads_*/freq/OUTCAR',
                        help="Glob pattern for batch mode")
    parser.add_argument('--output', default='zpe_results.json', help="Output JSON file")

    args = parser.parse_args()

    results = []

    if args.batch:
        outcar_files = sorted(glob.glob(args.pattern))
        if not outcar_files:
            print(f"No files matching: {args.pattern}")
            return

        print(f"Found {len(outcar_files)} OUTCAR files")
        print("=" * 70)

        for outcar_path in outcar_files:
            try:
                result = process_outcar(outcar_path, verbose=False)
                results.append(result)

                # Extract parent directory name for display
                parent = os.path.dirname(os.path.dirname(outcar_path))
                warnings = ""
                if result['n_imaginary'] > 0:
                    warnings = f"  *** {result['n_imaginary']} imag freq!"
                print(f"  {parent:<40}  ZPE = {result['zpe_eV']:>10.6f} eV  "
                      f"({result['n_real']} real / {result['n_imaginary']} imag){warnings}")
            except Exception as e:
                print(f"  {outcar_path:<40}  ERROR: {e}")

        # Summary
        if results:
            print(f"\n{'=' * 70}")
            print("ZPE SUMMARY")
            print("=" * 70)
            print(f"  {'Config':<40}  {'ZPE (eV)':>10}  {'ZPE (meV)':>10}  {'Real':>5}  {'Imag':>5}")
            print("  " + "-" * 75)
            for r in results:
                parent = os.path.dirname(os.path.dirname(r['outcar']))
                print(f"  {parent:<40}  {r['zpe_eV']:>10.6f}  {r['zpe_meV']:>10.3f}  "
                      f"{r['n_real']:>5}  {r['n_imaginary']:>5}")

    elif args.outcar:
        result = process_outcar(args.outcar, verbose=True)
        results.append(result)
    else:
        parser.error("Specify --outcar or --batch")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
