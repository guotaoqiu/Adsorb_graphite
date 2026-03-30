import os
import numpy as np
from pathlib import Path
from surfaxe.generation import generate_slabs

def setup_complex_boride_surfaces(input_file='CONTCAR', vacuum=20.0, thickness=20.0):
    if not os.path.exists(input_file):
        print(f"Error: Cannot find file {input_file}")
        return

    hkl_indices = [
        (1,0,0), (0,1,0), (0,0,1),
        (1,1,0), (1,0,1), (0,1,1),
        (1,1,1)
    ]

    print(f"Generating non-equivalent surfaces for {input_file}...")

    all_slabs = generate_slabs(
        structure=input_file,
        hkl=hkl_indices,
        thicknesses=[thickness],
        vacuums=[vacuum],
        save_slabs=False,
        save_metadata=False,
        processes=4
    )

    if not all_slabs:
        print("No non-polar surfaces found.")
        return

    for i, slab_dict in enumerate(all_slabs):
        try:
            hkl_str = "".join(map(str, slab_dict['hkl']))
            label = slab_dict.get('label', i)
            folder_name = f"surf_{hkl_str}_term_{label}"

            Path(folder_name).mkdir(parents=True, exist_ok=True)

            slab_struct = slab_dict['slab']
            slab_struct.to(fmt="poscar", filename=os.path.join(folder_name, 'POSCAR'))

            m = slab_struct.lattice.matrix
            area = np.linalg.norm(np.cross(m[0], m[1]))

            t_type = slab_dict.get('tasker_reflection', slab_dict.get('tasker', 'Unknown'))

            with open(os.path.join(folder_name, 'surface_info.log'), 'w') as f:
                f.write(f"HKL Index: {slab_dict['hkl']}\n")
                f.write(f"Tasker Type: {t_type}\n")
                f.write(f"Area (A^2): {area:.4f}\n")
                f.write(f"Slab Thickness: {thickness} A\n")
                f.write(f"Vacuum Layer: {vacuum} A\n")

            print(f"Generated: {folder_name} (Tasker: {t_type}, Area: {area:.2f})")

        except Exception as e:
            print(f"Error processing slab {i}: {e}")
            continue

if __name__ == "__main__":
    setup_complex_boride_surfaces()
