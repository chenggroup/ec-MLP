# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
from ase import io


def parse_lammps_thermo(log_file):
    """Parse thermo output from LAMMPS log file, return (TotEng, pressure) arrays."""
    with open(log_file, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.strip().startswith('Step') and 'TotEng' in line and 'Press' in line:
            header = line.split()
            toteng_idx = header.index('TotEng')
            press_idx = header.index('Press')

            data = []
            for j in range(i + 1, len(lines)):
                try:
                    values = lines[j].split()
                    data.append([float(values[toteng_idx]), float(values[press_idx])])
                except (ValueError, IndexError):
                    break
            return np.array(data)
    raise ValueError(f"Could not find thermo data in {log_file}")


for atoms_ref, atoms_test in zip(
    io.iread("ref/ref_out/dump.lammpstrj"), io.iread("test/ref_out/dump.lammpstrj")
):
    assert np.allclose(atoms_ref.get_positions(), atoms_test.get_positions())
    assert np.allclose(atoms_ref.get_forces(), atoms_test.get_forces())

ref_data = parse_lammps_thermo("ref/ref_out/log.lammps")[1:]  # skip first data point
test_data = parse_lammps_thermo("test/ref_out/log.lammps.0")[1:]  # skip first data point

assert np.allclose(ref_data[:, 0], test_data[:, 0])  # TotEng
assert np.allclose(ref_data[:, 1], test_data[:, 1])  # Press

print("All tests passed!")
