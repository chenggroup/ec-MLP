# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
from ase import io

for atoms_ref, atoms_test in zip(
    io.iread("ref/ref_out/dump.lammpstrj"), io.iread("test/ref_out/dump.lammpstrj")
):
    assert np.allclose(atoms_ref.get_positions(), atoms_test.get_positions())
    assert np.allclose(atoms_ref.get_forces(), atoms_test.get_forces())

print("All tests passed!")
