# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from pathlib import Path

import numpy as np
import torch
from ase import io
from torch_admp.nblist import dp_nblist, sort_pairs, vesin_nblist


class TestNBList(unittest.TestCase):
    """Test nblist"""

    def setUp(self) -> None:
        atoms = io.read(Path(__file__).parent / "../data/512_h2o.xyz")
        type_map = ["O", "H"]

        atype = np.zeros(atoms.get_number_of_atoms(), dtype=np.int32)
        for ii, _atype in enumerate(type_map):
            atype[atoms.symbols == _atype] = ii

        self.positions = torch.tensor(atoms.get_positions())
        self.atype = torch.tensor(atype)
        self.box = torch.tensor(atoms.cell.array)

        self.rcut = 5.0
        self.nnei = 150

    def test_consistent(self):
        pairs_1, ds_1, _buffer_scales = dp_nblist(
            self.positions, self.box, self.nnei, self.rcut
        )
        pairs_2, ds_2, _buffer_scales = vesin_nblist(
            self.positions, self.box, self.rcut
        )
        torch.testing.assert_close(sort_pairs(pairs_1), sort_pairs(pairs_2))
        torch.testing.assert_close(torch.sort(ds_1)[0], torch.sort(ds_2)[0])


if __name__ == "__main__":
    unittest.main()
