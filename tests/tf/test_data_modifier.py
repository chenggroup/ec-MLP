# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from pathlib import Path

import numpy as np
from deepmd.tf.modifier import DipoleChargeModifier

from ec_mlp.tf.modifier import DipoleChargeBetaModifier


class TestChargeDipoleBetaModifier(unittest.TestCase):
    def setUp(self) -> None:
        self.data_dir = (
            Path(__file__).parent / "../data/dp_train/tf/modifier_dipole_charge_beta"
        )
        self.model_charge_map = [-8]
        self.sys_charge_map = [6, 1]
        self.ewald_h = 0.2
        self.ewald_beta = 0.4

        self.model_name = str(self.data_dir / "dw.pb")
        self.coord = np.array(
            np.load(self.data_dir / "data/set.000/coord.npy")[0], dtype=np.float64
        ).reshape(1, -1)
        self.box = np.array(
            np.load(self.data_dir / "data/set.000/box.npy")[0], dtype=np.float64
        ).reshape(1, -1)
        type_map = np.loadtxt(self.data_dir / "data/type_map.raw", dtype=str)
        atype = np.loadtxt(self.data_dir / "data/type.raw", dtype=int)
        self.symbols = type_map[atype]

    def test_consistency(self):
        dm_ref = DipoleChargeModifier(
            self.model_name,
            self.model_charge_map,
            self.sys_charge_map,
            self.ewald_h,
            self.ewald_beta,
        )
        dm_pt = DipoleChargeBetaModifier(
            self.model_name,
            self.model_charge_map,
            self.sys_charge_map,
            self.ewald_h,
            self.ewald_beta,
            ewald_calculator="torch",
        )
        dm_jax = DipoleChargeBetaModifier(
            self.model_name,
            self.model_charge_map,
            self.sys_charge_map,
            self.ewald_h,
            self.ewald_beta,
            ewald_calculator="torch",
        )

        type_map = dm_ref.get_type_map()
        atype = np.zeros(len(self.symbols), dtype=np.int32)
        for ii, _atype in enumerate(type_map):
            atype[self.symbols == _atype] = ii

        # e, f, v
        out_ref = dm_ref.eval(self.coord, self.box, atype)
        for dm_test in [dm_pt, dm_jax]:
            out_test = dm_test.eval(self.coord, self.box, atype)
            # test e and f (virial has not been implemented)
            for ii in range(2):
                np.testing.assert_allclose(
                    out_ref[ii].reshape(-1),
                    out_test[ii].reshape(-1),
                    atol=1e-6,
                )
                # print(out_ref[ii].reshape(-1), out_test[ii].reshape(-1))


# pairs, ds, buffer_scales = dp_nblist(t_positions, t_box, self.nnei, self.rcut)

#         # mask, eta, chi, hardness, constraint_matrix, constraint_vals, ffield_electrode_mask, ffield_potential
#         input_data = setup_from_lammps(
#             t_positions.shape[0],
#             [
#                 LAMMPSElectrodeConstraint(
#                     indices=np.where(np.abs(charges.reshape(-1)) < 1e-5)[0],
#                     value=-charges.sum(),
#                     mode="conq",
#                     eta=self.eta,
#                 ),
#             ],
#             symm=False,
#         )
#         _q_opt, _efield = charge_optimization(
#             self.calculator,
#             t_positions,
#             t_box,
#             t_charges,
#             pairs,
#             ds,
#             buffer_scales,
#             *input_data,
#             method="matinv",
#         )
#         q_opt = t_charges.clone()
#         q_opt[input_data[0]] = _q_opt
#         self.er(
#             t_positions,
#             t_box,
#             pairs,
#             ds,
#             buffer_scales,
#             {"charge": q_opt},
#         )

if __name__ == "__main__":
    unittest.main()
