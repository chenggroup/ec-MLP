# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from pathlib import Path

import numpy as np
import torch
from deepmd.tf.env import tf
from deepmd.tf.modifier import DipoleChargeModifier
from torch_admp.electrode import (
    LAMMPSElectrodeConstraint,
    charge_optimization,
    setup_from_lammps,
)
from torch_admp.nblist import vesin_nblist

from ec_mlp.tf.modifier import DipoleChargeBetaModifier, DipoleChargeElectrodeModifier


def data_modifier_setup(ut: unittest.TestCase):
    ut.model_name = str(ut.data_dir / "dw.pb")
    ut.coord = np.array(
        np.load(ut.data_dir / "data/set.000/coord.npy")[0], dtype=np.float64
    ).reshape(1, -1)
    ut.box = np.array(
        np.load(ut.data_dir / "data/set.000/box.npy")[0], dtype=np.float64
    ).reshape(1, -1)
    type_map = np.loadtxt(ut.data_dir / "data/type_map.raw", dtype=str)
    atype = np.loadtxt(ut.data_dir / "data/type.raw", dtype=int)
    symbols = type_map[atype]

    ut.dm_ref = DipoleChargeModifier(
        ut.model_name,
        ut.model_charge_map,
        ut.sys_charge_map,
        ut.ewald_h,
        ut.ewald_beta,
    )
    type_map = ut.dm_ref.get_type_map()
    atype = np.zeros(len(symbols), dtype=np.int32)
    for ii, _atype in enumerate(type_map):
        atype[symbols == _atype] = ii
    ut.atype = atype


class TestChargeDipoleBetaModifier(unittest.TestCase):
    def setUp(self) -> None:
        self.data_dir = (
            Path(__file__).parent / "../data/dp_train/tf/modifier_dipole_charge_beta"
        )
        self.model_charge_map = [-8]
        self.sys_charge_map = [6, 1]
        self.ewald_h = 0.2
        self.ewald_beta = 0.4

        data_modifier_setup(self)

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_consistency(self):
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

        # e, f, v
        out_ref = self.dm_ref.eval(self.coord, self.box, self.atype)
        for dm_test in [dm_pt, dm_jax]:
            out_test = dm_test.eval(self.coord, self.box, self.atype)
            # test e and f (virial has not been implemented)
            for ii in range(2):
                np.testing.assert_allclose(
                    out_ref[ii].reshape(-1),
                    out_test[ii].reshape(-1),
                    atol=1e-6,
                )
                # print(out_ref[ii].reshape(-1), out_test[ii].reshape(-1))


class TestChargeDipoleElectrodeModifier(unittest.TestCase):
    def setUp(self) -> None:
        self.data_dir = (
            Path(__file__).parent
            / "../data/dp_train/tf/modifier_dipole_charge_electrode"
        )
        self.model_charge_map = [-8, -8, -8]
        self.sys_charge_map = [6, 1, 9, 7, 0]
        self.ewald_h = 0.2
        self.ewald_beta = 0.4

        data_modifier_setup(self)

        self.dm_test = DipoleChargeElectrodeModifier(
            self.model_name,
            self.model_charge_map,
            self.sys_charge_map,
            self.ewald_h,
            self.ewald_beta,
        )

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_consistency(self):
        # e, f, v
        out_ref = self.dm_ref.eval(self.coord, self.box, self.atype)
        out_test = self.dm_test.eval(self.coord, self.box, self.atype, electrode=False)
        # test e and f (virial has not been implemented)
        for ii in range(2):
            np.testing.assert_allclose(
                out_ref[ii].reshape(-1),
                out_test[ii].reshape(-1),
                atol=1e-6,
            )

    def test_electrode_charge(self):
        # setup charge
        charge = np.array(self.sys_charge_map)[self.atype]
        charge = np.tile(charge, [1, 1])
        # add wfcc
        all_coord, all_charge, _dipole = self.dm_test._extend_system(
            self.coord, self.box, self.atype, charge
        )

        t_positions = torch.tensor(all_coord.reshape(-1, 3), requires_grad=True)
        t_box = torch.tensor(self.box.reshape(3, 3), requires_grad=True)
        t_charges = torch.tensor(all_charge.reshape(-1))

        pairs, ds, buffer_scales = vesin_nblist(t_positions, t_box, self.dm_test.rcut)

        # mask, eta, chi, hardness, constraint_matrix, constraint_vals, ffield_electrode_mask, ffield_potential
        input_data = setup_from_lammps(
            t_positions.shape[0],
            [
                LAMMPSElectrodeConstraint(
                    indices=np.where(np.abs(all_charge.reshape(-1)) < 1e-5)[0],
                    value=-all_charge.sum(),
                    mode="conq",
                    eta=self.dm_test.eta,
                ),
            ],
            symm=False,
        )
        _q_opt, _efield = charge_optimization(
            self.dm_test.calculator,
            t_positions,
            t_box,
            t_charges,
            pairs,
            ds,
            buffer_scales,
            *input_data,
            method="matinv",
        )
        q_opt = t_charges.clone()
        q_opt[input_data[0]] = _q_opt
        self.assertTrue(abs((q_opt.sum()).item()) < 1e-10)


if __name__ == "__main__":
    unittest.main()
