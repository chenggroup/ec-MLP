# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import unittest
from pathlib import Path

from deepmd.pt.entrypoints.main import train as pt_train
from deepmd.tf.entrypoints.train import train as tf_train
from deepmd.tf.env import tf


class TFDPTrain:
    def test(self) -> None:
        input_file = "input.json"
        init_model = None
        restart = None
        init_frz_model = None
        output_file = "out.json"

        root_dir = Path(__file__).parent

        os.chdir(root_dir / self.dname)
        tf_train(
            INPUT=input_file,
            init_model=init_model,
            restart=restart,
            init_frz_model=init_frz_model,
            mpi_log="master",
            log_level=2,
            output=output_file,
            log_path=None,
        )
        os.chdir(root_dir)


class PTDPTrain:
    def test(self) -> None:
        input_file = "input.json"
        init_model = None
        restart = None
        init_frz_model = None
        output_file = "out.json"

        root_dir = Path(__file__).parent

        os.chdir(root_dir / self.dname)
        pt_train(
            input_file=input_file,
            init_model=init_model,
            restart=restart,
            init_frz_model=init_frz_model,
            finetune=None,
            model_branch="",
            output=output_file,
        )
        os.chdir(root_dir)


class TestDipoleChargeBetaModifier(TFDPTrain, unittest.TestCase):
    def setUp(self) -> None:
        self.dname = "data/dp_train/tf/modifier_dipole_charge_beta"

    def tearDown(self) -> None:
        tf.reset_default_graph()


class TestDipoleChargeElectrodeModifier(TFDPTrain, unittest.TestCase):
    def setUp(self) -> None:
        self.dname = "data/dp_train/tf/modifier_dipole_charge_electrode"

    def tearDown(self) -> None:
        tf.reset_default_graph()


if __name__ == "__main__":
    unittest.main()
