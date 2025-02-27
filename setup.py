# SPDX-License-Identifier: LGPL-3.0-or-later
from setuptools import find_packages, setup

setup(
    name="dp_dmff",
    version="1.0",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "deepmd.pt": [
            # "ener_pme_charge=dp_dmff.deepmd.model.model:PMEChargeEnergyModel",
            # "ener_pme_charge_model_args=dp_dmff.deepmd.utils.argcheck:ener_pme_charge_model_args",
            "ener_charge_qeq=dp_dmff.deepmd.model.model:EnergyChargeQEqModel",
            "ener_charge_qeq_model_args=dp_dmff.deepmd.utils.argcheck:ener_charge_qeq_model_args",
            "ener_charge=dp_dmff.deepmd.model.model:EnergyChargeModel",
            "ener_charge_model_args=dp_dmff.deepmd.utils.argcheck:ener_charge_model_args",
            "constant_atomic=dp_dmff.deepmd.model.atomic_model:ConstantAtomicModel",
            "constant_model_args=dp_dmff.deepmd.utils.argcheck:constant_model_args",
            "standard2_model_args=dp_dmff.deepmd.utils.argcheck:standard2_model_args",
            "loss_ener_charge=dp_dmff.deepmd.loss:EnergyChargeLoss",
            "loss_ener_charge_args=dp_dmff.deepmd.utils.argcheck:loss_ener_charge",
            "loss_ener_dipole=dp_dmff.deepmd.loss:EnergyDipoleLoss",
            "loss_ener_dipole_args=dp_dmff.deepmd.utils.argcheck:loss_ener_dipole",
            "loss_ener_charge_qeq=dp_dmff.deepmd.loss:EnergyChargeQEqLoss",
            "loss_ener_charge_qeq_args=dp_dmff.deepmd.utils.argcheck:loss_ener_charge_qeq",
            "loss_property_rel=dp_dmff.deepmd.loss:PropertyRelLoss",
            "loss_property_rel_args=dp_dmff.deepmd.utils.argcheck:loss_property_rel",
            "modifier_toy_model_args=dp_dmff.deepmd.utils.argcheck:modifier_toy_model_args",
            "modifier_toy_model=dp_dmff.deepmd.modifier.toy_model:ToyModelModifier",
            "modifier_const_charge_args=dp_dmff.deepmd.utils.argcheck:modifier_const_charge_args",
            "modifier_const_charge=dp_dmff.deepmd.modifier.const_charge:ConstChargeModifier",
            "modifier_charge_args=dp_dmff.deepmd.utils.argcheck:modifier_charge_args",
            "modifier_dipole_charge=dp_dmff.deepmd.modifier.dipole_charge:DipoleChargeModifier",
        ],
    },
)
