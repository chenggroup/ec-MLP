# SPDX-License-Identifier: LGPL-3.0-or-later
from dargs import Argument
from deepmd.utils.argcheck import make_link, modifier_args_plugin


@modifier_args_plugin.register("dipole_charge_beta")
def modifier_dipole_charge_beta_args():
    doc_model_name = "The name of the frozen dipole model file."
    doc_model_charge_map = f"The charge of the WFCC. The list length should be the same as the {make_link('sel_type', 'model[standard]/fitting_net[dipole]/sel_type')}. "
    doc_sys_charge_map = f"The charge of real atoms. The list length should be the same as the {make_link('type_map', 'model/type_map')}"
    doc_ewald_h = "The grid spacing of the FFT grid. Unit is A"
    doc_ewald_beta = f"The splitting parameter of Ewald sum. Unit is A^{-1}"

    return [
        Argument("model_name", str, optional=False, doc=doc_model_name),
        Argument(
            "model_charge_map", list[float], optional=False, doc=doc_model_charge_map
        ),
        Argument("sys_charge_map", list[float], optional=False, doc=doc_sys_charge_map),
        Argument("ewald_beta", float, optional=True, default=0.4, doc=doc_ewald_beta),
        Argument("ewald_h", float, optional=True, default=1.0, doc=doc_ewald_h),
    ]


@modifier_args_plugin.register("dipole_charge_cpm")
def modifier_dipole_charge_cpm_args():
    doc_model_name = "The name of the frozen dipole model file."
    doc_model_charge_map = f"The charge of the WFCC. The list length should be the same as the {make_link('sel_type', 'model[standard]/fitting_net[dipole]/sel_type')}. "
    doc_sys_charge_map = f"The charge of real atoms. The list length should be the same as the {make_link('type_map', 'model/type_map')}"
    doc_ewald_h = "The grid spacing of the FFT grid. Unit is A"
    doc_ewald_beta = f"The splitting parameter of Ewald sum. Unit is A^{-1}"
    doc_gaussian_sigma = "The width of the Gaussian function for CPM. Unit is A"

    return [
        Argument("model_name", str, optional=False, doc=doc_model_name),
        Argument(
            "model_charge_map", list[float], optional=False, doc=doc_model_charge_map
        ),
        Argument("sys_charge_map", list[float], optional=False, doc=doc_sys_charge_map),
        Argument("ewald_beta", float, optional=True, default=0.4, doc=doc_ewald_beta),
        Argument("ewald_h", float, optional=True, default=1.0, doc=doc_ewald_h),
        Argument(
            "gaussian_sigma",
            float,
            optional=True,
            default=0.554,
            doc=doc_gaussian_sigma,
        ),
    ]


@modifier_args_plugin.register("dipole_charge_qeq")
def modifier_dipole_charge_qeq_args():
    doc_dw_model_name = "The name of the frozen DW model file."
    doc_qeq_model_name = "The name of the frozen QEq model file."
    doc_model_charge_map = f"The charge of the WFCC. The list length should be the same as the {make_link('sel_type', 'model[standard]/fitting_net[dipole]/sel_type')}. "
    doc_sys_charge_map = f"The charge of real atoms. The list length should be the same as the {make_link('type_map', 'model/type_map')}"
    doc_ewald_h = "The grid spacing of the FFT grid. Unit is A"
    doc_ewald_beta = f"The splitting parameter of Ewald sum. Unit is A^{-1}"
    doc_gaussian_sigma = "The width of the Gaussian function for CPM. Unit is A"

    return [
        Argument("dw_model_name", str, optional=False, doc=doc_dw_model_name),
        Argument("qeq_model_name", str, optional=False, doc=doc_qeq_model_name),
        Argument(
            "model_charge_map", list[float], optional=False, doc=doc_model_charge_map
        ),
        Argument("sys_charge_map", list[float], optional=False, doc=doc_sys_charge_map),
        Argument("ewald_beta", float, optional=True, default=0.4, doc=doc_ewald_beta),
        Argument("ewald_h", float, optional=True, default=1.0, doc=doc_ewald_h),
        Argument(
            "gaussian_sigma",
            float,
            optional=True,
            default=0.554,
            doc=doc_gaussian_sigma,
        ),
    ]
