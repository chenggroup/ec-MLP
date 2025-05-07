# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Optional

import numpy as np
import torch
from deepmd.pt.modifier.base_modifier import BaseModifier
from deepmd.pt.utils import env
from torch_admp.pme import CoulombForceModule
from torch_admp.utils import calc_grads

# todo: for debugging
torch.backends.cuda.preferred_linalg_library("magma")


@BaseModifier.register("dipole_charge")
class DipoleChargeModifier(BaseModifier):
    """Parameters
    ----------
    charge_map
            Gives the amount of charge for the atoms
    ewald_h
            Grid spacing of the reciprocal part of Ewald sum. Unit: A
    ewald_beta
            Splitting parameter of the Ewald sum. Unit: A^{-1}
    """

    modifier_type = "dipole_charge"

    def __init__(
        self,
        model_name: str,
        model_charge_map: list[float],
        sys_charge_map: list[float],
        ewald_h: float = 1,
        ewald_beta: float = 1,
        # ethresh: float = 1e-5,
        # **kwargs,
    ) -> None:
        """Constructor."""
        super().__init__()
        self.modifier_type = "dipole_charge"
        self.model_name = model_name
        self.model_charge_map = torch.tensor(np.array(model_charge_map)).to(env.DEVICE)
        self.sys_charge_map = torch.tensor(np.array(sys_charge_map)).to(env.DEVICE)

        self.dw_model = torch.jit.load(model_name, map_location=env.DEVICE)
        self.dw_model.eval()
        self.rcut = self.dw_model.get_rcut()
        # self.ethresh = ethresh
        # # todo: add optional args for CoulombForceModule into argchecks
        self.ewald_h = ewald_h
        self.ewald_beta = ewald_beta
        self.er = CoulombForceModule(
            rcut=self.rcut,
            kappa=self.ewald_beta,
            spacing=self.ewald_h,
        )
        self.placeholder_pairs = torch.ones([1, 2], device=env.DEVICE).to(torch.long)
        self.placeholder_ds = torch.ones(1, device=env.DEVICE)
        self.placeholder_buffer_scales = torch.zeros(1, device=env.DEVICE)

    def serialize(self) -> dict:
        """Serialize the modifier.

        Returns
        -------
        dict
            The serialized data
        """
        data = {
            "@class": "Modifier",
            "type": self.modifier_type,
            "@version": 3,
            "model_name": self.model_name,
            "model_charge_map": self.model_charge_map,
            "sys_charge_map": self.sys_charge_map,
            "rcut": self.rcut,
            "ewald_h": self.ewald_h,
            "ewald_beta": self.ewald_beta,
        }
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "BaseModifier":
        """Deserialize the modifier.

        Parameters
        ----------
        data : dict
            The serialized data

        Returns
        -------
        BaseModel
            The deserialized modifier
        """
        data = data.copy()
        modifier = cls(**data)
        return modifier

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: Optional[torch.Tensor] = None,
        do_atomic_virial=False,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
    ):
        dw_model = self.dw_model.to(env.DEVICE)

        nframes = coord.shape[0]
        natoms = atype.shape[1]

        coord_input = torch.reshape(coord, [nframes, natoms, 3])
        coord_input.requires_grad = True
        # .to(env.GLOBAL_PT_FLOAT_PRECISION).to(env.DEVICE)
        if box is not None:
            box_input = torch.reshape(box, [nframes, 3, 3])
            box_input.requires_grad = True
        else:
            box_input = None
        if fparam is not None:
            fparam_input = torch.reshape(fparam, [nframes, dw_model.get_dim_fparam()])
        else:
            fparam_input = None
        if aparam is not None:
            aparam_input = torch.reshape(aparam, [nframes, dw_model.get_dim_aparam()])
        else:
            aparam_input = None

        # eval in batch
        split_coord = torch.split(coord_input, [1] * nframes, dim=0)
        split_atype = torch.split(atype, [1] * nframes, dim=0)
        if box_input is not None:
            split_box = torch.split(box_input, [1] * nframes, dim=0)
        else:
            split_box = [None] * nframes
        if fparam_input is not None:
            split_fparam = torch.split(fparam_input, [1] * nframes, dim=0)
        else:
            split_fparam = [None] * nframes
        if aparam_input is not None:
            split_aparam = torch.split(aparam_input, [1] * nframes, dim=0)
        else:
            split_aparam = [None] * nframes

        atomic_dipoles = []
        e_corr = []
        f_corr = []
        v_corr = []
        for ii in range(nframes):
            _coord = split_coord[ii]
            _atype = split_atype[ii]
            _box = split_box[ii]
            batch_output = dw_model(
                _coord,
                _atype,
                box=_box,
                do_atomic_virial=do_atomic_virial,
                fparam=split_fparam[ii],
                aparam=split_aparam[ii],
                atomic_weight=None,
            )
            if isinstance(batch_output, tuple):
                batch_output = batch_output[0]
            atomic_dipole = batch_output["dipole"]
            atomic_dipoles.append(atomic_dipole)
            mask = batch_output["mask"]

            # charge for ions
            ion_charges = self.sys_charge_map[_atype[0]]
            # charge for wfcc
            wc_charges = self.model_charge_map[_atype[0]] * mask[0]
            extended_charges = torch.cat([ion_charges, wc_charges], dim=0)
            extended_coords = torch.cat(
                [
                    _coord.reshape(natoms, 3),
                    _coord.reshape(natoms, 3) + atomic_dipole.reshape(natoms, 3),
                ],
                dim=0,
            ).detach()
            extended_coords.requires_grad = True
            # print(extended_charges.shape, extended_coords.shape)

            self.er(
                extended_coords,
                _box.reshape(3, 3) if _box is not None else None,
                self.placeholder_pairs,
                self.placeholder_ds,
                self.placeholder_buffer_scales,
                {"charge": extended_charges},
            )
            e_er = self.er.reciprocal_energy

            f_er = -calc_grads(self.er.reciprocal_energy, extended_coords)
            # if _box is not None:
            #     v_er = -calc_grads(self.er.reciprocal_energy, _box)
            # else:
            #     v_er = None

            # reciprocal electrostatic part
            f_corr_1 = (f_er[:natoms] + f_er[natoms:]).unsqueeze(0)
            # print(f_corr_1.shape)
            # calculation gradient of batch_output["global_dipole"] w.r.t. _coord
            batch_output = dw_model(
                _coord,
                _atype,
                box=_box,
                do_atomic_virial=do_atomic_virial,
                fparam=split_fparam[ii],
                aparam=split_aparam[ii],
                atomic_weight=f_er[natoms:].unsqueeze(0),
            )
            if isinstance(batch_output, tuple):
                batch_output = batch_output[0]
            # batch_output["force"]: nf x nloc x v_dim x 3
            # note: the sign!!!
            f_corr_2 = -batch_output["force"].sum(dim=-2)
            # print(f_corr_2.shape)

            e_corr.append(e_er)
            f_corr.append(f_corr_1 + f_corr_2)

        # nf x nloc x 3
        atomic_dipoles = torch.cat(atomic_dipoles, dim=0)
        # print(atomic_dipoles.shape)
        e_corr = torch.tensor(e_corr)
        f_corr = torch.cat(f_corr, dim=0)
        # # nf
        # print(e_corr.shape)
        # # nf x nloc x 3
        # print(f_corr.shape)

        v_corr = torch.zeros(
            coord.shape[0], 9, device=env.DEVICE, dtype=env.GLOBAL_PT_FLOAT_PRECISION
        )

        return e_corr, f_corr, v_corr

    def modify_data(self, data: dict) -> None:
        """Modify data.

        Parameters
        ----------
        data
            Internal data of DeepmdData.
            Be a dict, has the following keys
            - coord         coordinates
            - box           simulation box
            - atype          atom types
            - find_energy   tells if data has energy
            - find_force    tells if data has force
            - find_virial   tells if data has virial
            - energy        energy
            - force         force
            - virial        virial
        """
        if (
            "find_energy" not in data
            and "find_force" not in data
            and "find_virial" not in data
        ):
            return

        get_nframes = None
        coord = data["coord"][:get_nframes, :]
        if data["box"] is None:
            box = None
        else:
            box = data["box"][:get_nframes, :]
        atype = data["atype"][:get_nframes, :]
        # atype = atype[0]
        # nframes = coord.shape[0]

        tot_e, tot_f, tot_v = self(coord, atype, box, False, None, None)

        if "find_energy" in data and data["find_energy"] == 1.0:
            data["energy"] -= tot_e.reshape(data["energy"].shape)
        if "find_force" in data and data["find_force"] == 1.0:
            data["force"] -= tot_f.reshape(data["force"].shape)
        if "find_virial" in data and data["find_virial"] == 1.0:
            raise NotImplementedError  # todo
            # data["virial"] -= tot_v.reshape(data["virial"].shape)
