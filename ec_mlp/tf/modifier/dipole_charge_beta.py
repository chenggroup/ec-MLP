# SPDX-License-Identifier: LGPL-3.0-or-later
import warnings

import numpy as np
from deepmd.pt.utils.utils import to_numpy_array
from deepmd.tf.common import select_idx_map
from deepmd.tf.infer.ewald_recp import EwaldRecp
from deepmd.tf.modifier.base_modifier import BaseModifier
from deepmd.tf.modifier.dipole_charge import DipoleChargeModifier
from deepmd.tf.utils.data import DeepmdData
from scipy import constants

try:
    import torch
    from torch_admp.pme import CoulombForceModule
    from torch_admp.utils import calc_grads
except ImportError:
    warnings.warn(
        "torch and torch_admp is required for pt ewald_calculator of DipoleChargeBetaModifier",
    )

try:
    import jax
    import jax.numpy as jnp
    from dmff.admp.recip import Ck_1, generate_pme_recip
except ImportError:
    warnings.warn(
        "jax and dmff is required for jax ewald_calculator of DipoleChargeBetaModifier",
    )


@BaseModifier.register("dipole_charge_beta")
class DipoleChargeBetaModifier(DipoleChargeModifier):
    """Parameters
    ----------
    model_name
            The model file for the DeepDipole model
    model_charge_map
            Gives the amount of charge for the wfcc
    sys_charge_map
            Gives the amount of charge for the real atoms
    ewald_h
            Grid spacing of the reciprocal part of Ewald sum. Unit: A
    ewald_beta
            Splitting parameter of the Ewald sum. Unit: A^{-1}
    """

    def __init__(
        self,
        model_name: str,
        model_charge_map: list[float],
        sys_charge_map: list[float],
        ewald_h: float = 1,
        ewald_beta: float = 1,
        ewald_calculator: str = "torch",
        slab_corr: bool = False,
    ) -> None:
        """Constructor."""
        super().__init__(
            model_name, model_charge_map, sys_charge_map, ewald_h, ewald_beta
        )
        self.ewald_calculator = ewald_calculator
        self.slab_corr = slab_corr

        if ewald_calculator == "naive":
            self.er = EwaldRecp(ewald_h, ewald_beta)
        elif ewald_calculator == "jax":
            pass
        elif ewald_calculator == "torch":
            self.er = CoulombForceModule(
                rcut=self.rcut,
                rspace=False,
                kappa=ewald_beta,
                spacing=ewald_h,
                slab_corr=slab_corr,
            )
            self.placeholder_pairs = torch.ones(1, 2).to(torch.long)
            self.placeholder_ds = torch.ones(1)
            self.placeholder_buffer_scales = torch.zeros(1)
        else:
            raise ValueError(f"Unknown ewald_calculator: {ewald_calculator}")

    def _eval_ewald_naive(
        self,
        positions: np.ndarray,
        charges: np.ndarray,
        box: np.ndarray,
    ):
        e, f, v = self.er.eval(
            positions.reshape(1, -1),
            charges.reshape(1, -1),
            box.reshape(1, -1),
        )
        return e[0], f[0], v[0]

    def _eval_ewald_jax(
        self,
        positions: np.ndarray,
        charges: np.ndarray,
        box: np.ndarray,
    ):
        j2ev = constants.physical_constants["joule-electron volt relationship"][0]
        # kJ/mol to eV/particle
        energy_coeff = j2ev * constants.kilo / constants.Avogadro

        pme_recip_fn = generate_pme_recip(
            Ck_fn=Ck_1,
            kappa=self.ewald_beta,
            gamma=False,
            pme_order=6,
            K1=int(box[0, 0] / self.ewald_h),
            K2=int(box[1, 1] / self.ewald_h),
            K3=int(box[2, 2] / self.ewald_h),
            lmax=0,
        )
        positions = jnp.array(positions.reshape(-1, 3))
        box = jnp.array(box.reshape(3, 3))
        charges = jnp.array(charges.reshape(-1, 1))
        e = pme_recip_fn(positions, box, charges) * energy_coeff
        _f, v = jax.grad(pme_recip_fn, argnums=(0, 1))(positions, box, charges)
        f = -_f * energy_coeff
        v = -v * energy_coeff
        return e, f, box.reshape(3, 3).T @ v

    def _eval_ewald_torch(
        self,
        positions: np.ndarray,
        charges: np.ndarray,
        box: np.ndarray,
    ):
        slab_factor = 3.0
        if self.slab_corr:
            box = box.reshape(3, 3)
            box[2, 2] *= slab_factor

        t_positions = torch.tensor(positions.reshape(-1, 3), requires_grad=True)
        t_box = torch.tensor(box.reshape(3, 3), requires_grad=True)
        t_charges = torch.tensor(charges.reshape(-1))
        self.er(
            t_positions,
            t_box,
            self.placeholder_pairs,
            self.placeholder_ds,
            self.placeholder_buffer_scales,
            {"charge": t_charges},
        )

        e = (
            self.er.reciprocal_energy
            + self.er.non_neutral_energy
            + self.er.slab_corr_energy
        )
        f = -calc_grads(e, t_positions)
        v = -calc_grads(e, t_box)
        return (
            to_numpy_array(e),
            to_numpy_array(f),
            box.reshape(3, 3).T @ to_numpy_array(v),
        )

    def eval(
        self,
        coord: np.ndarray,
        box: np.ndarray,
        atype: np.ndarray,
        eval_fv: bool = True,
        modified_charge: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the modification.

        Parameters
        ----------
        coord
            The coordinates of atoms
        box
            The simulation region. PBC is assumed
        atype
            The atom types
        eval_fv
            Evaluate force and virial

        Returns
        -------
        tot_e
            The energy modification
        tot_f
            The force modification
        tot_v
            The virial modification
        """
        atype = np.array(atype, dtype=int)
        coord, atype, imap = self.sort_input(coord, atype)
        # natoms = coord.shape[1] // 3
        natoms = atype.size
        nframes = coord.shape[0]
        box = np.reshape(box, [nframes, 9])
        atype = np.reshape(atype, [natoms])
        sel_idx_map = select_idx_map(atype, self.sel_type)
        nsel = len(sel_idx_map)
        # setup charge
        charge = np.zeros([natoms])  # pylint: disable=no-explicit-dtype
        for ii in range(natoms):
            charge[ii] = self.sys_charge_map[atype[ii]]
        charge = np.tile(charge, [nframes, 1])

        # add wfcc
        all_coord, all_charge, dipole = self._extend_system(coord, box, atype, charge)
        # todo: modified_charge is experimental feature!!! will be replaced by integrated polarisable electrode
        if modified_charge is not None:
            all_charge = all_charge.reshape(-1)
            all_charge[np.where(np.abs(all_charge).reshape(-1) < 1e-10)[0]] = (
                modified_charge.reshape(-1)
            )
            all_charge = all_charge.reshape([nframes, -1])
        # print('compute er')
        tot_e = []
        all_f = []
        all_v = []
        for ii in range(nframes):
            e, f, v = getattr(self, f"_eval_ewald_{self.ewald_calculator}")(
                all_coord[ii], all_charge[ii], box[ii]
            )
            tot_e.append(e)
            all_f.append(f)
            all_v.append(v)
        tot_e = np.reshape(tot_e, [nframes, 1])
        all_f = np.reshape(all_f, [nframes, -1])
        all_v = np.reshape(all_v, [nframes, 9])

        tot_f = None
        tot_v = None
        batch_size = 5
        if self.force is None:
            self.force, self.virial, self.av = self.build_fv_graph()
        if eval_fv:
            # compute f
            ext_f = all_f[:, natoms * 3 :]
            corr_f = []
            corr_v = []
            corr_av = []
            for ii in range(0, nframes, batch_size):
                f, v, av = self._eval_fv(
                    coord[ii : ii + batch_size],
                    box[ii : ii + batch_size],
                    atype,
                    ext_f[ii : ii + batch_size],
                )
                corr_f.append(f)
                corr_v.append(v)
                corr_av.append(av)
            corr_f = np.concatenate(corr_f, axis=0)
            corr_v = np.concatenate(corr_v, axis=0)
            corr_av = np.concatenate(corr_av, axis=0)
            tot_f = all_f[:, : natoms * 3] + corr_f
            for ii in range(nsel):
                orig_idx = sel_idx_map[ii]
                tot_f[:, orig_idx * 3 : orig_idx * 3 + 3] += ext_f[
                    :, ii * 3 : ii * 3 + 3
                ]
            tot_f = self.reverse_map(np.reshape(tot_f, [nframes, -1, 3]), imap)
            # reshape
            tot_f = tot_f.reshape([nframes, natoms, 3])
            # compute v
            dipole3 = np.reshape(dipole, [nframes, nsel, 3])
            ext_f3 = np.reshape(ext_f, [nframes, nsel, 3])
            ext_f3 = np.transpose(ext_f3, [0, 2, 1])
            # fd_corr_v = -np.matmul(ext_f3, dipole3).T.reshape([nframes, 9])
            # fd_corr_v = -np.matmul(ext_f3, dipole3)
            # fd_corr_v = np.transpose(fd_corr_v, [0, 2, 1]).reshape([nframes, 9])
            fd_corr_v = -np.matmul(ext_f3, dipole3).reshape([nframes, 9])
            # print(all_v, '\n', corr_v, '\n', fd_corr_v)
            tot_v = all_v + corr_v + fd_corr_v
            # reshape
            tot_v = tot_v.reshape([nframes, 9])

        return tot_e, tot_f, tot_v

    def modify_data(self, data: dict, data_sys: DeepmdData) -> None:
        """Modify data.

        Parameters
        ----------
        data
            Internal data of DeepmdData.
            Be a dict, has the following keys
            - coord         coordinates
            - box           simulation box
            - type          atom types
            - find_energy   tells if data has energy
            - find_force    tells if data has force
            - find_virial   tells if data has virial
            - energy        energy
            - force         force
            - virial        virial
        data_sys : DeepmdData
            The data system.
        """
        if (
            "find_energy" not in data
            and "find_force" not in data
            and "find_virial" not in data
        ):
            return

        if self.ewald_calculator != "naive":
            if "find_virial" in data and data["find_virial"] == 1.0:
                raise RuntimeError("Virial is not supported")

        get_nframes = None
        coord = data["coord"][:get_nframes, :]
        if not data_sys.pbc:
            raise RuntimeError("Open systems (nopbc) are not supported")
        box = data["box"][:get_nframes, :]
        atype = data["type"][:get_nframes, :]
        atype = atype[0]

        tot_e, tot_f, tot_v = self.eval(coord, box, atype)

        if "find_energy" in data and data["find_energy"] == 1.0:
            data["energy"] -= tot_e.reshape(data["energy"].shape)
        if "find_force" in data and data["find_force"] == 1.0:
            data["force"] -= tot_f.reshape(data["force"].shape)
        if "find_virial" in data and data["find_virial"] == 1.0:
            data["virial"] -= tot_v.reshape(data["virial"].shape)
