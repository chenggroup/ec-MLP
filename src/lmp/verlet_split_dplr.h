// SPDX-License-Identifier: LGPL-3.0-or-later
/* -*- c++ -*- -------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef INTEGRATE_CLASS
// clang-format off
IntegrateStyle(verlet/split/dplr,VerletSplitDPLR);
// clang-format on
#else

#ifndef LMP_VERLET_SPLIT_DPLR_H
#define LMP_VERLET_SPLIT_DPLR_H

#include "verlet_split_kspace.h"

namespace LAMMPS_NS
{

  class VerletSplitDPLR : public VerletSplitKSpace
  {
  public:
    VerletSplitDPLR(class LAMMPS *, int, char **);
    ~VerletSplitDPLR() override;

  private:
    void k2r_comm() override;
  };

} // namespace LAMMPS_NS

#endif
#endif
