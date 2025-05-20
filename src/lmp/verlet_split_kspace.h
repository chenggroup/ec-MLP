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
IntegrateStyle(verlet/split/kspace,VerletSplitKSpace);
// clang-format on
#else

#ifndef LMP_VERLET_SPLIT_KSPACE_H
#define LMP_VERLET_SPLIT_KSPACE_H

#include "verlet.h"

namespace LAMMPS_NS
{

  class VerletSplitKSpace : public Verlet
  {
  public:
    VerletSplitKSpace(class LAMMPS *, int, char **);
    ~VerletSplitKSpace() override;
    void init() override;
    void setup(int) override;
    void setup_minimal(int) override;
    void run(int) override;
    double memory_usage() override;

    void setup_kspace_bins(double **kspace_sublo_list, double **kspace_subhi_list,
                           double **sortbin_lo, double **sortbin_hi);
    void resort_rspace_atom(double **sortbin_lo, double **sortbin_hi);

    // spatial sorting of atoms

    int nbins;               // # of sorting bins
    int nbinx, nbiny, nbinz; // bins in each dimension
    int maxbin;              // max # of bins
    int maxnext;             // max size of next,permute
    int *binhead;            // 1st atom in each bin
    int *next;               // next atom in bin
    int *permute;            // permutation vector

    int *atom_counts;

    double **kspace_sublo_list;
    double **kspace_subhi_list;

    double **sortbin_lo;
    double **sortbin_hi;

  protected:
    int master;   // 1 if an Rspace proc, 0 if Kspace
    int me_block; // proc ID within Rspace/Kspace block
    int ratio;    // ratio of Rspace procs to Kspace procs
    int *qsize, *qdisp, *xsize,
        *xdisp;     // MPI gather/scatter params for block comm
    MPI_Comm block; // communicator within one block

    int tip4pflag; // 1 if Kspace method sets tip4pflag

    double **f_kspace; // copy of Kspace forces on Rspace procs
    int maxatom;

    virtual void rk_setup();
    virtual void r2k_comm();
    virtual void k2r_comm();
    virtual void neigh_comm(int nflag, int n_pre_exchange, int n_pre_neighbor);
  };

} // namespace LAMMPS_NS

#endif
#endif
