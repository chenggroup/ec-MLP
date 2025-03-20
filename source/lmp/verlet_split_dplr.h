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
IntegrateStyle(verlet/split/dplr,VerletSplitDplr);
// clang-format on
#else

#ifndef LMP_VERLET_SPLIT_DPLR_H
#define LMP_VERLET_SPLIT_DPLR_H

#include "verlet.h"

namespace LAMMPS_NS {

class VerletSplitDplr : public Verlet {
 public:
  VerletSplitDplr(class LAMMPS *, int, char **);
  ~VerletSplitDplr() override;
  void init() override;
  void setup(int) override;
  void setup_minimal(int) override;
  void run(int) override;
  double memory_usage() override;
  
   void setup_kspace_bins(
    double **kspace_sublo_list, double **kspace_subhi_list,
    double **sortbin_lo, double **sortbin_hi);
   void resort_rspace_atom(double **sortbin_lo, double **sortbin_hi);

   // spatial sorting of atoms

  int nbins;                           // # of sorting bins
  int nbinx, nbiny, nbinz;             // bins in each dimension
  int maxbin;                          // max # of bins
  int maxnext;                         // max size of next,permute
  int *binhead;                        // 1st atom in each bin
  int *next;                           // next atom in bin
  int *permute;                        // permutation vector

  int *atom_counts;

  double **kspace_sublo_list;
  double **kspace_subhi_list;

  double **sortbin_lo;
  double **sortbin_hi;
 private:
  int master;                            // 1 if an Rspace proc, 0 if Kspace
  int me_block;                          // proc ID within Rspace/Kspace block
  int ratio;                            // ratio of Rspace procs to Kspace procs
  int *qsize, *qdisp, *xsize, *xdisp;    // MPI gather/scatter params for block comm
  MPI_Comm block;                        // communicator within one block

  int tip4pflag;                         // 1 if Kspace method sets tip4pflag

  double **f_kspace;    // copy of Kspace forces on Rspace procs
  int maxatom;

  void rk_setup();
  void r2k_comm();
  void k2r_comm();
  void neigh_comm(int nflag,int n_pre_exchange,int n_pre_neighbor);
};

}    // namespace LAMMPS_NS

#endif
#endif
