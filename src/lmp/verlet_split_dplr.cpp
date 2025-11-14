// SPDX-License-Identifier: LGPL-3.0-or-later
// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Si-Yuan Han, Jia-Xin Zhu (Xiamen University)
------------------------------------------------------------------------- */

#include "verlet_split_dplr.h"

#include "universe.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "atom.h"
#include "atom_vec.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "output.h"
#include "update.h"
#include "fix.h"
#include "modify.h"
#include "timer.h"
#include "memory.h"
#include "math_const.h"
#include "error.h"
#include "utils.h"

#include <vector>
#include "fix_dplr.h"
#include "pppm_dplr.h"

using namespace LAMMPS_NS;
using namespace MathConst;  
using std::vector;          

/* ---------------------------------------------------------------------- */

VerletSplitDPLR::VerletSplitDPLR(LAMMPS *lmp, int narg, char **arg) : VerletSplitKSpace(lmp, narg, arg) {}

/* ----------------------------------------------------------------------
   setup before run
   servant partition only sets up KSpace calculation
------------------------------------------------------------------------- */

void VerletSplitDPLR::setup(int flag)
{
  if (comm->me == 0 && screen)
    fprintf(screen,"Setting up Verlet/split/dplr run ...\n");
  
  int nlocal = atom->nlocal;
  if (!master) {
    force->kspace->setup();
  }
  else 
  {
    if (comm->me == 0 && screen) {
      fputs("Setting up Verlet run ...\n",screen);
      if (flag) {
        fmt::print(screen,"  Unit style    : {}\n"
                          "  Current step  : {}\n"
                          "  Time step     : {}\n",
                  update->unit_style,update->ntimestep,update->dt);
        timer->print_timeout(screen);
      }
    }

    if (lmp->kokkos)
      error->all(FLERR,"KOKKOS package requires run_style verlet/kk");

    update->setupflag = 1;

    // setup domain, communication and neighboring
    // acquire ghosts
    // build neighbor lists

    atom->setup();
    modify->setup_pre_exchange();
    if (triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    domain->reset_box();
    comm->setup();
    if (neighbor->style) neighbor->setup_bins();
    comm->exchange();
    if (atom->sortfreq > 0) atom->sort();
    comm->borders();
    if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
    domain->image_check();
    domain->box_too_small_check();
    modify->setup_pre_neighbor();
    neighbor->build(1);
    modify->setup_post_neighbor();
    neighbor->ncalls = 0;

    // compute all forces

    force->setup();
    ev_set(update->ntimestep);
    force_clear();
    modify->setup_pre_force(vflag);

    if (pair_compute_flag) force->pair->compute(eflag,vflag);
    else if (force->pair) force->pair->compute_dummy(eflag,vflag);

    if (atom->molecular != Atom::ATOMIC) {
      if (force->bond) force->bond->compute(eflag,vflag);
      if (force->angle) force->angle->compute(eflag,vflag);
      if (force->dihedral) force->dihedral->compute(eflag,vflag);
      if (force->improper) force->improper->compute(eflag,vflag);
    }

    // record force without kspace
    std::vector<double> tmp_f(nlocal * 3, 0.0);
    for (int i = 0; i < nlocal; i++) {
      tmp_f[i * 3 + 0] = atom->f[i][0];
      tmp_f[i * 3 + 1] = atom->f[i][1];
      tmp_f[i * 3 + 2] = atom->f[i][2];
    }

    if (force->kspace) {
      force->kspace->setup();
      if (kspace_compute_flag) force->kspace->compute(eflag,vflag);
      else force->kspace->compute_dummy(eflag,vflag);
    }
    
    // update fix_dplr->dfele based on kspace force
    // reset atom->f to the values without kspace contribution
    auto fix_dplr_list = modify->get_fix_by_style("dplr");
    if (fix_dplr_list.size() != 1) error->all(FLERR, "fix dplr should be used once");
    FixDPLR *fix_dplr = (FixDPLR *)fix_dplr_list[0];

    if(force->kspace_match("pppm/dplr", 1) == nullptr) {
      for (int i = 0; i < nlocal; i++) {
        fix_dplr->dfele[i * 3 + 0] = atom->f[i][0] - tmp_f[i * 3 + 0];
        fix_dplr->dfele[i * 3 + 1] = atom->f[i][1] - tmp_f[i * 3 + 1];
        fix_dplr->dfele[i * 3 + 2] = atom->f[i][2] - tmp_f[i * 3 + 2];
        atom->f[i][0] = tmp_f[i * 3 + 0];
        atom->f[i][1] = tmp_f[i * 3 + 1];
        atom->f[i][2] = tmp_f[i * 3 + 2];
      }
    }

    modify->setup_pre_reverse(eflag,vflag);
    if (force->newton) comm->reverse_comm();
    modify->setup(vflag);
    output->setup(flag);
    update->setupflag = 0;
  
  };

}

/* ----------------------------------------------------------------------
   communicate and sum Kspace atom forces back to Rspace
------------------------------------------------------------------------- */

void VerletSplitDPLR::k2r_comm()
{



  int n = 0;
  if (!master) n = atom->nlocal;
  if(force->kspace_match("pppm/dplr", 1) != nullptr) {
    PPPMDPLR *pppm_dplr = (PPPMDPLR *)force->kspace_match("pppm/dplr", 1);
    double *fe = &pppm_dplr->get_fele()[0];
    if (master) {
      int nlocal = atom->nlocal;
      for (int i = 0; i < nlocal * 3; i++) {
        fe[i] = 0.0;
      }
    }
    MPI_Gatherv(fe, n*3, MPI_DOUBLE, f_kspace[0], xsize, xdisp,
              MPI_DOUBLE, 0, block);
  }else{
  MPI_Gatherv(atom->f[0], n*3, MPI_DOUBLE, f_kspace[0], xsize, xdisp,
              MPI_DOUBLE, 0, block);
  }

  // Set the forces to FixDPLR object
  auto fix_dplr_list = modify->get_fix_by_style("dplr");
  if (fix_dplr_list.size() != 1) error->all(FLERR, "fix dplr should be used once");
  FixDPLR *fix_dplr = (FixDPLR *)fix_dplr_list[0];
  
  if (master) {
    int nlocal = atom->nlocal;
    for (int i = 0; i < nlocal; i++) {
      fix_dplr->dfele[i * 3 + 0] = f_kspace[i][0];
      fix_dplr->dfele[i * 3 + 1] = f_kspace[i][1];
      fix_dplr->dfele[i * 3 + 2] = f_kspace[i][2];
    }
  }else{
      // modify self energy contribution
      if(force->kspace_match("pppm/dplr", 1) == nullptr){
        modify_dplr_self_energy_contribution(eflag);
      
      }
  }
  if (eflag) MPI_Bcast(&force->kspace->energy, 1, MPI_DOUBLE, 1, block);
  if (vflag) MPI_Bcast(force->kspace->virial, 6, MPI_DOUBLE, 1, block);
}
void VerletSplitDPLR::modify_dplr_self_energy_contribution(int eflag)
{
  
  if (eflag & ENERGY_GLOBAL) {
    
    KSpace* kspace_fast = force->kspace;
    if (!kspace_fast) return;

    
    double qsum_local = 0.0;
    double qsqsum_local = 0.0;

    for (int i = 0; i < atom->nlocal; i++) {
      qsum_local += atom->q[i];
      qsqsum_local += atom->q[i]*atom->q[i];
    }

    double qsum, qsqsum;
    MPI_Allreduce(&qsum_local, &qsum, 1, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&qsqsum_local, &qsqsum, 1, MPI_DOUBLE, MPI_SUM, world);

    double g_ewald = kspace_fast->g_ewald;

    //get volume
    double *prd;
    int triclinic = domain->triclinic;
    if (triclinic == 0) prd = domain->prd;
    else prd = domain->prd_lamda;

    double xprd = prd[0];
    double yprd = prd[1];
    double zprd = prd[2];

    // KSpace slab_volfactor
    double slab_volfactor = kspace_fast->slab_volfactor;
    double zprd_slab = zprd * slab_volfactor;
    double volume = xprd * yprd * zprd_slab;
    
    // Test: which situations  scale != 1
    double scale = 1;
    double qscale = force->qqrd2e * scale;

    double self_energy = (g_ewald*qsqsum/MY_PIS +
                         MY_PI2*qsum*qsum/(g_ewald*g_ewald*volume)) * qscale;
    
    //force->kspace->energy 
    force->kspace->energy += self_energy;
  }
}
