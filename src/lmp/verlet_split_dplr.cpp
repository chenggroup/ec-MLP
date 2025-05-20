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
#include "error.h"

#include "pppm_dplr.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

VerletSplitDPLR::VerletSplitDPLR(LAMMPS *lmp, int narg, char **arg) : VerletSplitKSpace(lmp, narg, arg)
{
  // error checks on partitions

  if (universe->nworlds != 2)
    error->universe_all(FLERR,"Verlet/split/dplr requires 2 partitions");
  if (universe->procs_per_world[1] % universe->procs_per_world[0])
    error->universe_all(FLERR,"Verlet/split/dplr requires Kspace partition "
                        "size be multiple of Rspace partition size");
  if (comm->style != Comm::BRICK)
    error->universe_all(FLERR,"Verlet/split/dplr can only currently be used with comm_style brick");

  // master = 1 for Rspace procs, 0 for Kspace procs

  if (universe->iworld == 0) master = 1;
  else master = 0;

  ratio = universe->procs_per_world[1] / universe->procs_per_world[0];

  // Kspace root proc broadcasts info about Kspace proc layout to Rspace procs

  int rspace_procgrid[3];

  if (universe->me == universe->root_proc[0]) {
    rspace_procgrid[0] = comm->procgrid[0];
    rspace_procgrid[1] = comm->procgrid[1];
    rspace_procgrid[2] = comm->procgrid[2];
  }
  MPI_Bcast(rspace_procgrid,3,MPI_INT,universe->root_proc[0],universe->uworld);

  int ***rspace_grid2proc;
  memory->create(rspace_grid2proc,rspace_procgrid[0],
                 rspace_procgrid[1],rspace_procgrid[2],
                 "verlet/split/kspace:rspace_grid2proc");

  if (universe->me == universe->root_proc[0]) {
    for (int i = 0; i < comm->procgrid[0]; i++)
      for (int j = 0; j < comm->procgrid[1]; j++)
        for (int k = 0; k < comm->procgrid[2]; k++)
          rspace_grid2proc[i][j][k] = comm->grid2proc[i][j][k];
  }
  MPI_Bcast(&rspace_grid2proc[0][0][0],
            rspace_procgrid[0]*rspace_procgrid[1]*rspace_procgrid[2],MPI_INT,
            universe->root_proc[0],universe->uworld);

  // Kspace partition must be multiple of Rspace partition in each dim
  // so atoms of one Rspace proc coincide with atoms of several Kspace procs

  if (!master) {
    int flag = 0;
    if (comm->procgrid[0] % rspace_procgrid[0]) flag = 1;
    if (comm->procgrid[1] % rspace_procgrid[1]) flag = 1;
    if (comm->procgrid[2] % rspace_procgrid[2]) flag = 1;
    if (flag)
      error->one(FLERR,
                 "Verlet/split/dplr requires Kspace partition layout be "
                 "multiple of Rspace partition layout in each dim");
  }

  // block = 1 Rspace proc with set of Kspace procs it overlays
  // me_block = 0 for Rspace proc
  // me_block = 1 to ratio for Kspace procs
  // block = MPI communicator for that set of procs

  int iblock,key;

  if (master) {
    iblock = comm->me;
    key = 0;
  } else {
    int kpx = comm->myloc[0] / (comm->procgrid[0]/rspace_procgrid[0]);
    int kpy = comm->myloc[1] / (comm->procgrid[1]/rspace_procgrid[1]);
    int kpz = comm->myloc[2] / (comm->procgrid[2]/rspace_procgrid[2]);
    iblock = rspace_grid2proc[kpx][kpy][kpz];
    key = 1;
  }

  MPI_Comm_split(universe->uworld,iblock,key,&block);
  MPI_Comm_rank(block,&me_block);

  // output block groupings to universe screen/logfile
  // bmap is ordered by block and then by proc within block

  int *bmap = new int[universe->nprocs];
  for (int i = 0; i < universe->nprocs; i++) bmap[i] = -1;
  bmap[iblock*(ratio+1)+me_block] = universe->me;

  int *bmapall = new int[universe->nprocs];
  MPI_Allreduce(bmap,bmapall,universe->nprocs,MPI_INT,MPI_MAX,universe->uworld);

  if (universe->me == 0) {
    if (universe->uscreen) {
      fprintf(universe->uscreen,
              "Per-block Kspace/Rspace proc IDs (original proc IDs):\n");
      int m = 0;
      for (int i = 0; i < universe->nprocs/(ratio+1); i++) {
        fprintf(universe->uscreen,"  block %d:",i);
        int rspace_proc = bmapall[m];
        for (int j = 1; j <= ratio; j++)
          fprintf(universe->uscreen," %d",bmapall[m+j]);
        fprintf(universe->uscreen," %d",rspace_proc);
        rspace_proc = bmapall[m];
        for (int j = 1; j <= ratio; j++) {
          if (j == 1) fprintf(universe->uscreen," (");
          else fprintf(universe->uscreen," ");
          fprintf(universe->uscreen,"%d",
                  universe->uni2orig[bmapall[m+j]]);
        }
        fprintf(universe->uscreen," %d)\n",universe->uni2orig[rspace_proc]);
        m += ratio + 1;
      }
    }
    if (universe->ulogfile) {
      fprintf(universe->ulogfile,
              "Per-block Kspace/Rspace proc IDs (original proc IDs):\n");
      int m = 0;
      for (int i = 0; i < universe->nprocs/(ratio+1); i++) {
        fprintf(universe->ulogfile,"  block %d:",i);
        int kspace_proc = bmapall[m];
        for (int j = 1; j <= ratio; j++)
          fprintf(universe->ulogfile," %d",bmapall[m+j]);

        fprintf(universe->ulogfile," %d",kspace_proc);
        kspace_proc = bmapall[m];
        for (int j = 1; j <= ratio; j++) {
          if (j == 1) fprintf(universe->ulogfile," (");
          else fprintf(universe->ulogfile," ");
          fprintf(universe->ulogfile,"%d",
                  universe->uni2orig[bmapall[m+j]]);
        }
        fprintf(universe->ulogfile," %d)\n",universe->uni2orig[kspace_proc]);
        m += ratio + 1;
      }
    }
  }

  memory->destroy(rspace_grid2proc);
  delete [] bmap;
  delete [] bmapall;

  // size/disp = vectors for MPI gather/scatter within block

  qsize = new int[ratio+1];
  qdisp = new int[ratio+1];
  xsize = new int[ratio+1];
  xdisp = new int[ratio+1];

  // f_kspace = Rspace copy of Kspace forces
  // allocate dummy version for Kspace partition

  maxatom = 0;
  f_kspace = nullptr;
  //if (master) memory->create(f_kspace,1,1,"verlet/split:f_kspace");
  memory->create(f_kspace,1,1,"verlet/split:f_kspace");

  //for sort rspace atoms
  maxnext = maxbin = 0;
  binhead = nullptr;
  next = permute = nullptr;

  atom_counts = new int[ratio+1];


  kspace_sublo_list= nullptr;
  kspace_subhi_list= nullptr;

  sortbin_lo= nullptr;
  sortbin_hi= nullptr;

  // Instead of using new/delete, use memory->create/destroy
  memory->create(kspace_sublo_list, ratio, 3, "verlet/split/kspace:kspace_sublo_list");
  memory->create(kspace_subhi_list, ratio, 3, "verlet/split/kspace:kspace_subhi_list");

  memory->create(sortbin_lo, 3, ratio, "verlet/split/kspace:sortbin_lo");
  memory->create(sortbin_hi, 3, ratio, "verlet/split/kspace:sortbin_hi");

}

/* ---------------------------------------------------------------------- */

VerletSplitDPLR::~VerletSplitDPLR()
{
  delete [] qsize;
  delete [] qdisp;
  delete [] xsize;
  delete [] xdisp;
  memory->destroy(f_kspace);
  MPI_Comm_free(&block);


  memory->destroy(next);
  memory->destroy(permute);
  memory->destroy(binhead);

  delete [] atom_counts;

  memory->destroy(kspace_sublo_list);
  memory->destroy(kspace_subhi_list);
  memory->destroy(sortbin_lo);
  memory->destroy(sortbin_hi);
}

/* ----------------------------------------------------------------------
   communicate and sum Kspace atom forces back to Rspace
------------------------------------------------------------------------- */

void VerletSplitDPLR::k2r_comm()
{  
  int n = 0;
  if (!master) n = atom->nlocal;
  // eflag = 1;
  // vflag = 1;
  if (eflag) MPI_Bcast(&force->kspace->energy,1,MPI_DOUBLE,1,block);
  if (vflag) MPI_Bcast(force->kspace->virial,6,MPI_DOUBLE,1,block);

  if (!force->kspace_match("pppm/dplr", 1)) {
    // force!!!
    MPI_Gatherv(atom->f[0],n*3,MPI_DOUBLE,f_kspace[0],xsize,xdisp,
                MPI_DOUBLE,0,block);

    if (master) {
      double **f = atom->f;
      int nlocal = atom->nlocal;
      for (int i = 0; i < nlocal; i++) {
        f[i][0] += f_kspace[i][0];
        f[i][1] += f_kspace[i][1];
        f[i][2] += f_kspace[i][2];
      }
    }
  }
  else
  {
    PPPMDPLR *pppm_dplr = (PPPMDPLR *)force->kspace_match("pppm/dplr", 1);
    if (!pppm_dplr) {
        error->all(FLERR, "Invalid KSpace style for pppm/dplr");
    }
    double *fe = &pppm_dplr->get_fele()[0];

    MPI_Gatherv(fe,n*3,MPI_DOUBLE,f_kspace[0],xsize,xdisp,
              MPI_DOUBLE,0,block);

    if (master) {
      int nlocal = atom->nlocal;
      for (int i = 0; i < nlocal; i++) {
        fe[i*3+0] = f_kspace[i][0];
        fe[i*3+1] = f_kspace[i][1];
        fe[i*3+2] = f_kspace[i][2];
      }
    }
  }
}
