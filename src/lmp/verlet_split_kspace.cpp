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

#include "verlet_split_kspace.h"

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

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

VerletSplitKSpace::VerletSplitKSpace(LAMMPS *lmp, int narg, char **arg) :
  Verlet(lmp, narg, arg), qsize(nullptr), qdisp(nullptr), xsize(nullptr), xdisp(nullptr), f_kspace(nullptr)
{
  // error checks on partitions

  if (universe->nworlds != 2)
    error->universe_all(FLERR,"verlet/split/kspace requires 2 partitions");
  if (universe->procs_per_world[1] % universe->procs_per_world[0])
    error->universe_all(FLERR,"verlet/split/kspace requires Kspace partition "
                        "size be multiple of Rspace partition size");
  if (comm->style != Comm::BRICK)
    error->universe_all(FLERR,"verlet/split/kspace can only currently be used with comm_style brick");

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
                 "verlet/split/kspace requires Kspace partition layout be "
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

VerletSplitKSpace::~VerletSplitKSpace()
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
   initialization before run
------------------------------------------------------------------------- */

void VerletSplitKSpace::init()
{
  if (comm->style != Comm::BRICK)
    error->universe_all(FLERR,"Verlet/split can only currently be used with comm_style brick");
  if (!force->kspace && comm->me == 0)
    error->warning(FLERR,"A KSpace style must be defined with verlet/split");

  // error for as-yet unsupported verlet/split KSpace options

  int errflag = 0;
  if (!atom->q_flag) errflag = 1;
  if (force->kspace->tip4pflag) errflag = 1;
  if (force->kspace->dipoleflag) errflag = 1;
  if (force->kspace->spinflag) errflag = 1;

  if (errflag)
    error->all(FLERR,"Verlet/split cannot (yet) be used with kspace style {}", force->kspace_style);

  // partial support for TIP4P, see where this flag is used below

  tip4pflag = force->kspace->tip4pflag;

  // invoke parent Verlet init

  if(!master) force->pair->compute_flag = 0;
  Verlet::init();

}

/* ----------------------------------------------------------------------
   setup before run
   servant partition only sets up KSpace calculation
------------------------------------------------------------------------- */

void VerletSplitKSpace::setup(int flag)
{
  if (comm->me == 0 && screen)
    fprintf(screen,"Setting up Verlet/split run ...\n");

  if (!master) force->kspace->setup();
  else Verlet::setup(flag);
}

/* ----------------------------------------------------------------------
   setup without output
   flag = 0 = just force calculation
   flag = 1 = reneighbor and force calculation
   servant partition only sets up KSpace calculation
------------------------------------------------------------------------- */

void VerletSplitKSpace::setup_minimal(int flag)
{
  if (!master){force->kspace->setup();
  //force->kspace->compute(eflag,vflag);
  }
  else Verlet::setup_minimal(flag);
}

/* ----------------------------------------------------------------------
   run for N steps
   master partition does everything but Kspace
   servant partition does just Kspace
   communicate back and forth every step:
     atom coords from master -> servant
     kspace forces from servant -> master
     also box bounds from master -> servant if necessary
------------------------------------------------------------------------- */

void VerletSplitKSpace::run(int n)
{
  bigint ntimestep;
  int nflag,sortflag;

  // sync both partitions before start timer
  MPI_Barrier(universe->uworld);
  timer->init();
  timer->barrier_start();

  // setup initial Rspace <-> Kspace comm params
  //rk_setup();

  // check if OpenMP support fix defined
  Fix *fix_omp;
  int ifix = modify->find_fix("package_omp");
  if (ifix < 0) fix_omp = nullptr;
  else fix_omp = modify->fix[ifix];

  // flags for timestepping iterations
  int n_post_integrate = modify->n_post_integrate;
  int n_pre_exchange = modify->n_pre_exchange;
  int n_pre_neighbor = modify->n_pre_neighbor;
  int n_pre_force = modify->n_pre_force;
  int n_pre_reverse = modify->n_pre_reverse;
  int n_post_force = modify->n_post_force_any;
  int n_end_of_step = modify->n_end_of_step;

  if (atom->sortfreq > 0) sortflag = 1;
  else sortflag = 0;

  for (int i = 0; i < n; i++) {

    ntimestep = ++update->ntimestep;

    //error arises when meblock > 0 inherit integrate ,
    //so we use MPI_Send to solve the problem of meblock > 0 can't call ev_set()
    ev_set(ntimestep);

    // initial time integration
    timer->stamp();
    if (master) {
      modify->initial_integrate(vflag);
      if (n_post_integrate) modify->post_integrate();
    }
    timer->stamp(Timer::MODIFY);

    // regular communication vs neighbor list rebuild
    if (master) nflag = neighbor->decide();
    if (ntimestep==0){nflag = 1;}
    if (ntimestep==1){nflag = 1;}

    MPI_Bcast(&nflag,1,MPI_INT,0,block);

    neigh_comm(nflag,n_pre_exchange,n_pre_neighbor);

    // if reneighboring occurred, re-setup Rspace <-> Kspace comm params
    // comm Rspace atom coords to Kspace procs
    timer->stamp();
    if (nflag) rk_setup();
    timer->stamp(Timer::COMM);
    //pre_force of dplr will change coordinates, so r2k need after pre_force
    //r2k_comm();

    // force computations

    force_clear();

    timer->stamp();
    if (master) {if (n_pre_force) modify->pre_force(vflag);}
    timer->stamp(Timer::MODIFY);

    timer->stamp();
    r2k_comm();
    timer->stamp(Timer::COMM);

    if (master) {
      timer->stamp();
      if (force->pair) {
        force->pair->compute(eflag,vflag);
        timer->stamp(Timer::PAIR);
      }

      if (atom->molecular != Atom::ATOMIC) {
        if (force->bond) force->bond->compute(eflag,vflag);
        if (force->angle) force->angle->compute(eflag,vflag);
        if (force->dihedral) force->dihedral->compute(eflag,vflag);
        if (force->improper) force->improper->compute(eflag,vflag);
        timer->stamp(Timer::BOND);
      }

      if (n_pre_reverse) {
        modify->pre_reverse(eflag,vflag);
        timer->stamp(Timer::MODIFY);
      }
      if (force->newton) {
        comm->reverse_comm();
        timer->stamp(Timer::COMM);
      }

    } else {

      // run FixOMP as sole pre_force fix, if defined
      if (fix_omp) fix_omp->pre_force(vflag);

      if (force->kspace) {
        timer->stamp();
        force->kspace->compute(eflag,vflag);
        timer->stamp(Timer::KSPACE);
      }

      if (n_pre_reverse) {
        modify->pre_reverse(eflag,vflag);
        timer->stamp(Timer::MODIFY);
      }

    }

    // comm and sum Kspace forces back to Rspace procs
    timer->stamp();
    k2r_comm();
    timer->stamp(Timer::COMM);
    // force modifications, final time integration, diagnostics
    // all output

    if (master) {
      timer->stamp();
      if (n_post_force) modify->post_force(vflag);

      modify->final_integrate();
      if (n_end_of_step) modify->end_of_step();
      timer->stamp(Timer::MODIFY);

      if (ntimestep == output->next) {
        timer->stamp();
        output->write(ntimestep);
        timer->stamp(Timer::OUTPUT);
      }
    }

  }
}

/* ----------------------------------------------------------------------
   setup params for Rspace <-> Kspace communication
   called initially and after every reneighbor
   also communcicate atom charges from Rspace to KSpace since static
------------------------------------------------------------------------- */

void VerletSplitKSpace::rk_setup()
{
  // grow f_kspace array on master procs if necessary

  if (master) {
    if (atom->nmax > maxatom) {
      memory->destroy(f_kspace);
      maxatom = atom->nmax;
      memory->create(f_kspace,maxatom,3,"verlet/split:f_kspace");
    }
  }

  // qsize = # of atoms owned by each master proc in block

  if (master) {
    for(int i =1; i<= ratio;i++ )
    qsize[i] = atom_counts[i];}
  if (!master) {atom->nlocal = atom_counts[me_block];
      while (atom->nmax <= atom->nlocal) atom->avec->grow(0);
  }
  //qsize = atom_counts;

  // setup qdisp, xsize, xdisp based on qsize
  // only needed by Rspace proc
  // set Rspace nlocal to sum of Kspace nlocals
  // ensure Rspace atom arrays are large enough

  if (master) {
    qsize[0] = qdisp[0] = xsize[0] = xdisp[0] = 0;
    for (int i = 1; i <= ratio; i++) {
      qdisp[i] = qdisp[i-1]+qsize[i-1];
      xsize[i] = 3*qsize[i];
      xdisp[i] = xdisp[i-1]+xsize[i-1];
    }

    //atom->nlocal = qdisp[ratio] + qsize[ratio];
    while (atom->nmax <= atom->nlocal) atom->avec->grow(0);
    //atom->nghost = 0;
  }
  /*if (!master) {
    atom->nghost = 0;
  }*/

  // one-time scatter of Rspace atom charges to Kspace proc
  int n;
  n = atom->nlocal;
  MPI_Scatterv(master ? atom->q : nullptr, qsize, qdisp, MPI_DOUBLE,
              atom->q, n, MPI_DOUBLE, 0, block);

}

void VerletSplitKSpace::neigh_comm(int nflag,int n_pre_exchange,int n_pre_neighbor)
{
      int sortflag;
      bigint ntimestep;

      ntimestep = update->ntimestep;

      // regular communication vs neighbor list rebuild

      if (nflag == 0) {
        if(master){
        timer->stamp();
        comm->forward_comm();
        timer->stamp(Timer::COMM);}
      } else {
          if(master){
            if (n_pre_exchange) modify->pre_exchange();
            if (triclinic) domain->x2lamda(atom->nlocal);
            domain->pbc();
            if (domain->box_change) {
              domain->reset_box();
              comm->setup();
              if (neighbor->style) neighbor->setup_bins();
            }
            timer->stamp();
            comm->exchange();
            if (sortflag && ntimestep >= atom->nextsort) atom->sort();
          }

          //box information communication
          // send box bounds from Rspace to Kspace if simulation box is dynamic
          if (domain->box_change||ntimestep==1||ntimestep==2){

            MPI_Bcast(domain->boxlo, 3, MPI_DOUBLE, 0, block);
            MPI_Bcast(domain->boxhi, 3, MPI_DOUBLE, 0, block);

            if(!master){
              domain->set_global_box();
              domain->set_local_box();}

            setup_kspace_bins(kspace_sublo_list,kspace_subhi_list,sortbin_lo, sortbin_hi);
          }

          //sort atoms by kspace bins
          resort_rspace_atom(sortbin_lo, sortbin_hi);

          if(master){
            comm->borders();
            if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
            timer->stamp(Timer::COMM);
            if (n_pre_neighbor) modify->pre_neighbor();
            neighbor->build(1);
            timer->stamp(Timer::NEIGH);
          }

        }
  // send eflag,vflag from Rspace to Kspace
  MPI_Bcast(&eflag,1,MPI_INT,0,block);
  MPI_Bcast(&vflag,1,MPI_INT,0,block);

}

/* ----------------------------------------------------------------------
   communicate Rspace atom coords to Kspace
   also eflag,vflag and box bounds if needed
------------------------------------------------------------------------- */


void VerletSplitKSpace::r2k_comm()
{
  int n = 0;

  n = atom->nlocal;
  MPI_Scatterv(master ? atom->x[0] : nullptr, xsize, xdisp, MPI_DOUBLE,
              atom->x[0], n * 3, MPI_DOUBLE, 0, block);

  if (domain->box_change && !master) force->kspace->setup();

}

/* ----------------------------------------------------------------------
   communicate and sum Kspace atom forces back to Rspace
------------------------------------------------------------------------- */

void VerletSplitKSpace::k2r_comm()
{
  int n = 0;
  if (!master) n = atom->nlocal;
  //me_block = 1;
  eflag = 1;
  vflag = 1;
  if (eflag) MPI_Bcast(&force->kspace->energy,1,MPI_DOUBLE,1,block);
  if (vflag) MPI_Bcast(force->kspace->virial,6,MPI_DOUBLE,1,block);

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

/* ----------------------------------------------------------------------
   memory usage of Kspace force array on master procs
------------------------------------------------------------------------- */

double VerletSplitKSpace::memory_usage()
{
  double bytes = (double)maxatom*3 * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   We need Rspace process coordinate has same sort with block
------------------------------------------------------------------------- */

void VerletSplitKSpace::setup_kspace_bins(
    double **kspace_sublo_list, double **kspace_subhi_list,
    double **sortbin_lo, double **sortbin_hi) {
  int i;
  if (!master) {
    double kspace_sublo[3], kspace_subhi[3];
    for (int dim = 0; dim < 3; dim++) {
      kspace_sublo[dim] = domain->sublo[dim];
      kspace_subhi[dim] = domain->subhi[dim];
    }
    // Send to Rspace process (me_block = 0)
    MPI_Send(kspace_sublo, 3, MPI_DOUBLE, 0, 2, block);
    MPI_Send(kspace_subhi, 3, MPI_DOUBLE, 0, 3, block);
  }

  if (master) {
    for (i = 1; i <= ratio; i++) {
      MPI_Recv(kspace_sublo_list[i-1], 3, MPI_DOUBLE, i, 2, block, MPI_STATUS_IGNORE);
      MPI_Recv(kspace_subhi_list[i-1], 3, MPI_DOUBLE, i, 3, block, MPI_STATUS_IGNORE);
    }
    // Compute sorting bins
    for (int dim = 0; dim < 3; dim++) {
      for (i = 0; i < ratio; i++) {
        // Take the intersection of Kspace subdomain boundaries with Rspace subdomain
        sortbin_lo[dim][i] = MAX(kspace_sublo_list[i][dim], domain->sublo[dim]);
        sortbin_hi[dim][i] = MIN(kspace_subhi_list[i][dim], domain->subhi[dim]);
      }
    }
  }
}

void VerletSplitKSpace::resort_rspace_atom(double **sortbin_lo, double **sortbin_hi) {
  if(master){
  int i, m, n, ibin, empty;

  int nlocal = atom->nlocal;
  int nmax = atom->nmax;
  double **x = atom->x;
  AtomVec *avec = atom->avec;

  // Ensure 'next' and 'permute' arrays are large enough
  if (nlocal > maxnext) {
    memory->destroy(next);
    memory->destroy(permute);
    maxnext = nmax + 1;  // Add some extra space to prevent frequent reallocations
    memory->create(next, maxnext, "verlet/split/kspace:next");
    memory->create(permute, maxnext, "verlet/split/kspace:permute");
  }

  if (ratio > maxbin) {
    memory->destroy(binhead);
    maxbin = ratio;
    memory->create(binhead, maxbin, "verlet/split/kspace:binhead");
  }

  // Initialize binhead
  for (i = 0; i < ratio; i++) binhead[i] = -1;

  // **Declare and Initialize atom_counts Array**

  for (i = 0; i <= ratio; i++) atom_counts[i] = 0;

  // Assign atoms to bins and count atoms
  for (i = nlocal - 1; i >= 0; i--) {
    // Determine which bin the atom belongs to
    for (ibin = 0; ibin < ratio; ibin++) {
      bool inbin = true;
      for (int dim = 0; dim < 3; dim++) {
        if (x[i][dim] < sortbin_lo[dim][ibin] || x[i][dim] >= sortbin_hi[dim][ibin]) {
          inbin = false;
          break;
        }
      }
      if (inbin) break;
    }
    //if (ibin == ratio) ibin = ratio - 1; // Prevent out-of-bounds

    // **Increment atom count for this bin**
    atom_counts[ibin + 1]++;  // Indices from 1 to ratio

    // Add atom to the bin's linked list
    next[i] = binhead[ibin];
    binhead[ibin] = i;
  }

  // Build 'permute' array
  n = 0;
  for (m = 0; m < ratio; m++) {
    i = binhead[m];
    while (i >= 0) {
      permute[n++] = i;
      i = next[i];
    }
  }

  // Reorder atoms
  int *current = next;
  for (i = 0; i < nlocal; i++) current[i] = i;

  for (i = 0; i < nlocal; i++) {
    if (current[i] == permute[i]) continue;
    avec->copy(i, nlocal, 0);
    empty = i;
    while (permute[empty] != i) {
      avec->copy(permute[empty], empty, 0);
      empty = current[empty] = permute[empty];
    }
    avec->copy(nlocal, empty, 0);
    current[empty] = permute[empty];
  }


  atom_counts[0] = 0;
  MPI_Bcast(atom_counts, ratio + 1, MPI_INT,0,block);}
  else{MPI_Bcast(atom_counts, ratio + 1, MPI_INT,0,block);}

}
