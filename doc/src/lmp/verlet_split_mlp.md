# `verlet/split` for MLP

## Introduction

[placeholder]

## Usage

Lammps input:

```bash
# use original kspace_style (compatible with verlet/split) rather than pppm/dplr
kspace_style    pppm 1e-5
# ...

# add and use run_style verlet/split/dplr before the run command
add_run_style   verlet/split/dplr
run_style       verlet/split/dplr

timestep        0.0005
run             100
```

Run Lammps with two partitions:

```bash
# add the plugin path to the environment variable
export LAMMPS_PLUGIN_PATH=/path/to/plugin:$LAMMPS_PLUGIN_PATH

# mpirun -np [total np] -p [np for rspace] [np for kspace] lmp_mpi -i input.lmp
mpirun -np 5 -p 1 4 lmp_mpi -i input.lmp
```

Note: the number of processors for the kspace partition should be either the same or an integer multiple of number of processors for the rspace partition.

## Code structure

<img width="1162" alt="image" src="https://github.com/user-attachments/assets/912ccba3-cd68-4385-87d6-1aeb7527cdd2">

- Red color indicates new functions in verlet/split/dplr.
- Blue color highlights GPU-intensive processes and loop flow.
