# `verlet/split` for MLP

## Introduction

[placeholder]

## Usage

```bash
export LAMMPS_PLUGIN_PATH=/path/to/plugin:$LAMMPS_PLUGIN_PATH
mpirun -np 5 -p 1 4 lmp_mpi -i input.lmp
```

## Code structure

<img width="1162" alt="image" src="https://github.com/user-attachments/assets/912ccba3-cd68-4385-87d6-1aeb7527cdd2">

- Red color indicates new functions in vert/split/dplr.
- Blue color highlights GPU-intensive processes and loop flow.
