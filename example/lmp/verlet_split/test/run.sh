conda activate deepmd

export LAMMPS_PLUGIN_PATH=/path/to/plugin:$LAMMPS_PLUGIN_PATH

mpirun -np 2 lmp_mpi -i input.lmp -p 1 1 
