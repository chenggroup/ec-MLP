export LAMMPS_PLUGIN_PATH=/home/jxzhu/workspace/softwares/ec-mlp/src/lmp/build:$LAMMPS_PLUGIN_PATH

# lmp_mpi -i input.lmp
mpirun -np 2 lmp_mpi -i input.lmp -p 1 1 
