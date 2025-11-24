mkdir -p build
cd build

cmake -DLAMMPS_SOURCE_DIR=$LAMMPS_PREFIX/src \
	-DDEEPMD_SOURCE_DIR=$deepmd_source_dir/source/lmp \
	-DCMAKE_PREFIX_PATH=$deepmd_root \
	..
make
