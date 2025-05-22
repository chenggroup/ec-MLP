# README

DeepMD-kit plugin for ElectroChemical interfaces simulations.

## Installation

This package should be used in combination with the `devel` branch of DeepMD-kit:

```bash
git clone -b devel https://github.com/deepmodeling/deepmd-kit.git
```

The plugin for python interface can be installed by:

```bash
git clone https://git.xmu.edu.cn/cheng-group/ec-MLP.git
pip install ec-MLP
```

For lammps interface, the plugin can be installed by:

```bash
cd src/lmp
mkdir -p build && cd build
# $LAMMPS_PREFIX: the path of lammps code (including src, cmake, lib, etc.)
# $deepmd_source_dir: the path of deepmd-kit source code (including deepmd, source, examples, etc.)
# $deepmd_root: the path of deepmd-kit’s C++ interface installed
cmake -DLAMMPS_SOURCE_DIR=$LAMMPS_PREFIX/src \
      -DDEEPMD_SOURCE_DIR=$deepmd_source_dir/source/lmp \
      -DCMAKE_PREFIX_PATH=$deepmd_root \
      ..
make
```
