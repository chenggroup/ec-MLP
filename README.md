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
# adapt LAMMPS_SOURCE_DIR and DEEPMD_SOURCE_DIR
cmake -DLAMMPS_SOURCE_DIR=$LAMMPS_PREFIX/src \
      -DDEEPMD_SOURCE_DIR=$deepmd_source_dir/source/lmp \
      ..
make
```
