# README

DeepMD-kit plugin for ElectroChemical interfaces simulations.

## Installation

### Python interface (training and inference)

After installing deepmd-kit (python interface), the plugin for python interface can be installed by:

```bash
git clone https://git.xmu.edu.cn/cheng-group/ec-MLP.git
pip install ec-MLP[torch_admp]
```

For testing, you might need to install additional dependencies via

```bash
pip install ec-MLP[torch_admp,test]
cd ec-MLP
python -m unittest discover tests
```

### Lammps interface

For a LAMMPS interface, a patch should be applied to the source code of deepmd-kit:

```bash
# get deepmd-kit source code
git clone -b v3.1.1 https://github.com/deepmodeling/deepmd-kit.git
# add patch for fix dplr
wget -c https://patch-diff.githubusercontent.com/raw/ChiahsinChu/deepmd-kit/pull/1.patch
git am -3 1.patch
# install deepmd-kit from src...
```

After installing deepmd-kit (dp-lmp interface), the plugin can be installed by:

```bash
cd ec-MLP/src/lmp
mkdir -p build && cd build
# $LAMMPS_PREFIX: the path of lammps code (including src, cmake, lib, etc.)
# $deepmd_source_dir: the path of deepmd-kit source code (including deepmd, source, examples, etc.)
# $deepmd_root: the path of deepmd-kit’s C++ interface installed (including bin, include, lib, share, etc.)
cmake -DLAMMPS_SOURCE_DIR=$LAMMPS_PREFIX/src \
      -DDEEPMD_SOURCE_DIR=$deepmd_source_dir/source/lmp \
      -DCMAKE_PREFIX_PATH=$deepmd_root \
      ..
make
```

## Version compatibility

This plugin is compatible with [DeepMD-kit v3.1.1](https://github.com/deepmodeling/deepmd-kit/releases/tag/v3.1.1) and [Lammps Stable release 22 July 2025](https://github.com/lammps/lammps/releases/tag/stable_22Jul2025). Older versions of both softwares do not work.
