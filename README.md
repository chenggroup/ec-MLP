# README

DeepMD-kit plugin for ElectroChemical interfaces simulations.

## Installation

This plugin can be installed by:

```bash
git clone https://git.xmu.edu.cn/cheng-group/ec-MLP.git
pip install ec-MLP
```

This package should be used in combination with a custom version of DeepMD-kit:

```bash
git clone -b devel-modifier-plugin https://github.com/ChiahsinChu/deepmd-kit.git
```

## Usage

### `dipole_charge` modifier with multiple backends

When [training a DPLR model](https://docs.deepmodeling.com/projects/deepmd/en/stable/train/train-input.html#argument:model/modifier), `dipole_charge` modifier should be set. However, the calculation of Ewald reciprocal interaction is implemented for CPU, which would cost considerable time in GPU-based training tasks. Here, we implemented two new calculator (i.e., jax and torch) based on [dmff](https://github.com/deepmodeling/DMFF) and [torch-dmff](https://github.com/chiahsinchu/torch-dmff) packages, respectively, which allow GPU-accelereated calculation of Ewald reciprocal interaction. The usage of these calculators is shown below:

```json
    "modifier": {
      "type": "dipole_charge_beta",
      "model_name": "dw.pb",
      "model_charge_map": [
        -8
      ],
      "sys_charge_map": [
        6,
        1
      ],
      "ewald_h": 1.00,
      "ewald_beta": 0.40,
      "ewald_calculator": "torch"
    },
```

The `type` should be set as `dipole_charge_beta` to use the new calculators. The `ewald_calculator` can be set as `naive`, `jax` or `torch` to use the original CPU-based calculator, the jax-based calculator or the torch-based calculator, respectively.

**Note: virial calculation has not been implemented in the torch-/jax-based calculators yet.**
