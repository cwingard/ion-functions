# Installation

## Requirements

ion-functions is installed into a conda environment built from conda-forge packages. Install
[Miniforge](https://github.com/conda-forge/miniforge), which uses the conda-forge channel by default. The dependencies,
including the documentation tools, are listed in `conda_env.yml` at the repository root. Installing the package
compiles two Cython extensions, so a C compiler is also required (on Windows, the Microsoft C++ Build Tools).

## Installing

```bash
git clone https://github.com/oceanobservatories/ion-functions.git
cd ion-functions
conda env create -f conda_env.yml
conda activate ion
pip install -e .
```

## Running the tests

```bash
pytest
```

## Contributing

To report a bug, propose a change, or set up a development fork, see [Contributing](CONTRIBUTING.md).
