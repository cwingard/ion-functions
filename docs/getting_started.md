# Getting Started

## Installation

ion-functions requires a conda environment. The recommended approach:

```bash
conda env create -f conda_env.yml
conda activate ion
pip install -e .
```

This installs all runtime dependencies (numpy, scipy, gsw, ppigrf, numexpr)
and builds the Cython extensions in-place.

### Cython extensions

Two Cython extensions must be compiled before use:

- `ion_functions/qc/qc_extensions.pyx` — QC algorithms (spike, stuck, gradient)
- `ion_functions/data/polycals.pyx` — Polynomial calibration

To rebuild them explicitly:

```bash
python setup.py build_ext --inplace
```

## Running tests

```bash
# All tests
pytest

# A specific module
pytest ion_functions/data/test/test_ctd_functions.py

# A specific test
pytest ion_functions/data/test/test_ctd_functions.py::CtdFunctionsTestCase::test_ctd_sbe16plus_tempwat
```

## Key dependencies

| Package | Purpose |
|---------|---------|
| numpy | Array operations |
| scipy | Scientific algorithms |
| gsw | TEOS-10 seawater equations |
| ppigrf | International Geomagnetic Reference Field |
| numexpr | Fast numerical expressions |
| cython | C extension compilation |
