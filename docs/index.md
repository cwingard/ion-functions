# ion-functions

Python library of oceanographic data processing functions for the
[Ocean Observatories Initiative (OOI)](https://oceanobservatories.org/).

ion-functions transforms raw instrument data (L0) into calibrated scientific
data products at L1 and L2 levels, covering CTD, ADCP, velocity, dissolved
oxygen, CO2, fluorometry, pH, pressure, meteorology, and more.

## Instrument families

| Module | Instruments | Data products |
|--------|-------------|---------------|
| [Hydrophone](api/hyd_functions.md) | HYDBB, HYDLF | HYDAPBB, HYDAPLF |

## Data product levels

- **L0** — Raw instrument output (counts, voltages)
- **L1** — Converted/calibrated engineering units
- **L2** — Derived scientific products (e.g., practical salinity, dissolved oxygen concentration)

## Quick start

See [Getting Started](getting_started.md) for installation and usage.
