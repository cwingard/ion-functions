# Hydrophone

Functions for the OOI Broadband Hydrophone (HYDBB) and Low Frequency
Hydrophone (HYDLF) instrument families.

## Background

OOI deployed two passive acoustic sensor types as part of the hydrophone
instrument family:

**Broadband Hydrophone (HYDBB)** senses acoustic pressure waves from 5 Hz to
100 kHz at 24-bit resolution. Raw output is a time-series voltage, scaled to
a 3 V full-scale range, which is then corrected for external amplifier gain to
produce the HYDAPBB_L1 data product.

**Low Frequency Hydrophone (HYDLF)** is digitized by a Guralp DM24 data
logger with a fixed bit weight of 3.2 µV/count. Raw output is an integer
count, which is multiplied by the bit weight to produce the HYDAPLF_L1 data
product.

Both instruments record passive ambient sound fields in the ocean and are used
to study biological, geophysical, and anthropogenic sound sources. No Data
Product Specification document was published for the HYDAPBB data product;
see [References](#references) for the HYDAPLF specification.

## Core functions

### hyd_bb_acoustic_pwaves

::: ion_functions.data.hyd_functions.hyd_bb_acoustic_pwaves

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `hyd_bb_acoustic_pwaves` | 2014-05-16 | Christopher Wingard | Initial implementation |
| `hyd_bb_acoustic_pwaves` | 2023-08-15 | Samuel Dahlberg | Removed numexpr; updated variable names |

#### References

None

### hyd_lf_acoustic_pwaves

::: ion_functions.data.hyd_functions.hyd_lf_acoustic_pwaves

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `hyd_lf_acoustic_pwaves` | 2014-07-09 | Christopher Wingard | Initial implementation |
| `hyd_lf_acoustic_pwaves` | 2023-08-15 | Samuel Dahlberg | Removed numexpr |

#### References

OOI (2013). Data Product Specification for Low Frequency Acoustic Pressure
Waves. Document Control Number 1341-00821. [https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00821_Data_Product_SPEC_HYDAPLF_OOI.pdf](https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00821_Data_Product_SPEC_HYDAPLF_OOI.pdf)
