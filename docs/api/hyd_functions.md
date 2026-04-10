# Hydrophone

Functions for the OOI Broadband Hydrophone (HYDBB) and Low Frequency
Hydrophone (HYDLF) instrument families.

## Background

OOI deployed two passive acoustic instrument types as part of the hydrophone
instrument family, covering complementary frequency bands. Both produce L1
time-series pressure wave data products from raw L0 digitizer output; neither
instrument produces an L2 data product.

### Broadband Hydrophone (HYDBB) — HYDAPBB_L1

The Ocean Sonics icListen HF broadband hydrophone senses passive acoustic
pressure waves from 5 Hz to 100 kHz at 24-bit resolution, streaming data as
WAV files over a 10/100BaseT Ethernet connection. The instrument applies
internal signal conditioning — filtering and preamplifier gain — and an
optional external amplifier gain before digitization. The L0 data product
(HYDAPBB_L0) is the raw WAV file time-series voltage.

HYDBB instruments were deployed on the RSN system at Hydrate Ridge (subsites
PN1A and PN1B) and Axial Seamount (PN3A and PN3B) Hybrid Moorings, and on the
Endurance Array Hybrid Moorings at subsites PN1C (Offshore Benthic Package)
and PN1D (Shelf Benthic Package).

**L0 → L1 conversion (`hyd_bb_acoustic_pwaves`):**

The WAV file encodes the full-scale ±3 V analog range in 24-bit signed
integers. The L1 conversion scales raw WAV values to volts and removes the
external gain:

```
volts = wav × 3
tsv   = volts / gain_linear,   where gain_linear = 10^(gain_dB / 20)
```

The gain parameter (`gain`, in dB) is the external amplifier gain setting read
from the WAV file event header. Internal signal conditioning gain is folded
into the instrument sensitivity (OCVR) and is not removed at L1. The L1
product is therefore a gain-compensated voltage, not an absolute pressure. The
nominal sensitivity of the icListen HF is approximately −176 dB re 1 µPa/V;
because sensitivity varies with frequency, conversion of the time-series to
pressure units is not performed at L1.

The frequency-dependent sensitivity (OCVR) for each deployed unit is provided
on the factory instrument characterization sheet in units of dBV re 1 µPa. No
field or shipboard calibration is performed.

### Low Frequency Hydrophone (HYDLF) — HYDAPLF_L1

The Low Frequency Hydrophone is physically and electrically attached to a
co-located Broadband Seismometer (OBSBB or OBSBK). Its analog output is
digitized by the seismometer's Guralp DM24S3EAM data logger at up to
1000 samples per second with 24-bit depth. Data are transmitted in SEED
blockette format via the SEEDlink protocol, routed through a US Navy data
diversion switch, and delivered to OOI ION for storage. The L0 data product
(HYDAPLF_L0) is the raw integer count stream extracted from the SEED
blockettes.

HYDLF instruments were deployed on the RSN system at Hydrate Ridge (PN1A and
PN1B) and Axial Seamount (PN3A and PN3B).

**L0 → L1 conversion (`hyd_lf_acoustic_pwaves`):**

The Guralp DM24 digitizer has a fixed bit weight of 3.2 µV/count. The L1
conversion is a single scalar multiply:

```
HYDAPLF_L1 [V] = raw [counts] × 3.2 × 10⁻⁶ [V/count]
```

As with HYDAPBB, the instrument sensitivity (OCVR) is frequency-dependent and
is not applied at L1; the output is a calibrated voltage, not an absolute
pressure in Pa or dB re 1 µPa.

**Calibration note:** For both instrument types, L1 removes only the external
gain. Conversion to absolute acoustic pressure requires the frequency-dependent
OCVR calibration curve supplied on the individual instrument characterization
sheet; that step is outside the scope of these functions.

## Core functions

::: ion_functions.data.hyd_functions.hyd_bb_acoustic_pwaves

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `hyd_bb_acoustic_pwaves` | 2014-05-16 | Christopher Wingard | Initial implementation |
| `hyd_bb_acoustic_pwaves` | 2023-08-15 | Samuel Dahlberg | Removed numexpr; updated variable names |

---

::: ion_functions.data.hyd_functions.hyd_lf_acoustic_pwaves

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `hyd_lf_acoustic_pwaves` | 2014-07-09 | Christopher Wingard | Initial implementation |
| `hyd_lf_acoustic_pwaves` | 2023-08-15 | Samuel Dahlberg | Removed numexpr |

## References

OOI (2013). Data Product Specification for Broadband Acoustic Pressure Waves.
Document Control Number 1341-00820. (Not released.)

[OOI (2013). Data Product Specification for Low Frequency Acoustic Pressure
Waves. Document Control Number 1341-00821.](https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00821_Data_Product_SPEC_HYDAPLF_OOI.pdf)
