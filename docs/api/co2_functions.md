# CO2

Functions for the OOI CO2 instrument families: SAMI-II PCO2W (pCO2 in
seawater) and PCO2A (pCO2 in air and surface seawater). Produces CO2THRM_L1,
PCO2WAT_L1, PCO2ATM_L1, PCO2SSW_L1, and CO2FLUX_L2 data products.

## Background

OOI deployed two instrument families to measure the partial pressure of CO2
in the ocean–atmosphere system.

### PCO2W: SAMI-II pCO2 Seawater Sensor

The Sunburst Sensors SAMI-II PCO2W measures pCO2 in seawater using a
colorimetric indicator method based on equilibrium CO2 chemistry. Seawater is
pumped through a membrane where dissolved CO2 equilibrates with an internal
indicator solution containing bromothymol blue (BTB). As dissolved CO2
increases, the BTB solution acidifies, shifting the indicator from blue (base
form) toward yellow (acid form). A dual-wavelength optical measurement at
434 nm and 620 nm tracks this color shift.

The instrument outputs a 14-element light measurement array per record
containing dark counts, reference counts, and absorbance ratios at 434 and
620 nm. Column 6 of this array holds the 434 nm ratio (CO2ABS1_L0);
column 7 holds the 620 nm ratio (CO2ABS2_L0). A thermistor in the
measurement cell provides the in-situ temperature (CO2THRM_L0) used to
correct for temperature-dependent shifts in indicator equilibrium.

#### L0 → L1 data product chain

Two measurement types are produced: type 4 records are pCO2 measurements;
type 5 records are blank measurements taken with DI water (no indicator).
The blank records characterize the instrument's baseline optical response and
are used to correct subsequent pCO2 measurements.

**Thermistor temperature (CO2THRM_L1):** Raw thermistor counts are converted
to temperature using a Steinhart-Hart equation applied in `pco2_thermistor`.
The instrument is available in 12-bit (original) and 14-bit (upgraded) ADC
variants, which differ in their full-scale count value (4096 vs. 16384).

**pCO2 in seawater (PCO2WAT_L1):** Computed in `pco2_calc_pco2` following the
2018 Sunburst Sensors vendor algorithm. The absorbance ratios from the current
record are normalized by the blank absorbance from the most recent type 5
record:

```
A434 = -log10(ratio434 / blank434)
A620 = -log10(ratio620 / blank620)
R = A620 / A434
```

The pCO2 is then derived from R using fixed spectral constants
(e1 = 0.0043, e2 = 2.136, e3 = 0.2105) that replace the earlier
reagent-coefficient formulation:

```
rco2_1 = -log10((R - e1) / (e2 - e3*R))
rco2_2 = (T - calt) * 0.007 + rco2_1
t_coeff = 0.0075778 + 0.0012389*rco2_2 - 0.00048757*rco2_2^2
t_cor_rco2 = rco2_1 + t_coeff * (T - calt)
pCO2 = 10^((-calb + sqrt(calb^2 - 4*cala*(calc - t_cor_rco2))) / (2*cala))
```

where T is the thermistor temperature in counts, and calt, cala, calb, calc
are instrument-specific calibration coefficients from the factory calibration
sheet.

#### Calibration coefficients

The molar absorptivity coefficients ea434, eb434, ea620, eb620 were used in
the original vendor algorithm to compute e1–e3 dynamically. In the current
(2018) formulation, e1–e3 are fixed constants and the molar absorptivity
arguments are accepted by the API but not used.

The four instrument calibration coefficients (calt, cala, calb, calc) are
specific to each SAMI-II unit and are provided on the factory calibration
sheet. calt is a temperature offset [degC]; cala, calb, and calc are the
quadratic equation coefficients for the final pCO2 conversion.

### PCO2A: pCO2 Air-Sea Sensor

The PCO2A instrument family measures the mole fraction of CO2 (XCO2) in two
gas streams — ambient air and headspace-equilibrated surface seawater — using
a non-dispersive infrared (NDIR) CO2 analyzer. A pressure sensor measures
the gas stream pressure (PRESAIR_L0) in the analyzer cell.

**Partial pressure (PCO2ATM_L1, PCO2SSW_L1):** The mole fraction XCO2 [ppm]
is converted to partial pressure [uatm] by `pco2_ppressure`:

```
pCO2 = XCO2 * P / P0
```

where P is the measured gas stream pressure and P0 is standard atmospheric
pressure (1013.25 mbar). This yields PCO2ATM_L1 when applied to the air
stream (XCO2ATM_L0) and PCO2SSW_L1 when applied to the equilibrated
seawater stream (XCO2SSW_L0).

### CO2FLUX_L2: Sea-to-Air CO2 Flux

The sea-to-air CO2 flux (CO2FLUX_L2) is computed in `pco2_co2flux` by
combining PCO2SSW_L1 and PCO2ATM_L1 from a PCO2A instrument with wind speed,
sea surface temperature, and salinity from a co-located METBK instrument.

The algorithm follows three steps:

**Step 1 — Schmidt number:** Sc for CO2 is computed from sea surface
temperature using the Wanninkhof (1992) polynomial fit (Table A1):

```
Sc = 2073.1 - 125.62*T + 3.6276*T^2 - 0.043219*T^3
```

**Step 2 — Gas transfer velocity:** k [cm h^-1] is computed using the
Sweeney et al. (2007) wind speed parameterization:

```
k = 0.27 * u10^2 * sqrt(660 / Sc)
```

k is then converted to m s^-1. The reference Schmidt number of 660
corresponds to CO2 in seawater at 20 degC.

**Step 3 — Solubility and flux:** CO2 solubility K0 [mol atm^-1 m^-3] is
computed from the Weiss (1974) volumetric formulation (Eqn. 12 and Table I).
The flux is:

```
F = k * K0 * (pCO2_water - pCO2_air)
```

Positive flux indicates CO2 outgassing from ocean to atmosphere; negative
flux indicates ocean uptake.

## Core functions

::: ion_functions.data.co2_functions.pco2_thermistor

#### History

| Date | Author | Change |
|---|---|---|
| 2013-04-20 | Christopher Wingard | Initial implementation |
| 2023-01-12 | Mark Steiner | Added sami_bits argument for hardware variants |
| 2023-08-15 | Samuel Dahlberg | Renamed local variables; replaced Numexpr with NumPy |
| 2026-04-10 | Christopher Wingard | Converted to NumPy docstring format; updated documentation |

---

::: ion_functions.data.co2_functions.pco2_battery

#### History

| Date | Author | Change |
|---|---|---|
| 2023-02-23 | Mark Steiner | Initial implementation |
| 2026-04-10 | Christopher Wingard | Converted to NumPy docstring format; updated documentation |

---

::: ion_functions.data.co2_functions.pco2_blank

#### History

| Date | Author | Change |
|---|---|---|
| 2013-04-20 | Christopher Wingard | Initial implementation |
| 2018-03-04 | Christopher Wingard | Updated blank calculation per revised vendor code |
| 2026-04-10 | Christopher Wingard | Converted to NumPy docstring format; updated documentation |

---

::: ion_functions.data.co2_functions.pco2_calc_pco2

#### Additional Notes

The molar absorptivity arguments ea434, eb434, ea620, eb620 are accepted for
API compatibility but are not used in the current (2018) Sunburst Sensors
formulation. These were used in the original algorithm to compute e1–e3
dynamically; they have been replaced by fixed constants. Passing fill values
or zeros for these arguments does not affect the output.

#### History

| Date | Author | Change |
|---|---|---|
| 2013-04-20 | Christopher Wingard | Initial Python implementation (from J. Newton, Sunburst Sensors, original MATLAB) |
| 2018-03-04 | Christopher Wingard | Updated to 2018 vendor formulation with fixed spectral constants; corrected blank handling |
| 2023-01-12 | Mark Steiner | Updated thermistor argument handling |
| 2023-08-15 | Samuel Dahlberg | Renamed local variables to follow naming convention |
| 2026-04-10 | Christopher Wingard | Converted to NumPy docstring format; updated documentation |

---

::: ion_functions.data.co2_functions.pco2_ppressure

#### History

| Date | Author | Change |
|---|---|---|
| 2014-10-27 | Christopher Wingard | Initial implementation |
| 2023-08-15 | Samuel Dahlberg | Removed Numexpr |
| 2026-04-10 | Christopher Wingard | Converted to NumPy docstring format; updated documentation |

---

::: ion_functions.data.co2_functions.pco2_co2flux

#### History

| Date | Author | Change |
|---|---|---|
| 2012-03-28 | Mathias Lankhorst | Original MATLAB implementation |
| 2013-04-20 | Christopher Wingard | Initial Python implementation |
| 2026-04-10 | Christopher Wingard | Converted to NumPy docstring format; updated documentation |

## OOI System Interface

The following functions are single-output wrappers that extract individual
L0 data products from the raw light measurement array, or dispatch the core
pCO2 calculation based on measurement type. They exist to satisfy an internal
OOI system requirement that each data product be addressable by a named
function. External users should call `pco2_calc_pco2` directly.

::: ion_functions.data.co2_functions.pco2_abs434_ratio

#### History

| Date | Author | Change |
|---|---|---|
| 2013-04-20 | Christopher Wingard | Initial implementation |
| 2026-04-10 | Christopher Wingard | Converted to NumPy docstring format; updated documentation |

---

::: ion_functions.data.co2_functions.pco2_abs620_ratio

#### History

| Date | Author | Change |
|---|---|---|
| 2013-04-20 | Christopher Wingard | Initial implementation |
| 2026-04-10 | Christopher Wingard | Converted to NumPy docstring format; updated documentation |

---

::: ion_functions.data.co2_functions.pco2_pco2wat

#### History

| Date | Author | Change |
|---|---|---|
| 2013-04-20 | Christopher Wingard | Initial implementation |
| 2014-03-19 | Christopher Wingard | Optimized per feedback from Chris Fortin |
| 2017-04-04 | Pete Cable | Updated to use thermistor and blank counts per DPS |
| 2026-04-10 | Christopher Wingard | Converted to NumPy docstring format; updated documentation |

## References

[OOI (2012). Data Product Specification for Partial Pressure of CO2 in
Seawater. Document Control Number 1341-00490.](https://oceanobservatories.org/wp-content/uploads/2015/09/1341-00490_Data_Product_SPEC_PCO2WAT_OOI.pdf)

[OOI (2012). Data Product Specification for Partial Pressure of CO2 in Air
and Surface Seawater. Document Control Number 1341-00260.](https://oceanobservatories.org/wp-content/uploads/2014/10/1341-00260_Data_Product_SPEC_PCO2ATM_PCO2SSW_OOI.pdf)

[OOI (2012). Data Product Specification for Flux of CO2 into the Atmosphere.
Document Control Number 1341-00270.](https://oceanobservatories.org/wp-content/uploads/2014/10/1341-00270_Data_Product_SPEC_CO2FLUX_OOI.pdf)

Sweeney, C., E. Gloor, A.R. Jacobson, R.M. Key, G. McKinley, J.L. Sarmiento,
and R. Wanninkhof (2007). Constraining global air-sea gas exchange for CO2
with recent bomb C-14 measurements. Global Biogeochemical Cycles, 21, GB2015.

Wanninkhof, R. (1992). Relationship between wind speed and gas exchange over
the ocean. Journal of Geophysical Research, 97(C5), 7373–7382.

Weiss, R.F. (1974). Carbon dioxide in water and seawater: the solubility of
a non-ideal gas. Marine Chemistry, 2, 203–215.
