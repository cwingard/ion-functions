# CTD

Functions for the OOI CTD instrument family: SBE 16Plus, SBE 37IM, SBE 52MP,
and glider-mounted CTDs (CTDBP, CTDMO, CTDPF, CTDGV).

## Background

OOI deployed Sea-Bird Electronics CTD instruments across four instrument
classes — CTDBP, CTDMO, CTDPF, and CTDGV — covering fixed moorings,
McLane Moored Profilers, and autonomous gliders. Each class produces the
same five data products: three L1 sensor products (TEMPWAT, PRESWAT,
CONDWAT) and two L2 derived products (PRACSAL, DENSITY).

### Temperature (TEMPWAT_L1)

Water temperature is calculated from raw hexadecimal counts output by the
CTD. The conversion differs by instrument make/model.

**SBE 16Plus** (CTDBP, CTDPF series A/B): The raw 24-bit A/D count `t0`
is converted through two intermediate steps before the calibration equation
is applied:

```
MV = (t0 − 524288) / 1.6 × 10⁷
R  = (MV × 2.9 × 10⁹ + 1.024 × 10⁸) / (2.048 × 10⁴ − MV × 2.0 × 10⁵)
T [°C] = 1 / (a0 + a1·ln R + a2·ln²R + a3·ln³R) − 273.15
```

The four coefficients (a0–a3) are provided on the individual instrument
calibration sheet and stored as metadata. Instrument accuracy is ±0.005 °C
with a stability of ±0.0002 °C/month.

**SBE 37IM telemetered / recovered_host** (CTDMO): The instrument outputs
engineering units encoded as a scaled integer. The L1 conversion is:

```
T [°C] = t0 / 10000 − 10
```

No calibration coefficients are required. Accuracy is ±0.002 °C with a
stability of ±0.0002 °C/month.

**SBE 37IM instrument-recovered** (CTDMO): When data are recovered directly
from instrument memory, raw resistance counts are stored before the onboard
conversion. The calibration equation is applied directly to the raw counts:

```
T [°C] = 1 / (a0 + a1·ln t0 + a2·ln²t0 + a3·ln³t0) − 273.15
```

The same four factory calibration coefficients are required, but the
intermediate MV and R steps are absent. This algorithm was not included in
the TEMPWAT DPS as of June 2016.

**SBE 52MP** (CTDPF series C/K/L): The profiler outputs a scaled integer;
the L1 conversion is:

```
T [°C] = t0 / 10000 − 5
```

**Glider CTDs** (CTDGV): Glider-mounted CTDs are processed onboard the
vehicle with vendor software and transmit temperature already in decimal
degrees Celsius; no L0-to-L1 conversion is performed by ion-functions.

All OOI CTDs are configured to use the ITS-90 temperature scale (Sea-Bird
App Note 42). The older IPTS-68 scale relates to ITS-90 by
T₆₈ = 1.00024 × T₉₀; this correction is not applied because all deployed
instruments were calibrated to ITS-90 standards.

### Pressure (PRESWAT_L1)

Pressure is calculated from raw hexadecimal counts and reported as sea
pressure in decibars (absolute pressure minus one standard atmosphere).
The conversion differs by instrument make/model and, for the SBE 16Plus,
by pressure sensor type.

**SBE 16Plus strain-gauge** (CTDBP most series, CTDPF series A/B): The
calibration uses twelve coefficients from the instrument calibration sheet —
three for the pressure thermistor polynomial (PTEMPA0–2), six for
temperature compensation of the bridge response (PTCA0–2 and PTCB0–2), and
three for the pressure polynomial (PA0–2):

```
tv       = t0 / 13107
t        = PTEMPA0 + PTEMPA1·tv + PTEMPA2·tv²
x        = p0 − PTCA0 − PTCA1·t − PTCA2·t²
n        = x·PTCB0 / (PTCB0 + PTCB1·t + PTCB2·t²)
p_psi    = PA0 + PA1·n + PA2·n²
P [dbar] = p_psi × 0.689475729 − 10.1325
```

The psi-to-dbar conversion and the one-atmosphere subtraction follow the
TEOS-10 convention (101325 Pa = 10.1325 dbar). An optional `offset`
parameter corrects for a known systematic Druck sensor offset error on some
deployments; it is added to the final pressure in dbar. This correction was
introduced in March 2017 (see History tables).

**SBE 16Plus digiquartz** (CTDBP-N and CTDBP-O only): Ten calibration
coefficients (C1–3, D1–2, T1–5) are used. The raw pressure count is first
converted to frequency `pf` [Hz] and the thermistor count to a temperature
`U` [°C], then:

```
C        = C1 + C2·U + C3·U²
D        = D1 + D2·U
T0       = T1 + T2·U + T3·U² + T4·U³ + T5·U⁴
T        = (1 / pf) × 10⁶          (resonator period, µs)
p_psi    = C × (1 − T0²/T²) × (1 − D × (1 − T0²/T²))
P [dbar] = p_psi × 0.689475729 − 10.1325
```

**SBE 37IM telemetered / recovered_host** (CTDMO): The instrument outputs a
scaled integer. A single metadata coefficient `P_range` (the instrument's
full-scale pressure range in psia, stored in the instrument) is required:

```
P_range [dbar] = (P_range [psia] − 14.7) × 0.6894757
P [dbar]       = p0 × P_range / (0.85 × 65536) − 0.05 × P_range
```

**SBE 37IM instrument-recovered** (CTDMO): The same twelve-coefficient
strain-gauge equation as the SBE 16Plus strain-gauge is applied to the raw
counts, but the thermistor count enters the temperature polynomial directly
rather than first being converted to volts. This algorithm was not included
in the PRESWAT DPS as of June 2016.

**SBE 52MP** (CTDPF series C/K/L):

```
P [dbar] = p0 / 100 − 10
```

**Glider CTDs** (CTDGV): Pressure is reported in bar by the onboard
processing; the conversion to dbar is multiplication by 10.

Instrument pressure accuracy for all variants is 0.1% of the full-scale
range, with a stability of 0.05–0.1% of full-scale per year.

### Conductivity (CONDWAT_L1)

Conductivity is calculated from raw hexadecimal counts. Because the
geometry of the conductivity cell deforms slightly with changes in
temperature and pressure, the calibration equation for the SBE 16Plus
includes explicit correction terms for both (see Sea-Bird App Note 10).

**SBE 16Plus** (CTDBP, CTDPF series A/B): The raw count is converted to
frequency in kHz; six calibration coefficients from the instrument
calibration sheet are then applied:

```
f [kHz]   = (c0 / 256) / 1000
C [S m⁻¹] = (g + h·f² + i·f³ + j·f⁴) / (1 + CTcor·T + CPcor·P)
```

where T is TEMPWAT_L1 [°C] and P is PRESWAT_L1 [dbar]. Instrument accuracy
is ±0.0005 S m⁻¹ with a stability of ±0.0003 S m⁻¹/month.

**SBE 37IM telemetered / recovered_host** (CTDMO): The instrument outputs
engineering units encoded as a scaled integer:

```
C [S m⁻¹] = c0 / 100000 − 0.5
```

No calibration coefficients are required.

**SBE 37IM instrument-recovered** (CTDMO): The same polynomial form as the
SBE 16Plus is applied, with the addition of a thermal mass correction
coefficient `wbotc`:

```
f [kHz]   = (c0 / 256) / 1000 × √(1 + wbotc·T)
C [S m⁻¹] = (g + h·f² + i·f³ + j·f⁴) / (1 + CTcor·T + CPcor·P)
```

This algorithm was not included in the CONDWAT DPS as of June 2016.

**SBE 52MP** (CTDPF series C/K/L):

```
C [mmho cm⁻¹] = c0 / 10000 − 0.5
C [S m⁻¹]     = 0.1 × C [mmho cm⁻¹]
```

### Practical Salinity (PRACSAL_L2)

Practical salinity is computed from L1 conductivity, temperature, and
pressure using the Practical Salinity Scale 1978 (PSS-78), implemented via
the TEOS-10 Gibbs Seawater (GSW) Python library function
`SP_from_C(C, t, p)`. Conductivity from the L1 product (S m⁻¹) is
converted to mS cm⁻¹ before the call:

```
C [mS cm⁻¹] = C [S m⁻¹] × 10
SP           = gsw.SP_from_C(C, t, p)
```

The PSS-78 algorithm is valid in the range 2 < SP < 42 and
−2 °C < T < 35 °C. Below SP = 2 the GSW library automatically substitutes
a modified Hill et al. (1986) formula, adjusted to be continuous with PSS-78
at SP = 2. Practical salinity is dimensionless and is not interchangeable
with Absolute Salinity. Expected accuracy is ±0.005 for deployments below
200 m and ±0.01 near the surface, driven primarily by conductivity accuracy.

PSS-78 and the UNESCO EOS-80 standard share the same mathematical definition
of practical salinity, so TEOS-10 and EOS-80 salinity values are numerically
equivalent; only the density calculation differs between the two standards.

### Density (DENSITY_L2)

In-situ density is computed using the TEOS-10 standard, adopted by the
Intergovernmental Oceanographic Commission (IOC) in 2010, implemented via
the GSW library. The computation proceeds in three steps:

1. **Practical salinity → Absolute Salinity (SA):**
   `gsw.SA_from_SP(SP, p, lon, lat)` applies a geographically-interpolated
   Absolute Salinity anomaly correction (SAAR lookup table) to convert
   PSS-78 salinity to Absolute Salinity in g kg⁻¹.
2. **In-situ temperature → Conservative Temperature (CT):**
   `gsw.CT_from_t(SA, T, p)` converts ITS-90 in-situ temperature to
   Conservative Temperature.
3. **Density:** `gsw.rho(SA, CT, p)` evaluates a computationally efficient
   48-term expression (McDougall et al., 2011).

Latitude and longitude are required inputs for the SAAR lookup in step 1.
For moored instruments (CTDBP, CTDMO), these are the fixed mooring
coordinates from deployment metadata. For glider-mounted instruments
(CTDGV), the GPS position at each profile point is used.

Sea-Bird's proprietary SeaSoft software computes density using EOS-80 rather
than TEOS-10 and will produce different values; the TEOS-10 implementation
is the OOI standard. Expected accuracy is approximately ±0.021–0.022 kg m⁻³,
dominated by pressure uncertainty at depth.

## Core functions

::: ion_functions.data.ctd_functions.ctd_sbe16plus_tempwat

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe16plus_tempwat` | 2013-04-12 | Luke Campbell | Initial implementation |
| `ctd_sbe16plus_tempwat` | 2013-04-12 | Christopher Wingard | Minor edits |
| `ctd_sbe16plus_tempwat` | 2013-05-10 | Christopher Wingard | Minor comment edits |
| `ctd_sbe16plus_tempwat` | 2014-01-31 | Russell Desiderio | Standardized comment format |
| `ctd_sbe16plus_tempwat` | 2023-08-15 | Samuel Dahlberg | Removed numexpr |

---

::: ion_functions.data.ctd_functions.ctd_sbe37im_tempwat_instrument_recovered

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe37im_tempwat_instrument_recovered` | 2016-06-16 | Russell Desiderio | Initial implementation |
| `ctd_sbe37im_tempwat_instrument_recovered` | 2023-08-15 | Samuel Dahlberg | Removed numexpr |

---

::: ion_functions.data.ctd_functions.ctd_sbe37im_tempwat

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe37im_tempwat` | 2014-02-05 | Russell Desiderio | Initial implementation |

---

::: ion_functions.data.ctd_functions.ctd_sbe52mp_tempwat

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe52mp_tempwat` | 2014-02-17 | Russell Desiderio | Initial implementation |

---

::: ion_functions.data.ctd_functions.ctd_sbe16plus_preswat

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe16plus_preswat` | 2013-04-12 | Christopher Wingard | Initial implementation |
| `ctd_sbe16plus_preswat` | 2013-05-10 | Christopher Wingard | Minor comment edits |
| `ctd_sbe16plus_preswat` | 2014-01-31 | Russell Desiderio | Standardized comment format |
| `ctd_sbe16plus_preswat` | 2017-03-31 | Dan Mergens | Added Druck sensor offset correction |

---

::: ion_functions.data.ctd_functions.ctd_sbe16digi_preswat

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe16digi_preswat` | 2013-05-10 | Christopher Wingard | Initial implementation |
| `ctd_sbe16digi_preswat` | 2014-01-31 | Russell Desiderio | Standardized comment format; switched from pressure counts to pressure [Hz] per SBE 16Plus V2 manual p. 57 item 5 |

---

::: ion_functions.data.ctd_functions.ctd_sbe37im_preswat_instrument_recovered

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe37im_preswat_instrument_recovered` | 2016-06-16 | Russell Desiderio | Initial implementation |

---

::: ion_functions.data.ctd_functions.ctd_sbe37im_preswat

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe37im_preswat` | 2014-02-05 | Russell Desiderio | Initial implementation |

---

::: ion_functions.data.ctd_functions.ctd_glider_preswat

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_glider_preswat` | 2015-10-28 | Russell Desiderio | Initial implementation |

---

::: ion_functions.data.ctd_functions.ctd_sbe52mp_preswat

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe52mp_preswat` | 2014-02-17 | Russell Desiderio | Initial implementation |

---

::: ion_functions.data.ctd_functions.ctd_sbe16plus_condwat

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe16plus_condwat` | 2013-04-12 | Christopher Wingard | Initial implementation |
| `ctd_sbe16plus_condwat` | 2013-05-10 | Christopher Wingard | Minor comment edits |
| `ctd_sbe16plus_condwat` | 2014-01-31 | Russell Desiderio | Standardized comment format |

---

::: ion_functions.data.ctd_functions.ctd_sbe37im_condwat_instrument_recovered

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe37im_condwat_instrument_recovered` | 2016-06-16 | Russell Desiderio | Initial implementation |
| `ctd_sbe37im_condwat_instrument_recovered` | 2023-08-15 | Samuel Dahlberg | Removed numexpr |

---

::: ion_functions.data.ctd_functions.ctd_sbe37im_condwat

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe37im_condwat` | 2014-02-05 | Russell Desiderio | Initial implementation |

---

::: ion_functions.data.ctd_functions.ctd_sbe52mp_condwat

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe52mp_condwat` | 2014-02-17 | Russell Desiderio | Initial implementation |

---

::: ion_functions.data.ctd_functions.ctd_pracsal

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_pracsal` | 2013-03-13 | Christopher Wingard | Initial implementation |
| `ctd_pracsal` | 2013-05-10 | Christopher Wingard | Minor comment edits |
| `ctd_pracsal` | 2014-01-31 | Russell Desiderio | Standardized comment format |
| `ctd_pracsal` | 2023-08-15 | Samuel Dahlberg | Replaced pygsw with GSW library |

---

::: ion_functions.data.ctd_functions.ctd_density

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_density` | 2013-03-11 | Christopher Mueller | Initial implementation |
| `ctd_density` | 2013-03-13 | Christopher Wingard | Added commenting; moved to ctd_functions |
| `ctd_density` | 2013-05-10 | Christopher Wingard | Minor comment edits |
| `ctd_density` | 2014-01-31 | Russell Desiderio | Standardized comment format |
| `ctd_density` | 2023-08-15 | Samuel Dahlberg | Replaced pygsw with GSW library |

## References

[OOI (2012). Data Product Specification for Water Temperature.
Document Control Number 1341-00010.](https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00010_Data_Product_SPEC_TEMPWAT_OOI.pdf)

[OOI (2012). Data Product Specification for Pressure (Depth).
Document Control Number 1341-00020.](https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00020_Data_Product_SPEC_PRESWAT_OOI.pdf)

[OOI (2012). Data Product Specification for Conductivity.
Document Control Number 1341-00030.](https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00030_Data_Product_SPEC_CONDWAT_OOI.pdf)

[OOI (2012). Data Product Specification for Salinity.
Document Control Number 1341-00040.](https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00040_Data_Product_SPEC_PRACSAL_OOI.pdf)

[OOI (2012). Data Product Specification for Density.
Document Control Number 1341-00050.](https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00050_Data_Product_SPEC_DENSITY_OOI.pdf)


