# CTD

Functions for the OOI CTD instrument family: SBE 16Plus, SBE 37IM, SBE 52MP,
and glider-mounted CTDs (CTDBP, CTDMO, CTDPF, CTDGV).

## Background

OOI deployed four Sea-Bird Electronics CTD instrument families, each requiring
a distinct set of calibration algorithms to convert raw output to L1 data
products. The L2 products (practical salinity and density) are computed from
L1 inputs using the [TEOS-10](https://www.teos-10.org/) Gibbs Seawater (GSW)
library.

**SBE 16Plus** (CTDBP, CTDPF series A/B) outputs raw temperature and
conductivity counts and—depending on sensor type—either strain-gauge or
digiquartz pressure counts. The strain-gauge pressure equation applies to most
CTDBP series; the digiquartz equation applies to CTDBP-N and CTDBP-O only.

**SBE 37IM** (CTDMO) has two distinct L0 data streams: telemetered and
recovered_host data require different calibration equations than
instrument-recovered data. Use the `*_instrument_recovered` variants for data
recovered directly from the instrument.

**SBE 52MP** (CTDPF series C/K/L) is integrated into the McLane Moored
Profiler and delivers temperature, pressure, and conductivity as scaled integer
counts with simple linear conversions.

**Glider CTDs** (CTDGV) on Slocum and Spray gliders report pressure in bar;
the single conversion function rescales to dbar.

## Core functions

### ctd_sbe16plus_tempwat

::: ion_functions.data.ctd_functions.ctd_sbe16plus_tempwat

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe16plus_tempwat` | 2013-04-12 | Luke Campbell | Initial implementation |
| `ctd_sbe16plus_tempwat` | 2013-04-12 | Christopher Wingard | Minor edits |
| `ctd_sbe16plus_tempwat` | 2013-05-10 | Christopher Wingard | Minor comment edits |
| `ctd_sbe16plus_tempwat` | 2014-01-31 | Russell Desiderio | Standardized comment format |
| `ctd_sbe16plus_tempwat` | 2023-08-15 | Samuel Dahlberg | Removed numexpr |

#### References

OOI (2012). Data Product Specification for Water Temperature.
Document Control Number 1341-00010.
[https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00010_Data_Product_SPEC_TEMPWAT_OOI.pdf](https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00010_Data_Product_SPEC_TEMPWAT_OOI.pdf)

---

### ctd_sbe37im_tempwat_instrument_recovered

::: ion_functions.data.ctd_functions.ctd_sbe37im_tempwat_instrument_recovered

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe37im_tempwat_instrument_recovered` | 2016-06-16 | Russell Desiderio | Initial implementation |
| `ctd_sbe37im_tempwat_instrument_recovered` | 2023-08-15 | Samuel Dahlberg | Removed numexpr |

#### References

OOI (2012). Data Product Specification for Water Temperature.
Document Control Number 1341-00010.
[https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00010_Data_Product_SPEC_TEMPWAT_OOI.pdf](https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00010_Data_Product_SPEC_TEMPWAT_OOI.pdf)

---

### ctd_sbe37im_tempwat

::: ion_functions.data.ctd_functions.ctd_sbe37im_tempwat

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe37im_tempwat` | 2014-02-05 | Russell Desiderio | Initial implementation |

#### References

OOI (2012). Data Product Specification for Water Temperature.
Document Control Number 1341-00010.
[https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00010_Data_Product_SPEC_TEMPWAT_OOI.pdf](https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00010_Data_Product_SPEC_TEMPWAT_OOI.pdf)

---

### ctd_sbe52mp_tempwat

::: ion_functions.data.ctd_functions.ctd_sbe52mp_tempwat

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe52mp_tempwat` | 2014-02-17 | Russell Desiderio | Initial implementation |

#### References

OOI (2012). Data Product Specification for Water Temperature.
Document Control Number 1341-00010.
[https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00010_Data_Product_SPEC_TEMPWAT_OOI.pdf](https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00010_Data_Product_SPEC_TEMPWAT_OOI.pdf)

---

### ctd_sbe16plus_preswat

::: ion_functions.data.ctd_functions.ctd_sbe16plus_preswat

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe16plus_preswat` | 2013-04-12 | Christopher Wingard | Initial implementation |
| `ctd_sbe16plus_preswat` | 2013-05-10 | Christopher Wingard | Minor comment edits |
| `ctd_sbe16plus_preswat` | 2014-01-31 | Russell Desiderio | Standardized comment format |
| `ctd_sbe16plus_preswat` | 2017-03-31 | Dan Mergens | Added Druck sensor offset correction |

#### References

OOI (2012). Data Product Specification for Pressure (Depth).
Document Control Number 1341-00020.
[https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00020_Data_Product_SPEC_PRESWAT_OOI.pdf](https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00020_Data_Product_SPEC_PRESWAT_OOI.pdf)

---

### ctd_sbe16digi_preswat

::: ion_functions.data.ctd_functions.ctd_sbe16digi_preswat

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe16digi_preswat` | 2013-05-10 | Christopher Wingard | Initial implementation |
| `ctd_sbe16digi_preswat` | 2014-01-31 | Russell Desiderio | Standardized comment format; switched from pressure counts to pressure [Hz] per SBE 16Plus V2 manual p. 57 item 5 |

#### References

OOI (2012). Data Product Specification for Pressure (Depth).
Document Control Number 1341-00020.
[https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00020_Data_Product_SPEC_PRESWAT_OOI.pdf](https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00020_Data_Product_SPEC_PRESWAT_OOI.pdf)

---

### ctd_sbe37im_preswat_instrument_recovered

::: ion_functions.data.ctd_functions.ctd_sbe37im_preswat_instrument_recovered

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe37im_preswat_instrument_recovered` | 2016-06-16 | Russell Desiderio | Initial implementation |

#### References

OOI (2012). Data Product Specification for Pressure (Depth).
Document Control Number 1341-00020.
[https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00020_Data_Product_SPEC_PRESWAT_OOI.pdf](https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00020_Data_Product_SPEC_PRESWAT_OOI.pdf)

---

### ctd_sbe37im_preswat

::: ion_functions.data.ctd_functions.ctd_sbe37im_preswat

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe37im_preswat` | 2014-02-05 | Russell Desiderio | Initial implementation |

#### References

OOI (2012). Data Product Specification for Pressure (Depth).
Document Control Number 1341-00020.
[https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00020_Data_Product_SPEC_PRESWAT_OOI.pdf](https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00020_Data_Product_SPEC_PRESWAT_OOI.pdf)

---

### ctd_glider_preswat

::: ion_functions.data.ctd_functions.ctd_glider_preswat

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_glider_preswat` | 2015-10-28 | Russell Desiderio | Initial implementation |

#### References

OOI (2015). Data Product Specification for Pressure (Depth).
Document Control Number 1341-00020.
[https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00020_Data_Product_SPEC_PRESWAT_OOI.pdf](https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00020_Data_Product_SPEC_PRESWAT_OOI.pdf)

---

### ctd_sbe52mp_preswat

::: ion_functions.data.ctd_functions.ctd_sbe52mp_preswat

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe52mp_preswat` | 2014-02-17 | Russell Desiderio | Initial implementation |

#### References

OOI (2012). Data Product Specification for Pressure (Depth).
Document Control Number 1341-00020.
[https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00020_Data_Product_SPEC_PRESWAT_OOI.pdf](https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00020_Data_Product_SPEC_PRESWAT_OOI.pdf)

---

### ctd_sbe16plus_condwat

::: ion_functions.data.ctd_functions.ctd_sbe16plus_condwat

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe16plus_condwat` | 2013-04-12 | Christopher Wingard | Initial implementation |
| `ctd_sbe16plus_condwat` | 2013-05-10 | Christopher Wingard | Minor comment edits |
| `ctd_sbe16plus_condwat` | 2014-01-31 | Russell Desiderio | Standardized comment format |

#### References

OOI (2012). Data Product Specification for Conductivity.
Document Control Number 1341-00030.
[https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00030_Data_Product_SPEC_CONDWAT_OOI.pdf](https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00030_Data_Product_SPEC_CONDWAT_OOI.pdf)

---

### ctd_sbe37im_condwat_instrument_recovered

::: ion_functions.data.ctd_functions.ctd_sbe37im_condwat_instrument_recovered

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe37im_condwat_instrument_recovered` | 2016-06-16 | Russell Desiderio | Initial implementation |
| `ctd_sbe37im_condwat_instrument_recovered` | 2023-08-15 | Samuel Dahlberg | Removed numexpr |

#### References

OOI (2012). Data Product Specification for Conductivity.
Document Control Number 1341-00030.
[https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00030_Data_Product_SPEC_CONDWAT_OOI.pdf](https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00030_Data_Product_SPEC_CONDWAT_OOI.pdf)

---

### ctd_sbe37im_condwat

::: ion_functions.data.ctd_functions.ctd_sbe37im_condwat

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe37im_condwat` | 2014-02-05 | Russell Desiderio | Initial implementation |

#### References

OOI (2012). Data Product Specification for Conductivity.
Document Control Number 1341-00030.
[https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00030_Data_Product_SPEC_CONDWAT_OOI.pdf](https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00030_Data_Product_SPEC_CONDWAT_OOI.pdf)

---

### ctd_sbe52mp_condwat

::: ion_functions.data.ctd_functions.ctd_sbe52mp_condwat

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_sbe52mp_condwat` | 2014-02-17 | Russell Desiderio | Initial implementation |

#### References

OOI (2012). Data Product Specification for Conductivity.
Document Control Number 1341-00030.
[https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00030_Data_Product_SPEC_CONDWAT_OOI.pdf](https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00030_Data_Product_SPEC_CONDWAT_OOI.pdf)

---

### ctd_pracsal

::: ion_functions.data.ctd_functions.ctd_pracsal

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_pracsal` | 2013-03-13 | Christopher Wingard | Initial implementation |
| `ctd_pracsal` | 2013-05-10 | Christopher Wingard | Minor comment edits |
| `ctd_pracsal` | 2014-01-31 | Russell Desiderio | Standardized comment format |
| `ctd_pracsal` | 2023-08-15 | Samuel Dahlberg | Replaced pygsw with GSW library |

#### References

OOI (2012). Data Product Specification for Salinity.
Document Control Number 1341-00040.
[https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00040_Data_Product_SPEC_PRACSAL_OOI.pdf](https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00040_Data_Product_SPEC_PRACSAL_OOI.pdf)

---

### ctd_density

::: ion_functions.data.ctd_functions.ctd_density

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `ctd_density` | 2013-03-11 | Christopher Mueller | Initial implementation |
| `ctd_density` | 2013-03-13 | Christopher Wingard | Added commenting; moved to ctd_functions |
| `ctd_density` | 2013-05-10 | Christopher Wingard | Minor comment edits |
| `ctd_density` | 2014-01-31 | Russell Desiderio | Standardized comment format |
| `ctd_density` | 2023-08-15 | Samuel Dahlberg | Replaced pygsw with GSW library |

#### References

OOI (2012). Data Product Specification for Density.
Document Control Number 1341-00050.
[https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00050_Data_Product_SPEC_DENSITY_OOI.pdf](https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00050_Data_Product_SPEC_DENSITY_OOI.pdf)
