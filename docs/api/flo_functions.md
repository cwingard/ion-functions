# Fluorometer

Functions for the OOI Fluorometer instrument family: WET Labs ECO FLORD
(two-channel) and FLORT (three-channel), including FLNTU variants.

## Background

OOI deployed the WET Labs ECO family of fluorometers to measure three optical
properties of seawater: fluorescence from chlorophyll-a (CHLAFLO), colored
dissolved organic matter fluorescence (CDOMFLO), and volume scattering at
near-infrared wavelengths (FLUBSCT). Two instrument configurations were used:

**FLORD** (two-channel): measures two of the three optical properties, typically
chlorophyll-a fluorescence and backscatter or CDOM. Variants include FLORD D,
G, L, M, and the FLNTU series A.

**FLORT** (three-channel): measures all three optical properties simultaneously.
Variants include FLORT D, J, K, M, N, and O, as well as the ECO-BB3 (now
classified as FLORT series O).

### L1 products: scale and offset

All three L1 products (CHLAFLO_L1, CDOMFLO_L1, FLUBSCT_L1) are computed by
the same calibration expression: subtract the instrument's dark count (signal
output measured in clean water with the detector blocked) and multiply by an
instrument-specific scale factor. The dark count and scale factor are
determined from factory calibration and updated periodically.

### L2 product: total optical backscatter

The total optical backscatter coefficient (FLUBSCT_L2) requires the volume
scattering function (FLUBSCT_L1) plus in-situ temperature and salinity from a
co-located CTD. The calculation follows Zhang et al. (2009) to compute the
pure-seawater contribution, which is subtracted from the measured signal to
isolate the particulate component. The particulate backscatter is then
integrated over all backward angles using the chi (χ) factor.

#### Instrument-specific parameters for FLUBSCT_L2

The effective (centroid) backscatter angle θ and the chi factor χ depend on
the instrument geometry and are **not** interchangeable between instrument
types:

| Instrument group | θ (degrees) | χ factor |
|---|---|---|
| ECO 3-channel instruments (FLORT D,J,K,M,N,O; FLORD D; ECO-BB3) | 124 | 1.076 |
| ECO 2-channel instruments (FLORD G,L,M; FLNTU series A) | 140 | 1.096 |

Note: the 3-channel centroid angle is 124°, not 117° as stated in older WET
Labs documentation. The ECO-BB3 was initially misclassified in OOI as an
OPTAA series M instrument; it is correctly classified as a FLORT series O
instrument.

All FLORD and FLORT instruments measure backscatter at a nominal wavelength of
700 nm. ECO-BB3 instruments may measure at multiple visible wavelengths. The
wavelength dependence of χ is thought to be very weak, and the values above
apply across the visible spectrum (M. Twardowski, personal communication).

#### Chi factor: definition and common misuse

The chi factor relates the volume scattering function measured at angle θ to
the total particulate backscatter coefficient (the integral of the volume
scattering function over all backward angles). It is a **sensor-geometry
scaling factor**, not an angular resolution as it has been labelled in some
OOI documentation.

#### Meaning of "total" in this context

The word "total" appears with different meanings in the backscatter literature
and in OOI documentation:

1. **Seawater + particulate** scattering (the measured signal and the final L2 product)
2. **Forward + backward** scattering (the integral over all angles)
3. **Backscatter integrated over all backward wavelengths**

In this module, "total" in `flo_bback_total` refers to meaning (1): the
combined seawater and particulate contribution.

## Core functions

::: ion_functions.data.flo_functions.flo_bback_total

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `flo_bback_total` | 2013-07-16 | Christopher Wingard | Initial implementation |
| `flo_bback_total` | 2014-04-23 | Christopher Wingard | Revisions to address integration issues and meet intent of DPS |
| `flo_bback_total` | 2015-10-26 | Russell Desiderio | Removed default values from argument list; revised documentation; added Notes |

#### Additional Notes

Instrument-specific values for the backscatter angle (θ) and chi factor (χ)
are given in the [Background](#background) section of this page, along with
instrument classification notes and the definition of "total" as used here.

#### References

OOI (2012). Data Product Specification for Optical Backscatter.
Document Control Number 1341-00540.
[https://oceanobservatories.org/wp-content/uploads/2015/10/1341-00540_Data_Product_SPEC_FLUBSCT_OOI.pdf](https://oceanobservatories.org/wp-content/uploads/2015/10/1341-00540_Data_Product_SPEC_FLUBSCT_OOI.pdf)

Sullivan, J.M., M.S. Twardowski, J.R.V. Zaneveld, and C.C. Moore (2013).
Measuring optical backscattering in water. Chapter 6 in Light Scattering
Reviews 7, pp 189–224.

---

::: ion_functions.data.flo_functions.flo_scat_seawater

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `flo_scat_seawater` | 2014-04-24 | Christopher Wingard | Initial implementation |

#### References

OOI (2012). Data Product Specification for Optical Backscatter.
Document Control Number 1341-00540.
[https://oceanobservatories.org/wp-content/uploads/2015/10/1341-00540_Data_Product_SPEC_FLUBSCT_OOI.pdf](https://oceanobservatories.org/wp-content/uploads/2015/10/1341-00540_Data_Product_SPEC_FLUBSCT_OOI.pdf)

---

::: ion_functions.data.flo_functions.flo_zhang_scatter_coeffs

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `flo_zhang_scatter_coeffs` | 2013-07-15 | Christopher Wingard | Initial implementation |
| `flo_zhang_scatter_coeffs` | 2023-08-15 | Samuel Dahlberg | Removed numexpr |

#### Additional Notes

Implements the Zhang et al. (2009) algorithm. This code is derived from MATLAB
code developed and made available by Dr. Xiaodong Zhang, University of North
Dakota. Sub-calculations are delegated to `flo_refractive_index` (Ciddor, 1996;
Quan and Fry, 1994), `flo_isotherm_compress` (Lepple and Millero, 1971; Millero,
1980), and `flo_density_seawater` (UNESCO, 1981). The water activity derivative
(dlnawds) uses polynomial coefficients fitted to Millero and Leung (1976). The
density derivative of the refractive index uses the PMH molecular scattering
theory model.

#### References

OOI (2012). Data Product Specification for Optical Backscatter.
Document Control Number 1341-00540.
[https://oceanobservatories.org/wp-content/uploads/2015/10/1341-00540_Data_Product_SPEC_FLUBSCT_OOI.pdf](https://oceanobservatories.org/wp-content/uploads/2015/10/1341-00540_Data_Product_SPEC_FLUBSCT_OOI.pdf)

Zhang, X., L. Hu, and M.-X. He (2009). Scattering by pure seawater: Effect of
salinity. Optics Express, 17(7), 5698–5710.

Millero, F.J., and Leung (1976). American Journal of Science, 276, 1035–1077.

---

::: ion_functions.data.flo_functions.flo_refractive_index

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `flo_refractive_index` | 2014-02-21 | Christopher Wingard | Initial implementation |
| `flo_refractive_index` | 2023-08-15 | Samuel Dahlberg | Removed numexpr |

#### Additional Notes

Refractive index of air from Ciddor (1996). Refractive index of seawater from
the empirical equations of Quan and Fry (1994). The returned `nsw` is the
absolute seawater refractive index (seawater relative to vacuum); `dnds` is
the partial derivative of `nsw` with respect to salinity.

#### References

Ciddor, P.E. (1996). Refractive index of air: new equations for the visible
and near infrared. Applied Optics, 35(9), 1566–1573.

Quan, X., and E.S. Fry (1994). Empirical equation for the index of refraction
of seawater. Applied Optics.

---

::: ion_functions.data.flo_functions.flo_isotherm_compress

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `flo_isotherm_compress` | 2014-02-21 | Christopher Wingard | Initial implementation |
| `flo_isotherm_compress` | 2023-08-15 | Samuel Dahlberg | Removed numexpr |

#### Additional Notes

Pure water secant bulk modulus from Millero (1980). The estimated error in the
computed isothermal compressibility is ±0.004×10⁻⁶ bar⁻¹ (Lepple and Millero,
1971, pp. 10–11).

#### References

Lepple, F.K., and F.J. Millero (1971). Deep-Sea Research, pp. 10–11.

Millero, F.J. (1980). Deep-Sea Research.

---

::: ion_functions.data.flo_functions.flo_density_seawater

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `flo_density_seawater` | 2014-02-21 | Christopher Wingard | Initial implementation |
| `flo_density_seawater` | 2023-08-15 | Samuel Dahlberg | Removed numexpr |

#### Additional Notes

Uses the UNESCO (1981) equation of state for seawater. This is the legacy
formulation specified by Zhang et al. (2009) and is distinct from the
GSW/TEOS-10 library used for `ctd_pracsal` and `ctd_density`.

#### References

UNESCO (1981). Background papers and supporting data on the International
Equation of State of Seawater 1980. UNESCO Technical Papers in Marine
Science, No. 38.

---

::: ion_functions.data.flo_functions.flo_scale_and_offset

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `flo_scale_and_offset` | 2014-01-30 | Craig Risien | Initial implementation |
| `flo_scale_and_offset` | 2023-08-15 | Samuel Dahlberg | Removed numexpr |

#### References

None

---

::: ion_functions.data.flo_functions.flo_chla

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `flo_chla` | 2014-01-30 | Craig Risien | Initial implementation |

#### Additional Notes

Chlorophyll-a is the primary photosynthetic pigment in phytoplankton. It
absorbs photons in the visible spectrum (400–700 nm) and fluoresces red light
(∼685 nm). Fluorometric concentration is a proxy for phytoplankton biomass
and, by extension, primary productivity in the water column.

#### References

OOI (2012). Data Product Specification for Fluorometric Chlorophyll-a
Concentration. Document Control Number 1341-00530.
[https://oceanobservatories.org/wp-content/uploads/2014/04/1341-00530_Data_Product_SPEC_CHLAFLO_OOI.pdf](https://oceanobservatories.org/wp-content/uploads/2014/04/1341-00530_Data_Product_SPEC_CHLAFLO_OOI.pdf)

---

::: ion_functions.data.flo_functions.flo_cdom

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `flo_cdom` | 2014-01-30 | Craig Risien | Initial implementation |

#### Additional Notes

CDOM (colored dissolved organic matter) is a pool of refractory organic
compounds — primarily tannins (polyphenols) and lignins (phenolic polymers)
from decaying plant material and animal decomposition. It absorbs ultraviolet
light and fluoresces visible blue light, imparting the tea-like color observed
in some coastal and estuarine water masses. CDOM can significantly reduce
light availability for primary production and affects satellite ocean-color
retrievals. It is a natural water-mass tracer and is used in applications
ranging from wastewater monitoring to coastal ecosystem studies.

#### References

OOI (2012). Data Product Specification for Fluorometric CDOM Concentration.
Document Control Number 1341-00550.
[https://oceanobservatories.org/wp-content/uploads/2014/04/1341-00550_Data_Product_SPEC_CDOMFLO_OOI.pdf](https://oceanobservatories.org/wp-content/uploads/2014/04/1341-00550_Data_Product_SPEC_CDOMFLO_OOI.pdf)

---

::: ion_functions.data.flo_functions.flo_beta

#### History

| Function | Date | Author | Change |
|---|---|---|---|
| `flo_beta` | 2014-01-30 | Craig Risien | Initial implementation |
| `flo_beta` | 2015-10-23 | Russell Desiderio | Revised documentation |

#### Additional Notes

Most FLORD, FLORT, and FLNTU instruments measure backscatter at a nominal
wavelength of 700 nm. The exception is the FLORT series O (ECO-BB3), which
has three backscatter channels at different visible wavelengths. To convert
FLUBSCT_L1 to the total backscatter coefficient (FLUBSCT_L2), pass the output
of this function to `flo_bback_total`. See the [Background](#background)
section for instrument-specific θ and χ values.

#### References

OOI (2012). Data Product Specification for Optical Backscatter.
Document Control Number 1341-00540.
[https://oceanobservatories.org/wp-content/uploads/2015/10/1341-00540_Data_Product_SPEC_FLUBSCT_OOI.pdf](https://oceanobservatories.org/wp-content/uploads/2015/10/1341-00540_Data_Product_SPEC_FLUBSCT_OOI.pdf)
