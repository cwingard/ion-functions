# Fluorometer

Functions for the OOI Fluorometer instrument family: WET Labs ECO FLORD
(two-channel) and FLORT (three-channel), including FLNTU variants.

## Background

OOI deployed the WET Labs ECO family of miniature fluorometers to measure three
optical properties of seawater: chlorophyll-a fluorescence (CHLAFLO), colored
dissolved organic matter (CDOM) fluorescence (CDOMFLO), and optical volume
scattering in the red wavelengths (FLUBSCT). Two instrument configurations were
used:

**FLORD** (two-channel): measures two of the three optical properties, typically
chlorophyll-a fluorescence and optical backscatter, or CDOM and optical
backscatter. Variants include FLORD D, G, L, and M, and the FLNTU series A.

**FLORT** (three-channel): measures all three optical properties simultaneously.
Variants include FLORT D, J, K, M, N, and O, as well as the ECO-BB3 (now
classified as FLORT series O).

The ECO fluorometer uses an LED excitation source with a bandpass interference
filter to reject out-of-band emission. The source illuminates a sample volume at
approximately 55–60° to the instrument face. A detector collects fluoresced or
scattered light at an angle where its acceptance cone intersects the source
beam; a second interference filter discriminates against scattered excitation
light. Raw output ranges from 0 to approximately 4096 counts (0–5 V).

### L1 products: scale and offset

All three L1 products — CHLAFLO_L1 [µg L⁻¹], CDOMFLO_L1 [ppb], and
FLUBSCT_L1 [m⁻¹ sr⁻¹] — are computed by the same linear calibration
expression implemented in `flo_scale_and_offset`:

```
value = (counts_output - counts_dark) * scale_factor
```

`flo_chla`, `flo_cdom`, and `flo_beta` are named wrappers that apply this
expression to assign the result to a specific OOI data product. All three
account for the same two calibration parameters:

**Dark count (`counts_dark`):** The signal output of the instrument in clean
water with the detector blocked. It represents the electronic baseline in the
absence of a light source contribution. The factory dark count is measured
using de-ionized water with black tape over the detector.

**Scale factor (`scale_factor`):** The instrument-specific linear multiplier
that converts corrected counts to physical units. The factory scale factor for
chlorophyll-a is derived from a monoculture of *Thalassiosira weissflogii*
phytoplankton at a concentration equivalent to 25 µg L⁻¹; for CDOM, from a
quinine sulfate dihydrate (QSDE) standard at 308 ppb; for backscatter, from
a polystyrene bead microsphere polydispersion using Mie theory to calculate
the theoretical scattering phase function and a factory ac-s meter to measure
beam attenuation. 

Both the dark count and scale factor values are provided on instrument
characterization sheets shipped with each meter.

#### Optical specifications (WET Labs ECO, typical)

| Channel | Excitation / Emission | Sensitivity | Range |
|---|---|---|---|
| Chlorophyll-a | 470 / 695 nm | 0.02 µg L⁻¹ | 0–125 µg L⁻¹ |
| CDOM | 370 / 460 nm | 0.09 ppb | 0–500 ppb |
| Backscatter | 700 nm (nominal) | instrument-specific | 0–5 m⁻¹ |

#### Known limitations of L1 products

For chlorophyll-a, fluorescence from degraded chlorophyll (phaeopigments) and
CDOM can cause overestimates of concentration, particularly in the surface
layer. Additionally, the fluorescence quantum yield of chlorophyll-a decreases
under light-saturating conditions; fluorescence will decrease even though
chlorophyll concentration is unchanged, so near-surface daytime data should
be interpreted with caution.

For CDOM, the nature of the CDOM pool probed is a function of both the
excitation and emission wavelengths used. Because the FLORT uses a single
channel for CDOM, no information on the composition of the CDOM present can
be derived.

For backscatter, along-path attenuation by seawater absorption introduces a
small bias that is generally negligible in open ocean conditions but can reach
10–15% in turbid coastal environments.

### L2 product: total optical backscatter

The total optical backscatter coefficient FLUBSCT_L2 [m⁻¹] is computed from
FLUBSCT_L1 plus in-situ temperature and salinity from a co-located CTD. The
computation follows Zhang et al. (2009) and proceeds in three steps:

**Step 1 — volume scattering of particles only:**
Subtract the theoretical pure-seawater volume scattering function β_sw(θ, λ)
from the measured total volume scattering function β(θ, λ):

```
β_p(θ, λ) = β(θ, λ) - β_sw(θ, λ)
```

β_sw is calculated by `flo_zhang_scatter_coeffs` using in-situ temperature
and salinity from the co-located CTD. The Zhang et al. (2009) algorithm
accounts for scattering by density fluctuations and concentration fluctuations
in seawater.

**Step 2 — particulate backscatter coefficient:**
Scale the particulate volume scattering function at angle θ to the particulate
backscatter coefficient integrated over all backward angles:

```
b_bp(λ) = χ · 2π · β_p(θ, λ)
```

The factor 2π arises from integration over the polar angle. χ (the chi factor)
is a dimensionless sensor-geometry scaling factor that relates the volume
scattering function at a single angle θ to the full hemispheric integral;
it is determined empirically from high-angular-resolution measurements across
diverse water types (Sullivan and Twardowski, 2009).

**Step 3 — total backscatter coefficient:**
Add the seawater backscatter coefficient b_bsw (half the total seawater
scattering coefficient, because pure-water scattering is symmetric in the
forward and backward directions):

```
b_b(λ) = b_bp(λ) + b_bsw(λ),   where b_bsw = b_sw / 2
```

#### Instrument-specific parameters for FLUBSCT_L2

The centroid backscatter angle θ and chi factor χ are geometry-specific and
are **not** interchangeable between instrument types:

| Instrument group | θ (degrees) | χ |
|---|---|---|
| ECO 3-channel (FLORT D,J,K,M,N,O; FLORD D; ECO-BB3) | 124 | 1.076 |
| ECO 2-channel (FLORD G,L,M; FLNTU series A) | 140 | 1.096 |

The 3-channel centroid angle is 124°; older WET Labs documentation incorrectly
stated 117°. The ECO-BB3 was initially misclassified in OOI as an OPTAA series
M instrument; it is correctly classified as FLORT series O.

Most FLORD and FLORT instruments measure backscatter at a nominal wavelength of
700 nm. ECO-BB3 instruments may measure at multiple visible wavelengths. The
wavelength dependence of χ is thought to be negligible across the visible
spectrum (M. Twardowski, personal communication); the values above apply at
all OOI-deployed wavelengths.

#### Meaning of "total" in this context

The word "total" appears with multiple meanings in the backscatter literature:

1. **Seawater + particulate** — the measured signal and the FLUBSCT_L2 product
2. **Forward + backward** — the integral of the volume scattering function
   over all solid angles

In this module, "total" in `flo_bback_total` refers to meaning (1): the
combined seawater and particulate contribution to the backscatter coefficient.

## Core functions

::: ion_functions.data.flo_functions.flo_bback_total

#### Additional Notes

Instrument-specific values for the backscatter angle (θ) and chi factor (χ)
are given in the [Background](#background) section of this page, along with
instrument classification notes and the definition of "total" as used here.

#### History

| Date | Author | Change |
|---|---|---|
| 2013-07-16 | Christopher Wingard | Initial implementation |
| 2014-04-23 | Christopher Wingard | Revisions to address integration issues and meet intent of DPS |
| 2015-10-26 | Russell Desiderio | Removed default values from argument list; revised documentation; added Notes |
| 2026-04-10 | Christopher Wingard | Converted to NumPy docstring format; updated documentation |

---

::: ion_functions.data.flo_functions.flo_scat_seawater

#### History

| Date | Author | Change |
|---|---|---|
| 2014-04-24 | Christopher Wingard | Initial implementation |
| 2026-04-10 | Christopher Wingard | Converted to NumPy docstring format; updated documentation |

---

::: ion_functions.data.flo_functions.flo_zhang_scatter_coeffs

#### Additional Notes

Implements the Zhang et al. (2009) algorithm. This code is derived from MATLAB
code developed and made available by Dr. Xiaodong Zhang, University of North
Dakota. Sub-calculations are delegated to `flo_refractive_index` (Ciddor, 1996;
Quan and Fry, 1994), `flo_isotherm_compress` (Lepple and Millero, 1971; Millero,
1980), and `flo_density_seawater` (UNESCO, 1981). The water activity derivative
(dlnawds) uses polynomial coefficients fitted to Millero and Leung (1976). The
density derivative of the refractive index uses the PMH molecular scattering
theory model.

#### History

| Date | Author | Change |
|---|---|---|
| 2013-07-15 | Christopher Wingard | Initial implementation |
| 2023-08-15 | Samuel Dahlberg | Removed numexpr |
| 2026-04-10 | Christopher Wingard | Converted to NumPy docstring format; updated documentation |

---

::: ion_functions.data.flo_functions.flo_refractive_index

#### Additional Notes

Refractive index of air from Ciddor (1996). Refractive index of seawater from
the empirical equations of Quan and Fry (1994). The returned `nsw` is the
absolute seawater refractive index (seawater relative to vacuum); `dnds` is
the partial derivative of `nsw` with respect to salinity.

#### History

| Date | Author | Change |
|---|---|---|
| 2014-02-21 | Christopher Wingard | Initial implementation |
| 2023-08-15 | Samuel Dahlberg | Removed numexpr |
| 2026-04-10 | Christopher Wingard | Converted to NumPy docstring format; updated documentation |

---

::: ion_functions.data.flo_functions.flo_isotherm_compress

#### Additional Notes

Pure water secant bulk modulus from Millero (1980). The estimated error in the
computed isothermal compressibility is ±0.004×10⁻⁶ bar⁻¹ (Lepple and Millero,
1971, pp. 10–11).

#### History

| Date | Author | Change |
|---|---|---|
| 2014-02-21 | Christopher Wingard | Initial implementation |
| 2023-08-15 | Samuel Dahlberg | Removed numexpr |
| 2026-04-10 | Christopher Wingard | Converted to NumPy docstring format; updated documentation |

---

::: ion_functions.data.flo_functions.flo_density_seawater

#### Additional Notes

Uses the UNESCO (1981) equation of state for seawater. This is the legacy
formulation specified by Zhang et al. (2009) and is distinct from the
GSW/TEOS-10 library used for `ctd_pracsal` and `ctd_density`.

#### History

| Date | Author | Change |
|---|---|---|
| 2014-02-21 | Christopher Wingard | Initial implementation |
| 2023-08-15 | Samuel Dahlberg | Removed numexpr |
| 2026-04-10 | Christopher Wingard | Converted to NumPy docstring format; updated documentation |

---

::: ion_functions.data.flo_functions.flo_scale_and_offset

#### History

| Date | Author | Change |
|---|---|---|
| 2014-01-30 | Craig Risien | Initial implementation |
| 2023-08-15 | Samuel Dahlberg | Removed numexpr |
| 2026-04-10 | Christopher Wingard | Converted to NumPy docstring format; updated documentation |

## OOI System Interface

The following functions are single-output wrappers for `flo_scale_and_offset`.
They exist to satisfy an internal OOI system requirement that each data product
be addressable by a named function. External users should call
`flo_scale_and_offset` directly.

::: ion_functions.data.flo_functions.flo_chla

#### Additional Notes

Chlorophyll-a is the primary photosynthetic pigment in phytoplankton. It
absorbs photons in the visible spectrum (400–700 nm) and fluoresces red light
(~685 nm). Fluorometric concentration is a proxy for phytoplankton biomass
and, by extension, primary productivity in the water column. See the
[Background](#background) section for calibration context and known limitations.

#### History

| Date | Author | Change |
|---|---|---|
| 2014-01-30 | Craig Risien | Initial implementation |
| 2026-04-10 | Christopher Wingard | Converted to NumPy docstring format; updated documentation |

---

::: ion_functions.data.flo_functions.flo_cdom

#### Additional Notes

CDOM (colored dissolved organic matter) is a pool of refractory organic
compounds — primarily tannins (polyphenols) and lignins (phenolic polymers)
from decaying plant material and animal decomposition. It absorbs ultraviolet
light and fluoresces visible blue light, imparting the tea-like color observed
in some coastal and estuarine water masses. See the [Background](#background)
section for calibration context and known limitations.

#### History

| Date | Author | Change |
|---|---|---|
| 2014-01-30 | Craig Risien | Initial implementation |
| 2026-04-10 | Christopher Wingard | Converted to NumPy docstring format; updated documentation |

---

::: ion_functions.data.flo_functions.flo_beta

#### Additional Notes

The volume scattering function β(θ, λ) is measured at the instrument's
effective (centroid) backscatter angle θ and wavelength λ. Most FLORD, FLORT,
and FLNTU instruments measure at 700 nm; the FLORT series O (ECO-BB3) has
three backscatter channels at different visible wavelengths. To convert
FLUBSCT_L1 to the total backscatter coefficient (FLUBSCT_L2), pass the output
of this function to `flo_bback_total`. See the [Background](#background)
section for instrument-specific θ and χ values.

#### History

| Date | Author | Change |
|---|---|---|
| 2014-01-30 | Craig Risien | Initial implementation |
| 2015-10-23 | Russell Desiderio | Revised documentation |
| 2026-04-10 | Christopher Wingard | Converted to NumPy docstring format; updated documentation |

## References

Ciddor, P.E. (1996). Refractive index of air: new equations for the visible
and near infrared. Applied Optics, 35(9), 1566–1573.

Lepple, F.K., and F.J. Millero (1971). Deep-Sea Research, pp. 10–11.

Millero, F.J. (1980). Deep-Sea Research.

Millero, F.J., and Leung (1976). American Journal of Science, 276, 1035–1077.

[OOI (2012). Data Product Specification for Fluorometric Chlorophyll-a
Concentration. Document Control Number 1341-00530.](https://oceanobservatories.org/wp-content/uploads/2014/04/1341-00530_Data_Product_SPEC_CHLAFLO_OOI.pdf)

[OOI (2012). Data Product Specification for Fluorometric CDOM Concentration.
Document Control Number 1341-00550.](https://oceanobservatories.org/wp-content/uploads/2014/04/1341-00550_Data_Product_SPEC_CDOMFLO_OOI.pdf)

[OOI (2012). Data Product Specification for Optical Backscatter (Red Wavelengths).
Document Control Number 1341-00540.](https://oceanobservatories.org/wp-content/uploads/2015/10/1341-00540_Data_Product_SPEC_FLUBSCT_OOI.pdf)

Quan, X., and E.S. Fry (1994). Empirical equation for the index of refraction
of seawater. Applied Optics.

Sullivan, J.M., M.S. Twardowski, J.R.V. Zaneveld, and C.C. Moore (2013).
Measuring optical backscattering in water. Chapter 6 in Light Scattering
Reviews 7, pp 189–224.

UNESCO (1981). Background papers and supporting data on the International
Equation of State of Seawater 1980. UNESCO Technical Papers in Marine
Science, No. 38.

Zhang, X., L. Hu, and M.-X. He (2009). Scattering by pure seawater: Effect of
salinity. Optics Express, 17(7), 5698–5710.
