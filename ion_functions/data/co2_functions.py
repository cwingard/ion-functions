#!/usr/bin/env python
"""
Functions for the OOI CO2 instrument families: SAMI-II PCO2W (pCO2 in seawater)
and PCO2A (pCO2 in air and surface seawater). Produces CO2THRM_L1, PCO2WAT_L1,
PCO2ATM_L1, PCO2SSW_L1, and CO2FLUX_L2 data products.
"""

import numpy as np
from ion_functions.utils import fill_value


# wrapper functions to extract parameters from SAMI-II CO2 instruments (PCO2W)
# and process these extracted parameters to calculate pCO2
def pco2_abs434_ratio(light):
    """
    OOI single-output wrapper for CO2ABS1_L0. Returns absorbance ratio at 434 nm [unitless].

    See Also
    --------
    pco2_calc_pco2 : Core pCO2 algorithm; accepts the full light array directly.
    """
    light = np.atleast_2d(light)
    a434ratio = light[:, 6]
    return a434ratio


def pco2_abs620_ratio(light):
    """
    OOI single-output wrapper for CO2ABS2_L0. Returns absorbance ratio at 620 nm [unitless].

    See Also
    --------
    pco2_calc_pco2 : Core pCO2 algorithm; accepts the full light array directly.
    """
    light = np.atleast_2d(light)
    a620ratio = light[:, 7]
    return a620ratio


def pco2_blank(raw_blank):
    """
    Normalize raw SAMI-II blank counts to a dimensionless absorbance blank.

    Parameters
    ----------
    raw_blank : array_like
        Raw optical absorbance blank at 434 or 620 nm [counts].

    Returns
    -------
    blank : ndarray
        Normalized optical absorbance blank at 434 or 620 nm [unitless].

    Notes
    -----
    Divides raw counts by 16384 (2^14), the full-scale ADC value used by the
    SAMI-II for the absorbance ratio channels. The result is a [0, 1] scale
    factor used in the blank correction applied in `pco2_calc_pco2`.
    """
    blank = raw_blank / 16384.
    return blank


def pco2_thermistor(traw, sami_bits=12):
    """
    Convert raw SAMI-II thermistor counts to temperature for CO2THRM_L1.

    Parameters
    ----------
    traw : array_like
        Raw thermistor temperature (CO2THRM_L0) [counts].
    sami_bits : int or array_like, optional
        ADC bit depth of the SAMI hardware: 12 for original hardware,
        14 for newer hardware variants. Default is 12.

    Returns
    -------
    therm : ndarray
        Thermistor temperature (CO2THRM_L1) [degC].

    Notes
    -----
    Applies a Steinhart-Hart thermistor equation:
    1/T = A + B*ln(R) + C*(ln(R))^3, where T is absolute temperature [K] and
    R is the thermistor resistance computed from the raw count relative to a
    17.4 kOhm reference resistor. The 12-bit SAMI uses a 4096-count full scale;
    the 14-bit uses 16384 counts.
    """
    # reset inputs to arrays
    traw = np.atleast_1d(traw)
    sami_bits = np.atleast_1d(sami_bits)

    # convert raw thermistor readings from counts to degrees Centigrade
    # conversion depends on whether the SAMI is older 12 bit or newer 14 bit hardware
    if sami_bits[0] == 14:
        rt = np.log((traw / (16384.0 - traw)) * 17400.0)
    else:
        rt = np.log((traw / (4096. - traw)) * 17400.)
    inv_t = 0.0010183 + 0.000241 * rt + 0.00000015 * rt**3
    therm = (1. / inv_t) - 273.15
    return therm


def pco2_battery(braw, sami_bits):
    """
    Convert raw SAMI-II battery counts to battery voltage.

    Parameters
    ----------
    braw : array_like
        Raw battery voltage [counts].
    sami_bits : int or array_like
        ADC bit depth of the SAMI hardware: 12 for original hardware,
        14 for newer hardware variants.

    Returns
    -------
    volts : ndarray
        Battery voltage [V].
    """
    # reset inputs to arrays
    braw = np.atleast_1d(braw)
    sami_bits = np.atleast_1d(sami_bits)

    # convert raw battery readings from counts to Volts
    if sami_bits[0] == 14:
        volts = braw * 3. / 4000.
    else:
        volts = braw * 15. / 4096.
    return volts


def pco2_pco2wat(mtype, light, therm, ea434, eb434, ea620, eb620,
                 calt, cala, calb, calc, a434blank, a620blank):
    """
    OOI single-output wrapper for PCO2WAT_L1. Returns pCO2 in seawater [uatm].

    See Also
    --------
    pco2_calc_pco2 : Core algorithm; use directly for full pCO2 calculation.
    """
    # reset inputs to arrays
    # measurements
    mtype = np.atleast_1d(mtype)
    light = np.atleast_2d(light)
    therm = np.atleast_1d(therm)
    # calibration coefficients
    ea434 = np.atleast_1d(ea434)
    eb434 = np.atleast_1d(eb434)
    ea620 = np.atleast_1d(ea620)
    eb620 = np.atleast_1d(eb620)
    calt = np.atleast_1d(calt)
    cala = np.atleast_1d(cala)
    calb = np.atleast_1d(calb)
    calc = np.atleast_1d(calc)
    # blank measurements
    a434blank = np.atleast_1d(a434blank)
    a620blank = np.atleast_1d(a620blank)

    # calculate the pco2 value
    pco2 = pco2_calc_pco2(light, therm, ea434, eb434, ea620, eb620,
                          calt, cala, calb, calc, a434blank, a620blank)

    # reset dark measurements to the fill value
    m = np.where(mtype == 5)[0]
    pco2[m] = fill_value

    return pco2


# L1a PCO2WAT calculation
def pco2_calc_pco2(light, therm, ea434, eb434, ea620, eb620,
                   calt, cala, calb, calc, a434blank, a620blank):
    """
    Compute partial pressure of CO2 in seawater (PCO2WAT_L1) from SAMI-II light measurements.

    Parameters
    ----------
    light : array_like, shape (N, 14)
        Array of raw light measurements from the SAMI-II PCO2W instrument.
        Column 6 contains the 434 nm absorbance ratio (CO2ABS1_L0);
        column 7 contains the 620 nm absorbance ratio (CO2ABS2_L0) [counts].
    therm : array_like
        Thermistor temperature (CO2THRM_L0) [counts].
    ea434 : array_like
        Reagent-specific molar absorptivity coefficient at 434 nm. Not used in
        the current (2018) vendor formulation; retained for API compatibility.
    eb434 : array_like
        Reagent-specific molar absorptivity coefficient at 434 nm (indicator
        base form). Not used in the current formulation; retained for API
        compatibility.
    ea620 : array_like
        Reagent-specific molar absorptivity coefficient at 620 nm. Not used in
        the current formulation; retained for API compatibility.
    eb620 : array_like
        Reagent-specific molar absorptivity coefficient at 620 nm (indicator
        base form). Not used in the current formulation; retained for API
        compatibility.
    calt : array_like
        Instrument-specific calibration coefficient for temperature correction [degC].
    cala : array_like
        Instrument-specific calibration coefficient a for the pCO2 equation.
    calb : array_like
        Instrument-specific calibration coefficient b for the pCO2 equation.
    calc : array_like
        Instrument-specific calibration coefficient c for the pCO2 equation.
    a434blank : array_like
        Blank measurement at 434 nm from the most recent blank record [counts].
    a620blank : array_like
        Blank measurement at 620 nm from the most recent blank record [counts].

    Returns
    -------
    pco2 : ndarray
        Partial pressure of CO2 in seawater (PCO2WAT_L1) [uatm].

    Notes
    -----
    Uses the 2018 Sunburst Sensors vendor formulation with fixed spectral ratio
    constants e1=0.0043, e2=2.136, e3=0.2105 in place of the earlier reagent-
    coefficient expressions. The molar absorptivity arguments (ea434, eb434,
    ea620, eb620) are accepted but not used in the current algorithm. Blank
    measurement records, identified by equal blank-corrected absorbance ratios
    at both wavelengths, are spoofed to 0.99999 to avoid log-domain errors and
    subsequently reset to the fill value.
    """
    # set constants -- original vendor formulation, reset below
    # ea434 = ea434 - 29.3 * calt
    # eb620 = eb620 - 70.6 * calt
    # e1 = ea620 / ea434
    # e2 = eb620 / ea434
    # e3 = eb434 / ea434

    # set the e constants, values provided by Sunburst
    e1 = 0.0043
    e2 = 2.136
    e3 = 0.2105

    # Extract variables from light array
    ratio434 = light[:, 6]     # 434nm Ratio
    ratio620 = light[:, 7]     # 620nm Ratio

    # correct the absorbance ratios using the blanks
    ar434 = (ratio434 / a434blank)
    ar4620 = (ratio620 / a620blank)

    # map out blank measurements and spoof the ratios to avoid throwing an error
    m = np.where(ar434 == ar4620)[0]
    ar434[m] = 0.99999
    ar4620[m] = 0.99999

    # Calculate the final absorbance ratio
    a434 = -1 * np.log10(ar434)  # 434 absorbance
    a620 = -1 * np.log10(ar4620)  # 620 absorbance
    ratio = a620 / a434          # Absorbance ratio

    # calculate pCO2
    v1 = ratio - e1
    v2 = e2 - e3 * ratio
    rco21 = -1 * np.log10(v1 / v2)
    rco22 = (therm - calt) * 0.007 + rco21
    t_coeff = 0.0075778 + 0.0012389 * rco22 - 0.00048757 * rco22**2
    t_cor_rco2 = rco21 + t_coeff * (therm - calt)
    pco2 = 10.**((-1. * calb + (calb**2 - (4. * cala * (calc - t_cor_rco2)))**0.5) / (2. * cala))
    pco2[m] = fill_value  # reset the blanks captured earlier to a fill value

    return np.real(pco2)


def pco2_ppressure(xco2, p, std=1013.25):
    """
    Compute partial pressure of CO2 in air or seawater (PCO2ATM_L1 or PCO2SSW_L1).

    Parameters
    ----------
    xco2 : array_like
        CO2 mole fraction in air or surface seawater (XCO2ATM_L0 or
        XCO2SSW_L0) [ppm].
    p : array_like
        Gas stream pressure (PRESAIR_L0) [mbar].
    std : float, optional
        Standard atmospheric pressure [mbar]. Default is 1013.25 mbar.

    Returns
    -------
    ppres : ndarray
        Partial pressure of CO2 in air or surface seawater
        (PCO2ATM_L1 or PCO2SSW_L1) [uatm].
    """
    ppres = xco2 * p / std
    return ppres


def pco2_co2flux(pco2w, pco2a, u10, t, s):
    """
    Compute sea-to-air CO2 flux (CO2FLUX_L2) from PCO2A and METBK data.

    Parameters
    ----------
    pco2w : array_like
        Partial pressure of CO2 in seawater (PCO2SSW_L1) [uatm].
    pco2a : array_like
        Partial pressure of CO2 in air (PCO2ATM_L1) [uatm].
    u10 : array_like
        Wind speed at 10 m height (WIND10M_L2) [m s^-1].
    t : array_like
        Sea surface temperature (TEMPSRF_L1) [degC].
    s : array_like
        Sea surface salinity (SALSURF_L2) [psu].

    Returns
    -------
    flux : ndarray
        Estimated sea-to-air CO2 flux (CO2FLUX_L2) [mol m^-2 s^-1].
        Positive values indicate outgassing; negative values indicate uptake.

    Notes
    -----
    The Schmidt number Sc is computed from the Wanninkhof (1992) polynomial
    (Table A1). The gas transfer velocity k uses the Sweeney et al. (2007)
    parameterization (k = 0.27 * u10^2 * sqrt(660/Sc)), converted from
    cm h^-1 to m s^-1. CO2 solubility K0 follows the volumetric formulation
    of Weiss (1974, Eqn. 12 and Table I) with units of mol atm^-1 m^-3.
    """
    # convert micro-atm to atm
    pco2a = pco2a / 1.0e6
    pco2w = pco2w / 1.0e6

    # Compute Schmidt number (after Wanninkhof, 1992, Table A1)
    Sc = 2073.1 - (125.62 * t) + (3.6276 * t**2) - (0.043219 * t**3)

    # Compute gas transfer velocity (after Sweeney et al. 2007, Fig. 3 and Table 1)
    k = 0.27 * u10**2 * np.sqrt(660.0 / Sc)

    # convert cm h-1 to m s-1
    k = k / (100.0 * 3600.0)

    # Compute the absolute temperature
    T = t + 273.15

    # Compute solubility (after Weiss 1974, Eqn. 12 and Table I).
    # Note that there are two versions, one for units per volume and
    # one per mass. Here, the volume version is used.
    # mol atm-1 m-3
    T100 = T / 100
    K0 = 1000 * np.exp(-58.0931 + (90.5069 * (100/T)) + (22.2940 * np.log(T100)) +
                       s * (0.027766 - (0.025888 * T100) + (0.0050578 * T100**2)))

    # mol atm-1 kg-1
    #K0 = np.exp(-60.2409 + (93.4517 * (100/T)) + (23.3585 * np.log(T100)) +
    #            s * (0.023517 - (0.023656 * T100) + (0.0047036 * T100**2)))

    # Compute flux (after Wanninkhof, 1992, eqn. A2)
    flux = k * K0 * (pco2w - pco2a)
    return flux
