#!/usr/bin/env python
"""
Functions for processing data from the OOI Fluorometer instrument family.

Covers WET Labs ECO fluorometer instruments: FLORD (two-channel) and FLORT
(three-channel), including FLNTU variants. Produces L1 chlorophyll-a
(CHLAFLO), CDOM (CDOMFLO), and volume scattering (FLUBSCT) data products,
and the L2 total optical backscatter coefficient (FLUBSCT_L2).
"""
import numpy as np


def flo_bback_total(beta, degC, psu, theta, wlngth, xfactor):
    """
    Compute total optical backscatter coefficient (FLUBSCT_L2).

    Calculates the total (seawater + particulate) optical backscatter
    coefficient at the measurement wavelength from the volume scattering
    function and in-situ CTD data.

    Parameters
    ----------
    beta : array_like
        Volume scattering function at angle theta and wavelength wlngth
        (FLUBSCT_L1) [m-1 sr-1].
    degC : array_like
        In-situ water temperature from co-located CTD [deg_C].
    psu : array_like
        In-situ salinity from co-located CTD [psu].
    theta : float
        Effective (centroid) backscatter angle determined by sensor geometry
        [degrees]. See Notes.
    wlngth : float
        Optical backscatter measurement wavelength [nm].
    xfactor : float
        Chi factor scaling particulate scattering at theta to total
        particulate backscatter coefficient [unitless]. See Notes.

    Returns
    -------
    bback : ndarray
        Total optical backscatter coefficient (FLUBSCT_L2) [m-1].

    Notes
    -----
    Instrument-specific values for theta and xfactor:

    - **ECO 3-channel instruments** (FLORT D,J,K,M,N,O; FLORD D):
      theta = 124 degrees, xfactor = 1.076
    - **ECO 2-channel instruments** (FLORD G,L,M; FLNTU series A):
      theta = 140 degrees, xfactor = 1.096

    Most FLORD/FLORT instruments measure backscatter at 700 nm. The chi
    factor is the scaling factor relating particulate scattering at angle
    theta to the total particulate backscatter integrated over all backward
    angles. See the [Background](flo_functions.md#background) and
    [References](flo_functions.md#references) sections for full details.
    """
    # calculate:
    #    betasw, the theoretical value of the volume scattering function for seawater only
    #        at the measurement angle theta and wavelength wlngth [m-1 sr-1], and,
    #    bsw, the theoretical value for the total (in this case meaning forward + backward)
    #         scattering coefficient for seawater (also with no particulate contribution)
    #         at wavelength wlngth [m-1].
    # Values below are computed using provided code from Zhang et al 2009.
    betasw, bsw = flo_zhang_scatter_coeffs(degC, psu, theta, wlngth)

    # calculate the volume scattering at angle theta of particles only, betap.
    #     beta = scattering measured at angle theta for seawater + particulates
    #     betasw = theoretical seawater only value calculated at angle theta
    betap = beta - betasw

    # calculate the particulate backscatter coefficient bbackp [m-1] which is effectively
    # the particulate scattering function integrated over all backwards angles. The factor
    # of 2*pi arises from the integration over the (implicit) polar angle variable.
    pi = np.pi
    bbackp = xfactor * 2.0 * pi * betap

    # calculate the backscatter coefficient due to seawater from the total (forward + backward)
    # scattering coefficient bsw. because the effective scattering centers in pure seawater are
    # much smaller than the wavelength, the shape of the scattering function is symmetrical in
    # the forward and backward directions.
    bbsw = bsw / 2

    # calculate the total (particulates + seawater) backscatter coefficient.
    bback = bbackp + bbsw

    return bback


def flo_scat_seawater(degC, psu, theta, wlngth, delta=0.039):
    """
    Compute the total scattering coefficient of pure seawater.

    Convenience wrapper returning only the total scattering coefficient
    from `flo_zhang_scatter_coeffs`. Used as an intermediate step in the
    FLUBSCT_L2 calculation.

    Parameters
    ----------
    degC : array_like
        In-situ water temperature from co-located CTD [deg_C].
    psu : array_like
        In-situ salinity from co-located CTD [psu].
    theta : float
        Optical backscatter angle [degrees]. See Notes in `flo_bback_total`.
    wlngth : float
        Optical backscatter measurement wavelength [nm].
    delta : float, optional
        Depolarization ratio [unitless]. Default is 0.039.

    Returns
    -------
    bsw : ndarray
        Total scattering coefficient of pure seawater [m-1].

    """
    _, bsw = flo_zhang_scatter_coeffs(degC, psu, theta, wlngth, delta)
    return bsw


def flo_zhang_scatter_coeffs(degC, psu, theta, wlngth, delta=0.039):
    """
    Compute seawater volume scattering and total scattering coefficients.

    Calculates the volume scattering function of pure seawater at a given
    angle and wavelength, and the total seawater scattering coefficient,
    using the Zhang et al. (2009) method as specified in the FLUBSCT DPS.

    Parameters
    ----------
    degC : array_like
        In-situ water temperature from co-located CTD [deg_C].
    psu : array_like
        In-situ salinity from co-located CTD [psu].
    theta : float
        Optical backscatter angle [degrees]. See Notes in `flo_bback_total`.
    wlngth : float
        Optical backscatter measurement wavelength [nm].
    delta : float, optional
        Depolarization ratio [unitless]. Default is 0.039.

    Returns
    -------
    betasw : ndarray
        Volume scattering function of pure seawater at angle theta and
        wavelength wlngth [m-1 sr-1].
    bsw : ndarray
        Total scattering coefficient of pure seawater [m-1].

    """
    # values of the constants
    Na = 6.0221417930e23    # Avogadro's constant
    Kbz = 1.3806503e-23     # Boltzmann constant
    degK = degC + 273.15    # Absolute temperature
    M0 = 0.018              # Molecular weight of water in kg/mol
    pi = np.pi

    # convert the scattering angle from degrees to radians
    rad = np.radians(theta)

    # calculate the absolute refractive index of seawater and the partial
    # derivative of seawater refractive index with regards to salinity.
    nsw, dnds = flo_refractive_index(wlngth, degC, psu)

    # isothermal compressibility is from Lepple & Millero (1971,Deep
    # Sea-Research), pages 10-11 The error ~ +/-0.004e-6 bar^-1
    icomp = flo_isotherm_compress(degC, psu)

    # density of seawater from UNESCO 38 (1981).
    rho = flo_density_seawater(degC, psu)

    # water activity data of seawater is from Millero and Leung (1976, American
    # Journal of Science, 276, 1035-1077). Table 19 was reproduced using
    # Eq.(14,22,23,88,107) that were fitted to polynominal equation. dlnawds is
    # a partial derivative of the natural logarithm of water activity with
    # regards to salinity.
    dlnawds = (-5.58651e-4 + 2.40452e-7 * degC - 3.12165e-9 * degC**2 + 2.40808e-11 * degC**3) + \
            1.5 * (1.79613e-5 - 9.9422e-8 * degC + 2.08919e-9 * degC**2 - 1.39872e-11 * degC**3) * \
            psu**0.5 + 2 * (-2.31065e-6 - 1.37674e-9 * degC - 1.93316e-11 * degC**2) * psu

    # density derivative of refractive index from PMH model
    dfri = (nsw**2 - 1.0) * (1.0 + 2.0/3.0 * (nsw**2 + 2.0) * (nsw/3.0 - 1.0/3.0 / nsw)**2)

    # volume scattering at 90 degrees due to the density fluctuation
    beta_df = pi**2 / 2.0 * (wlngth*1e-9)**-4 * Kbz * degK * icomp * dfri**2 * (6.0 + 6.0 * delta) / (6.0 - 7.0 * delta)

    # volume scattering at 90 degree due to the concentration fluctuation
    flu_con = psu * M0 * dnds**2 / rho / -dlnawds / Na
    beta_cf = 2.0 * pi**2 * (wlngth * 1e-9)**-4 * nsw**2 * flu_con * (6.0 + 6.0 * delta) / (6.0 - 7.0 * delta)

    # total volume scattering at 90 degree
    beta90sw = beta_df + beta_cf

    # total scattering coefficient of seawater (m-1)
    bsw = 8.0 * pi / 3.0 * beta90sw * ((2.0 + delta) / (1.0 + delta))

    # total volume scattering coefficient of seawater (m-1 sr-1)
    betasw = beta90sw * (1.0 + ((1.0 - delta) / (1.0 + delta)) * np.cos(rad)**2)

    return betasw, bsw


def flo_refractive_index(wlngth, degC, psu):
    """
    Compute the absolute refractive index of seawater.

    Helper function for `flo_zhang_scatter_coeffs`. Uses the Quan and Fry
    (1994) model for seawater refractive index and the Ciddor (1996) model
    for the refractive index of air.

    Parameters
    ----------
    wlngth : float
        Backscatter measurement wavelength [nm].
    degC : array_like
        In-situ water temperature [deg_C].
    psu : array_like
        In-situ practical salinity [psu].

    Returns
    -------
    nsw : ndarray
        Absolute refractive index of seawater [unitless].
    dnds : ndarray
        Partial derivative of seawater refractive index with respect to
        salinity [unitless].
    """
    # refractive index of air is from Ciddor (1996, Applied Optics).
    n_air = 1.0 + (5792105.0 / (238.0185 - 1 / (wlngth/1e3)**2) + 167917.0 / (57.362 - 1 / (wlngth/1e3)**2)) / 1e8

    # refractive index of seawater is from Quan and Fry (1994, Applied Optics)
    n0 = 1.31405
    n1 = 1.779e-4
    n2 = -1.05e-6
    n3 = 1.6e-8
    n4 = -2.02e-6
    n5 = 15.868
    n6 = 0.01155
    n7 = -0.00423
    n8 = -4382.0
    n9 = 1.1455e6
    nsw = n0 + (n1 + n2 * degC + n3 * degC**2) * psu + n4 * degC**2 + (n5 + n6 * psu + n7 * degC) / \
          wlngth + n8 / wlngth**2 + n9 / wlngth**3

    # pure seawater
    nsw = nsw * n_air
    dnds = (n1 + n2 * degC + n3 * degC**2 + n6 / wlngth) * n_air

    return nsw, dnds


def flo_isotherm_compress(degC, psu):
    """
    Compute the isothermal compressibility of seawater.

    Helper function for `flo_zhang_scatter_coeffs`. Uses the Millero (1980)
    secant bulk modulus formulation.

    Parameters
    ----------
    degC : array_like
        In-situ water temperature [deg_C].
    psu : array_like
        In-situ practical salinity [psu].

    Returns
    -------
    iso_comp : ndarray
        Seawater isothermal compressibility [Pa-1].
    """
    # pure water secant bulk Millero (1980, Deep-sea Research)
    kw = 19652.21 + 148.4206 * degC - 2.327105 * degC**2 + 1.360477e-2 * degC**3 - 5.155288e-5 * degC**4

    # seawater secant bulk
    a0 = 54.6746 - 0.603459 * degC + 1.09987e-2 * degC**2 - 6.167e-5 * degC**3
    b0 = 7.944e-2 + 1.6483e-2 * degC - 5.3009e-4 * degC**2
    ks = kw + a0 * psu + b0 * psu**1.5

    # calculate seawater isothermal compressibility from the secant bulk
    iso_comp = 1 / ks * 1e-5  # unit is Pa

    return iso_comp


def flo_density_seawater(degC, psu):
    """
    Compute the density of seawater.

    Helper function for `flo_zhang_scatter_coeffs`. Uses the UNESCO (1981)
    equation of state for seawater.

    Parameters
    ----------
    degC : array_like
        In-situ water temperature [deg_C].
    psu : array_like
        In-situ practical salinity [psu].

    Returns
    -------
    rho_sw : ndarray
        Density of seawater [kg m-3].
    """
    # density of water and seawater,unit is Kg/m^3, from UNESCO, 38, 1981
    a0 = 8.24493e-1
    a1 = -4.0899e-3
    a2 = 7.6438e-5
    a3 = -8.2467e-7
    a4 = 5.3875e-9
    a5 = -5.72466e-3
    a6 = 1.0227e-4
    a7 = -1.6546e-6
    a8 = 4.8314e-4
    b0 = 999.842594
    b1 = 6.793952e-2
    b2 = -9.09529e-3
    b3 = 1.001685e-4
    b4 = -1.120083e-6
    b5 = 6.536332e-9

    # density for pure water
    rho_w = b0 + b1 * degC + b2 * degC**2 + b3 * degC**3 + b4 * degC**4 + b5 * degC**5

    # density for pure seawater
    rho_sw = rho_w + ((a0 + a1 * degC + a2 * degC**2 + a3 * degC**3 + a4 * degC**4) *
                      psu + (a5 + a6 * degC + a7 * degC**2) * psu**1.5 + a8 * psu**2)

    return rho_sw


def flo_scale_and_offset(counts_output, counts_dark, scale_factor):
    """
    Apply dark-count subtraction and scale factor to raw fluorometer counts.

    Core calibration function for CHLAFLO_L1, CDOMFLO_L1, and FLUBSCT_L1.
    Implements the linear scaling expression common to all three L1 products:

        value = (counts_output - counts_dark) * scale_factor

    `flo_chla`, `flo_cdom`, and `flo_beta` are named wrappers for this
    function that assign the result to a specific OOI data product.

    Parameters
    ----------
    counts_output : array_like
        Measured sample output in raw instrument counts (L0) [counts].
    counts_dark : float
        Dark count offset — signal output measured in clean water with the
        detector blocked. Determined from factory calibration [counts].
    scale_factor : float
        Instrument-specific linear scale factor. Determined from factory
        calibration [physical units counts-1].

    Returns
    -------
    value : ndarray
        Calibrated L1 output in instrument-specific physical units.

    See Also
    --------
    flo_chla : Wrapper for CHLAFLO_L1 [ug L-1].
    flo_cdom : Wrapper for CDOMFLO_L1 [ppb].
    flo_beta : Wrapper for FLUBSCT_L1 [m-1 sr-1].
    """
    value = (counts_output - counts_dark) * scale_factor
    return value


def flo_chla(counts_output, counts_dark, scale_factor):
    """
    OOI wrapper for CHLAFLO_L1. Returns chlorophyll-a concentration [ug L-1].

    See Also
    --------
    flo_scale_and_offset : Core implementation; use directly when the
        product identity does not need to be explicit.
    """
    chla_conc = flo_scale_and_offset(counts_output, counts_dark, scale_factor)
    return chla_conc


def flo_cdom(counts_output, counts_dark, scale_factor):
    """
    OOI wrapper for CDOMFLO_L1. Returns CDOM concentration [ppb].

    See Also
    --------
    flo_scale_and_offset : Core implementation; use directly when the
        product identity does not need to be explicit.
    """
    cdom_conc = flo_scale_and_offset(counts_output, counts_dark, scale_factor)
    return cdom_conc


def flo_beta(counts_output, counts_dark, scale_factor):
    """
    OOI wrapper for FLUBSCT_L1. Returns volume scattering function [m-1 sr-1].

    See Also
    --------
    flo_scale_and_offset : Core implementation; use directly when the
        product identity does not need to be explicit.
    flo_bback_total : Converts FLUBSCT_L1 to total backscatter (FLUBSCT_L2).
    """
    beta = flo_scale_and_offset(counts_output, counts_dark, scale_factor)
    return beta
