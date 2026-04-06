#!/usr/bin/env python
"""
Functions for processing data from the OOI CTD instrument family.

Covers Sea-Bird Electronics SBE 16Plus, SBE 37IM, SBE 52MP, and glider-mounted
CTD instruments (CTDBP, CTDMO, CTDPF, CTDGV), producing L1 temperature
(TEMPWAT), pressure (PRESWAT), and conductivity (CONDWAT) data products and
L2 practical salinity (PRACSAL) and density (DENSITY) data products.
"""

# Import Numpy and the GSW library
import numpy as np
import gsw

def ctd_sbe16plus_tempwat(t0, a0, a1, a2, a3):
    """
    Compute water temperature (TEMPWAT_L1) from SBE 16Plus raw counts.

    Applies the SBE thermistor calibration equation to convert raw temperature
    counts to degrees Celsius. Used for CTDBP (all series) and CTDPF (series
    A and B) instruments.

    Parameters
    ----------
    t0 : array_like
        Raw temperature (TEMPWAT_L0) [counts].
    a0 : float
        Thermistor calibration coefficient.
    a1 : float
        Thermistor calibration coefficient.
    a2 : float
        Thermistor calibration coefficient.
    a3 : float
        Thermistor calibration coefficient.

    Returns
    -------
    t : ndarray
        Sea water temperature (TEMPWAT_L1) [deg_C].

    References
    ----------
    OOI (2012). Data Product Specification for Water Temperature.
        Document Control Number 1341-00010. [Legacy document, archived]
        https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00010_Data_Product_SPEC_TEMPWAT_OOI.pdf
    """

    mv = (t0 - 524288) / 1.6e7
    r = (mv * 2.9e9 + 1.024e8)/(2.048e4 - mv * 2.0e5)
    t = 1 / (a0 + a1 * np.log(r) + a2 * np.log(r)**2 + a3 * np.log(r)**3) - 273.15
    return t


def ctd_sbe37im_tempwat_instrument_recovered(t0, a0, a1, a2, a3):
    """
    Compute water temperature (TEMPWAT_L1) from SBE 37IM instrument-recovered counts.

    Applies the SBE thermistor calibration equation to instrument-recovered
    temperature counts. Used for CTDMO (all series) data recovered directly
    from the instrument (not telemetered or recovered_host).

    Parameters
    ----------
    t0 : array_like
        Raw temperature (TEMPWAT_L0) recovered from the instrument [counts].
    a0 : float
        Thermistor calibration coefficient.
    a1 : float
        Thermistor calibration coefficient.
    a2 : float
        Thermistor calibration coefficient.
    a3 : float
        Thermistor calibration coefficient.

    Returns
    -------
    t : ndarray
        Sea water temperature (TEMPWAT_L1) [deg_C].

    Notes
    -----
    This algorithm was not included in the TEMPWAT DPS as of June 2016.

    References
    ----------
    OOI (2012). Data Product Specification for Water Temperature.
        Document Control Number 1341-00010. [Legacy document, archived]
        https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00010_Data_Product_SPEC_TEMPWAT_OOI.pdf
    """

    t = 1 / (a0 + a1 * np.log(t0) + a2 * np.log(t0)**2 + a3 * np.log(t0)**3) - 273.15
    return t


def ctd_sbe37im_tempwat(t0):
    """
    Compute water temperature (TEMPWAT_L1) from SBE 37IM telemetered counts.

    Converts raw temperature counts to degrees Celsius for telemetered and
    recovered_host data from CTDMO instruments (all series). For
    instrument-recovered data use `ctd_sbe37im_tempwat_instrument_recovered`.

    Parameters
    ----------
    t0 : array_like
        Raw temperature (TEMPWAT_L0) [counts].

    Returns
    -------
    t : ndarray
        Sea water temperature (TEMPWAT_L1) [deg_C].

    References
    ----------
    OOI (2012). Data Product Specification for Water Temperature.
        Document Control Number 1341-00010. [Legacy document, archived]
        https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00010_Data_Product_SPEC_TEMPWAT_OOI.pdf
    """

    t = t0 / 10000.0 - 10.0
    return t


def ctd_sbe52mp_tempwat(t0):
    """
    Compute water temperature (TEMPWAT_L1) from SBE 52MP raw counts.

    Converts raw temperature counts to degrees Celsius for CTDPF instruments
    (series C, K, and L).

    Parameters
    ----------
    t0 : array_like
        Raw temperature (TEMPWAT_L0) [counts].

    Returns
    -------
    t : ndarray
        Sea water temperature (TEMPWAT_L1) [deg_C].

    References
    ----------
    OOI (2012). Data Product Specification for Water Temperature.
        Document Control Number 1341-00010. [Legacy document, archived]
        https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00010_Data_Product_SPEC_TEMPWAT_OOI.pdf
    """

    t = t0 / 10000.0 - 5.0
    return t


def ctd_sbe16plus_preswat(p0, t0, ptempa0, ptempa1, ptempa2,
                          ptca0, ptca1, ptca2, ptcb0, ptcb1, ptcb2,
                          pa0, pa1, pa2, offset=0):
    """
    Compute water pressure (PRESWAT_L1) from SBE 16Plus strain gauge counts.

    Applies the strain gauge pressure calibration equation for SBE 16Plus
    instruments. Used for most CTDBP instruments (all series except N and O)
    and CTDPF instruments (series A and B).

    Parameters
    ----------
    p0 : array_like
        Raw pressure (PRESWAT_L0) [counts].
    t0 : array_like
        Raw temperature from pressure sensor thermistor [counts].
    ptempa0 : float
        Pressure thermistor calibration coefficient.
    ptempa1 : float
        Pressure thermistor calibration coefficient.
    ptempa2 : float
        Pressure thermistor calibration coefficient.
    ptca0 : float
        Strain gauge temperature compensation coefficient.
    ptca1 : float
        Strain gauge temperature compensation coefficient.
    ptca2 : float
        Strain gauge temperature compensation coefficient.
    ptcb0 : float
        Strain gauge temperature compensation coefficient.
    ptcb1 : float
        Strain gauge temperature compensation coefficient.
    ptcb2 : float
        Strain gauge temperature compensation coefficient.
    pa0 : float
        Strain gauge pressure calibration coefficient.
    pa1 : float
        Strain gauge pressure calibration coefficient.
    pa2 : float
        Strain gauge pressure calibration coefficient.
    offset : float, optional
        Correction for Druck sensor offset error [dbar]. Default is 0.

    Returns
    -------
    p_dbar : ndarray
        Sea water pressure (PRESWAT_L1) [dbar].

    References
    ----------
    OOI (2012). Data Product Specification for Pressure (Depth).
        Document Control Number 1341-00020. [Legacy document, archived]
        https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00020_Data_Product_SPEC_PRESWAT_OOI.pdf
    """
    # compute calibration parameters
    tv = t0 / 13107.0
    t = ptempa0 + ptempa1 * tv + ptempa2 * tv**2
    x = p0 - ptca0 - ptca1 * t - ptca2 * t**2
    n = x * ptcb0 / (ptcb0 + ptcb1 * t + ptcb2 * t**2)

    # compute pressure in psi, rescale and compute in dbar and return
    p_psi = pa0 + pa1 * n + pa2 * n**2
    p_dbar = (p_psi * 0.689475729) - 10.1325
    return p_dbar + offset


def ctd_sbe16digi_preswat(p0, t0, C1, C2, C3, D1, D2, T1, T2, T3, T4, T5):
    """
    Compute water pressure (PRESWAT_L1) from SBE 16Plus digiquartz counts.

    Applies the digiquartz pressure calibration equation for SBE 16Plus
    instruments equipped with a digiquartz pressure sensor. Applies to
    CTDBP-N and CTDBP-O instruments only.

    Parameters
    ----------
    p0 : array_like
        Raw pressure (PRESWAT_L0) [counts].
    t0 : array_like
        Raw temperature from pressure sensor thermistor [counts].
    C1 : float
        Digiquartz pressure calibration coefficient.
    C2 : float
        Digiquartz pressure calibration coefficient.
    C3 : float
        Digiquartz pressure calibration coefficient.
    D1 : float
        Digiquartz pressure calibration coefficient.
    D2 : float
        Digiquartz pressure calibration coefficient.
    T1 : float
        Digiquartz pressure calibration coefficient.
    T2 : float
        Digiquartz pressure calibration coefficient.
    T3 : float
        Digiquartz pressure calibration coefficient.
    T4 : float
        Digiquartz pressure calibration coefficient.
    T5 : float
        Digiquartz pressure calibration coefficient.

    Returns
    -------
    p_dbar : ndarray
        Sea water pressure (PRESWAT_L1) [dbar].

    References
    ----------
    OOI (2012). Data Product Specification for Pressure (Depth).
        Document Control Number 1341-00020. [Legacy document, archived]
        https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00020_Data_Product_SPEC_PRESWAT_OOI.pdf
    """
    # Convert raw pressure input to frequency [Hz]
    pf = p0 / 256.0

    # Convert raw temperature input to voltage
    tv = t0 / 13107.0

    # Calculate U (thermistor temp):
    U = (23.7 * (tv + 9.7917)) - 273.15

    # Calculate calibration parameters
    C = C1 + C2 * U + C3 * U**2
    D = D1 + D2 * U
    T0 = T1 + T2 * U + T3 * U**2 + T4 * U**3 + T5 * U**4

    # Calculate T (pressure period, in microseconds):
    T = (1.0 / pf) * 1.0e6

    # compute pressure in psi, rescale and compute in dbar and return
    p_psi = C * (1.0 - T0**2 / T**2) * (1.0 - D * (1.0 - T0**2 / T**2))
    p_dbar = (p_psi * 0.689475729) - 10.1325
    return p_dbar


def ctd_sbe37im_preswat_instrument_recovered(p0, pt0, ptempa0, ptempa1, ptempa2,
                                             ptca0, ptca1, ptca2, ptcb0, ptcb1, ptcb2,
                                             pa0, pa1, pa2):
    """
    Compute water pressure (PRESWAT_L1) from SBE 37IM instrument-recovered counts.

    Applies the strain gauge pressure calibration equation to instrument-recovered
    pressure counts from CTDMO instruments (all series). For telemetered or
    recovered_host data use `ctd_sbe37im_preswat`.

    Parameters
    ----------
    p0 : array_like
        Raw pressure (PRESWAT_L0) recovered from the instrument [counts].
    pt0 : array_like
        Raw temperature from pressure sensor thermistor [counts].
    ptempa0 : float
        Pressure thermistor calibration coefficient.
    ptempa1 : float
        Pressure thermistor calibration coefficient.
    ptempa2 : float
        Pressure thermistor calibration coefficient.
    ptca0 : float
        Strain gauge temperature compensation coefficient.
    ptca1 : float
        Strain gauge temperature compensation coefficient.
    ptca2 : float
        Strain gauge temperature compensation coefficient.
    ptcb0 : float
        Strain gauge temperature compensation coefficient.
    ptcb1 : float
        Strain gauge temperature compensation coefficient.
    ptcb2 : float
        Strain gauge temperature compensation coefficient.
    pa0 : float
        Strain gauge pressure calibration coefficient.
    pa1 : float
        Strain gauge pressure calibration coefficient.
    pa2 : float
        Strain gauge pressure calibration coefficient.

    Returns
    -------
    p_dbar : ndarray
        Sea water pressure (PRESWAT_L1) [dbar].

    Notes
    -----
    This algorithm was not included in the PRESWAT DPS as of June 2016.

    References
    ----------
    OOI (2012). Data Product Specification for Pressure (Depth).
        Document Control Number 1341-00020. [Legacy document, archived]
        https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00020_Data_Product_SPEC_PRESWAT_OOI.pdf
    """
    # compute calibration parameters
    t = ptempa0 + ptempa1 * pt0 + ptempa2 * pt0**2
    x = p0 - ptca0 - ptca1 * t - ptca2 * t**2
    n = x * ptcb0 / (ptcb0 + ptcb1 * t + ptcb2 * t**2)

    # compute pressure in psi, rescale and compute in dbar and return
    p_psi = pa0 + pa1 * n + pa2 * n**2
    p_dbar = (p_psi * 0.689475729) - 10.1325
    return p_dbar


def ctd_sbe37im_preswat(p0, p_range_psia):
    """
    Compute water pressure (PRESWAT_L1) from SBE 37IM telemetered counts.

    Converts raw pressure counts to dbar for telemetered and recovered_host
    data from CTDMO instruments (all series). For instrument-recovered data
    use `ctd_sbe37im_preswat_instrument_recovered`.

    Parameters
    ----------
    p0 : array_like
        Raw pressure (PRESWAT_L0) [counts].
    p_range_psia : float
        Pressure range calibration coefficient [psia].

    Returns
    -------
    p_dbar : ndarray
        Sea water pressure (PRESWAT_L1) [dbar].

    References
    ----------
    OOI (2012). Data Product Specification for Pressure (Depth).
        Document Control Number 1341-00020. [Legacy document, archived]
        https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00020_Data_Product_SPEC_PRESWAT_OOI.pdf
    """
    # compute pressure range in units of dbar
    p_range_dbar = (p_range_psia - 14.7) * 0.6894757

    # compute pressure in dbar and return
    p_dbar = p0 * p_range_dbar / (0.85 * 65536.0) - 0.05 * p_range_dbar
    return p_dbar


def ctd_glider_preswat(pr_bar):
    """
    Compute water pressure (PRESWAT_L1) from glider CTD pressure in bar.

    Converts pressure reported in bar by Seabird CTDs installed on gliders
    to dbar. Used for CTDGV instruments.

    Parameters
    ----------
    pr_bar : array_like
        Sea water pressure reported by the glider [bar].

    Returns
    -------
    pr_dbar : ndarray
        Sea water pressure (PRESWAT_L1) [dbar].

    References
    ----------
    OOI (2015). Data Product Specification for Pressure (Depth).
        Document Control Number 1341-00020. [Legacy document, archived]
        https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00020_Data_Product_SPEC_PRESWAT_OOI.pdf
    """

    pr_dbar = pr_bar * 10.0
    return pr_dbar


def ctd_sbe52mp_preswat(p0):
    """
    Compute water pressure (PRESWAT_L1) from SBE 52MP raw counts.

    Converts raw pressure counts to dbar for CTDPF instruments (series C,
    K, and L).

    Parameters
    ----------
    p0 : array_like
        Raw pressure (PRESWAT_L0) [counts].

    Returns
    -------
    p_dbar : ndarray
        Sea water pressure (PRESWAT_L1) [dbar].

    References
    ----------
    OOI (2012). Data Product Specification for Pressure (Depth).
        Document Control Number 1341-00020. [Legacy document, archived]
        https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00020_Data_Product_SPEC_PRESWAT_OOI.pdf
    """

    p_dbar = p0 / 100.0 - 10.0
    return p_dbar


def ctd_sbe16plus_condwat(c0, t1, p1, g, h, i, j, cpcor, ctcor):
    """
    Compute water conductivity (CONDWAT_L1) from SBE 16Plus raw counts.

    Applies the SBE conductivity calibration equation to convert raw counts
    to S m-1. Used for CTDBP instruments (all series) and CTDPF instruments
    (series A and B).

    Parameters
    ----------
    c0 : array_like
        Raw conductivity (CONDWAT_L0) [counts].
    t1 : array_like
        Sea water temperature (TEMPWAT_L1) [deg_C].
    p1 : array_like
        Sea water pressure (PRESWAT_L1) [dbar].
    g : float
        Conductivity calibration coefficient.
    h : float
        Conductivity calibration coefficient.
    i : float
        Conductivity calibration coefficient.
    j : float
        Conductivity calibration coefficient.
    cpcor : float
        Pressure correction factor for conductivity.
    ctcor : float
        Temperature correction factor for conductivity.

    Returns
    -------
    c : ndarray
        Sea water conductivity (CONDWAT_L1) [S m-1].

    References
    ----------
    OOI (2012). Data Product Specification for Conductivity.
        Document Control Number 1341-00030. [Legacy document, archived]
        https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00030_Data_Product_SPEC_CONDWAT_OOI.pdf
    """
    # convert raw conductivity measurement to frequency
    f = (c0 / 256.0) / 1000.0

    # calculate conductivity [S m-1]
    c = (g + h * f**2 + i * f**3 + j * f**4) / (1 + ctcor * t1 + cpcor * p1)
    return c


def ctd_sbe37im_condwat_instrument_recovered(c0, t1, p1, g, h, i, j, cpcor, ctcor, wbotc):
    """
    Compute water conductivity (CONDWAT_L1) from SBE 37IM instrument-recovered counts.

    Applies the SBE conductivity calibration equation to instrument-recovered
    conductivity counts from CTDMO instruments (all series). For telemetered
    or recovered_host data use `ctd_sbe37im_condwat`.

    Parameters
    ----------
    c0 : array_like
        Raw conductivity (CONDWAT_L0) recovered from the instrument [counts].
    t1 : array_like
        Sea water temperature (TEMPWAT_L1) [deg_C].
    p1 : array_like
        Sea water pressure (PRESWAT_L1) [dbar].
    g : float
        Conductivity calibration coefficient.
    h : float
        Conductivity calibration coefficient.
    i : float
        Conductivity calibration coefficient.
    j : float
        Conductivity calibration coefficient.
    cpcor : float
        Pressure correction factor for conductivity.
    ctcor : float
        Temperature correction factor for conductivity.
    wbotc : float
        Temperature coefficient for conductivity cell thermal mass correction.

    Returns
    -------
    c : ndarray
        Sea water conductivity (CONDWAT_L1) [S m-1].

    Notes
    -----
    This algorithm was not included in the CONDWAT DPS as of June 2016.

    References
    ----------
    OOI (2012). Data Product Specification for Conductivity.
        Document Control Number 1341-00030. [Legacy document, archived]
        https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00030_Data_Product_SPEC_CONDWAT_OOI.pdf
    """
    # convert raw conductivity measurement to frequency
    f = (c0 / 256.0) / 1000.0 * np.sqrt(1.0 + wbotc * t1)

    # calculate conductivity [S m-1]
    c = (g + h * f**2 + i * f**3 + j * f**4) / (1 + ctcor * t1 + cpcor * p1)
    return c


def ctd_sbe37im_condwat(c0):
    """
    Compute water conductivity (CONDWAT_L1) from SBE 37IM telemetered counts.

    Converts raw conductivity counts to S m-1 for telemetered and
    recovered_host data from CTDMO instruments (all series). For
    instrument-recovered data use `ctd_sbe37im_condwat_instrument_recovered`.

    Parameters
    ----------
    c0 : array_like
        Raw conductivity (CONDWAT_L0) [counts].

    Returns
    -------
    c : ndarray
        Sea water conductivity (CONDWAT_L1) [S m-1].

    References
    ----------
    OOI (2012). Data Product Specification for Conductivity.
        Document Control Number 1341-00030. [Legacy document, archived]
        https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00030_Data_Product_SPEC_CONDWAT_OOI.pdf
    """

    c = c0 / 100000.0 - 0.5
    return c


def ctd_sbe52mp_condwat(c0):
    """
    Compute water conductivity (CONDWAT_L1) from SBE 52MP raw counts.

    Converts raw conductivity counts to S m-1 for CTDPF instruments (series
    C, K, and L).

    Parameters
    ----------
    c0 : array_like
        Raw conductivity (CONDWAT_L0) [counts].

    Returns
    -------
    c : ndarray
        Sea water conductivity (CONDWAT_L1) [S m-1].

    References
    ----------
    OOI (2012). Data Product Specification for Conductivity.
        Document Control Number 1341-00030. [Legacy document, archived]
        https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00030_Data_Product_SPEC_CONDWAT_OOI.pdf
    """

    c_mmho_cm = c0 / 10000.0 - 0.5
    c_S_m = 0.1 * c_mmho_cm
    return c_S_m


def ctd_pracsal(c, t, p):
    """
    Compute practical salinity (PRACSAL_L2) from conductivity, temperature, and pressure.

    Calculates practical salinity (PSS-78) using the TEOS-10 Gibbs Seawater
    (GSW) library from L1 CTD data products.

    Parameters
    ----------
    c : array_like
        Sea water conductivity (CONDWAT_L1) [S m-1].
    t : array_like
        Sea water temperature (TEMPWAT_L1) [deg_C].
    p : array_like
        Sea water pressure (PRESWAT_L1) [dbar].

    Returns
    -------
    SP : ndarray
        Practical salinity, PSS-78 (PRACSAL_L2) [unitless].

    References
    ----------
    OOI (2012). Data Product Specification for Salinity.
        Document Control Number 1341-00040. [Legacy document, archived]
        https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00040_Data_Product_SPEC_PRACSAL_OOI.pdf
    """

    # Convert L1 Conductivity from S/m to mS/cm
    C10 = c * 10.0

    # Calculate the Practical Salinity (PSS-78) [unitless]
    SP = gsw.SP_from_C(C10, t, p)
    return SP


def ctd_density(SP, t, p, lat, lon):
    """
    Compute sea water density (DENSITY_L2) from salinity, temperature, and pressure.

    Calculates in-situ density using the TEOS-10 Gibbs Seawater (GSW) library
    via absolute salinity and conservative temperature.

    Parameters
    ----------
    SP : array_like
        Practical salinity, PSS-78 (PRACSAL_L2) [unitless].
    t : array_like
        Sea water temperature (TEMPWAT_L1) [deg_C].
    p : array_like
        Sea water pressure (PRESWAT_L1) [dbar].
    lat : array_like
        Latitude of measurement [decimal degrees N].
    lon : array_like
        Longitude of measurement [decimal degrees E].

    Returns
    -------
    rho : ndarray
        Sea water density (DENSITY_L2) [kg m-3].

    References
    ----------
    OOI (2012). Data Product Specification for Density.
        Document Control Number 1341-00050. [Legacy document, archived]
        https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00050_Data_Product_SPEC_DENSITY_OOI.pdf
    """
    # Calculate the density [kg m-3]
    sa = gsw.SA_from_SP(SP, p, lon, lat)  # absolute salinity
    ct = gsw.CT_from_t(sa, t, p)  # conservative temperature
    rho = gsw.rho(sa, ct, p)  # density
    return rho
