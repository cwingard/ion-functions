#!/usr/bin/env python
"""
Functions for processing data from the OOI Hydrophone instrument family.

Covers Broadband Hydrophone (HYDBB) and Low Frequency Hydrophone (HYDLF)
instruments, producing L1 acoustic pressure wave data products HYDAPBB and
HYDAPLF.
"""

import numpy as np


def hyd_bb_acoustic_pwaves(wav, gain):
    """
    Compute broadband acoustic pressure waves (HYDAPBB_L1) from raw voltage.

    Applies external gain correction and scaling to convert raw 24-bit
    hydrophone voltages to calibrated pressure wave voltages.

    Parameters
    ----------
    wav : array_like
        Raw time-series voltage [V] (HYDAPBB_L0). Shape (n_records, n_samples)
        or 1-D for a single record.
    gain : scalar or array_like
        External gain setting [dB]. Scalar or 1-D array of length n_records.

    Returns
    -------
    tsv : ndarray
        Time-series voltage compensated for external gain and wav format
        scaling (HYDAPBB_L1) [V].

    Notes
    -----
    The HYDBB instrument senses passive acoustic pressure waves from 5 Hz to
    100 kHz at 24-bit resolution. Raw voltages are scaled by 3 V full-scale
    before gain correction.
    """
    # shape inputs to correct dimensions
    wav = np.atleast_2d(wav)
    n_rec = wav.shape[0]

    if np.isscalar(gain) is True:
        gain = np.tile(gain, (n_rec, 1))
    else:
        gain = np.reshape(gain, (n_rec, 1))

    # Convert the gain from dB to a linear value
    gain = 10**(gain/20.)

    # convert the broadband acoustic pressure wave data to Volts
    volts = wav * 3.

    # and correct for the gain
    tsv = volts / gain
    return tsv


def hyd_lf_acoustic_pwaves(raw, gain=3.2):
    """
    Compute low frequency acoustic pressure waves (HYDAPLF_L1) from raw counts.

    Converts raw digitized counts from the Low Frequency Hydrophone (HYDLF)
    to calibrated voltage using the instrument's fixed gain bit weight.

    Parameters
    ----------
    raw : array_like
        Raw time-series digitized in counts (HYDAPLF_L0) [counts].
    gain : float, optional
        Guralp DM24 fixed gain bit weight [uV/count]. Default is 3.2.

    Returns
    -------
    hydaplf : ndarray
        Time-series of low frequency acoustic pressure waves (HYDAPLF_L1) [V].

    References
    ----------
    OOI (2013). Data Product Specification for Low Frequency Acoustic Pressure
        Waves. Document Control Number 1341-00821. [Legacy document, archived]
        https://oceanobservatories.org/wp-content/uploads/2023/09/1341-00821_Data_Product_SPEC_HYDAPLF_OOI.pdf
    """
    # apply the gain correction to convert the signal from counts to V
    gain = gain * 1.0e-6
    hydaplf = raw * gain
    return hydaplf
