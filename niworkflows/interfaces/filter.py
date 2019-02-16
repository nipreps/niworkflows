#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""
Temporal filtering operations for the image processing system
"""

import numpy as np


def _validate_passband(filter_type, passband):
    """Verify that the filter passband is reasonably formulated.

    Parameters
    ----------
    filter_type: str
        String indicating the type of filter.
    passband: tuple
        2-tuple indicating high-pass and low-pass cutoffs for the filter.
    """
    if passband[0] > passband[1]:
        raise ValueError(
            '\n'
            'High-pass cutoff must be less than low-pass cutoff\n'
            'for the selected filter type.\n'
            '==================================================\n'
            'Filter class:     {0}\n'
            'High-pass cutoff: {1[0]}\n'
            'Low-pass cutoff:  {1[1]}\n'.format(filter_type, passband))


def _get_fsl_passband(passband, sampling_rate):
    """
    Convert the passband to a form that FSL can understand.
    For use when filtering with a Gaussian kernel.
    1) Convert the cutoff frequencies from Hz (cycles per second) to cycles
       per repetition.
    2) Convert from frequency cutoff (in Hz) to cycle cutoff (in s). 
    3) Then, determine how many cycles of the cutoff per repetition.

    Parameters
    ----------
    passband: tuple
        2-tuple indicating high-pass and low-pass cutoffs for the filter.
    sampling_rate: float
        Repetition time.

    Returns
    -------
    tuple
        FSL-compatible passband.
    """
    passband_frequency = (0, -1)
    passband_frequency[0] = (1/passband[0])/(2 * sampling_rate)
    if passband[1] == 'nyquist':
        passband_frequency[1] = -1
    else:
        passband_frequency[1] = 1/passband[1]/(2 * sampling_rate)
    return passband_frequency


def _normalise_passband(passband, nyquist):
    """Normalise the passband according to the Nyquist frequency."""
    passband_norm = (0, 1)

    passband_norm[0] = float(passband[0])/nyquist
    if passband[1] == 'nyquist':
        passband_norm[1] = 1
    else:
        passband_norm[1] = float(passband[1])/nyquist
    return passband_norm


def _get_norm_passband(passband, sampling_rate):
    """Convert the passband to normalised form.

    Parameters
    ----------
    passband: tuple
        2-tuple indicating high-pass and low-pass cutoffs for the filter.
    sampling_rate: float
        Repetition time.

    Returns
    -------
    tuple
        Cutoff frequencies normalised between 0 and Nyquist.
    None or str
        Indicator of whether the filter is permissive at high or low
        frequencies. If None, this determination is 
    """
    nyquist = 0.5 * sampling_rate
    passband_norm = _normalise_passband(passband, nyquist)

    if passband_norm[1] >= 1:
        filter_pass = 'highpass'
    if passband_norm[0] <= 0:
        if filter_pass == 'highpass':
            raise ValueError(
                '\n'
                'Permissive filter for the specified sampling rate.\n'
                '==================================================\n'
                'Sampling rate:     {0}\n'
                'Nyquist frequency: {1}\n'
                'High-pass cutoff:  {2[0]}\n'
                'Low-pass cutoff:   {2[1]}\n'.format(
                    sampling_rate, nyquist, passband))
        passband_norm[0] = 0
        filter_pass = 'lowpass'

    if filter_pass == 'highpass':
        passband_norm = passband_norm[0]
    elif filter_pass == 'lowpass':
        passband_norm = passband_norm[1]
    elif passband_norm[0] > passband_norm[1]:
        filter_pass = 'bandstop'
    else:
        filter_pass = 'bandpass'
    return passband_norm, filter_pass
