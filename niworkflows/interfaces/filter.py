#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""
Temporal filtering operations for the image processing system
"""

import numpy as np
import pandas as pd
import nibabel as nb
from scipy import signal

from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, SimpleInterface, File)


class TemporalFilter4DInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='4D NIfTI time series to be temporally filtered')
    mask = File(exists=True,
                desc='Spatial mask over which the filter is to be applied')
    t_rep = traits.Either(None, traits.Float, default=None, usedefault=True,
                          desc='Repetition time (T_R)')
    filter_type = traits.Enum(
        'butterworth',
        'chebyshev1',
        'chebyshev2',
        'elliptic',
        usedefault=True,
        desc='Filter class (Butterworth, Chebyshev, or elliptic)')
    filter_order = traits.Int(1, usedefault=True,
                              desc='Temporal filter order')
    passband = traits.Tuple(traits.Float(0.01), traits.Float(0.08),
                            usedefault=True, desc='Frequency pass band')
    ripple_pass = traits.Float(5, usedefault=True,
                               desc='Pass band ripple')
    ripple_stop = traits.Float(20, usedefault=True,
                               desc='Stop band ripple')
    output_file = File(desc='Output path')


class TemporalFilter4DOutputSpec(TraitedSpec):
    output_file = File(exists=True, desc='Temporally filtered 4D NIfTI time series')


class TemporalFilter4D(SimpleInterface):
    """Temporally filter a 4D NIfTI dataset.
    """
    input_spec = TemporalFilter4DInputSpec
    output_spec = TemporalFilter4DOutputSpec

    def _run_interface(self, runtime):
        if isdefined(self.inputs.output_file):
            out_file = self.inputs.output_file
        else:
            out_file = fname_presuffix(self.inputs.in_file,
                                       suffix='_filter4D',
                                       newpath=runtime.cwd)

        self._results['output_file'] = general_filter_4d(
            timeseries_4d=self.inputs.in_file,
            timeseries_4d_out=out_file,
            brain_mask=self.inputs.mask,
            t_rep=self.inputs.t_rep,
            filter_type=self.inputs.filter_type,
            filter_order=self.inputs.filter_order,
            passband=self.inputs.passband,
            ripple_pass=self.inputs.ripple_pass,
            ripple_stop=self.inputs.ripple_stop
        )
        return runtime


class TemporalFilter2DInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='1D or 2D TSV time series to be temporally filtered')
    t_rep = traits.Float(mandatory=True, desc='Sampling time (T_R)')
    filter_type = traits.Enum(
        'butterworth',
        'chebyshev1',
        'chebyshev2',
        'elliptic',
        usedefault=True,
        desc='Filter class (Butterworth, Chebyshev, or elliptic)'
    )
    filter_order = traits.Int(1, usedefault=True,
                              desc='Temporal filter order')
    passband = traits.Tuple(traits.Float(0.01), traits.Float(0.08),
                            usedefault=True, desc='Frequency pass band')
    ripple_pass = traits.Float(5, usedefault=True,
                               desc='Pass band ripple')
    ripple_stop = traits.Float(20, usedefault=True,
                               desc='Stop band ripple')
    output_file = File(desc='Output path')


class TemporalFilter2DOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Temporally filtered TSV time series')


class TemporalFilter2D(SimpleInterface):
    """Temporally filter a 2D TSV dataset.
    """
    input_spec = TemporalFilter2DInputSpec
    output_spec = TemporalFilter2DOutputSpec

    def _run_interface(self, runtime):
        if isdefined(self.inputs.output_file):
            out_file = self.inputs.output_file
        else:
            out_file = fname_presuffix(self.inputs.in_file,
                                       suffix='_filter2D',
                                       newpath=runtime.cwd)

        self._results['out_file'] = general_filter_2d(
            timeseries_2d=self.inputs.in_file,
            t_rep=self.inputs.t_rep,
            timeseries_2d_out=out_file,
            filter_type=self.inputs.filter_type,
            filter_order=self.inputs.filter_order,
            passband=self.inputs.passband,
            ripple_pass=self.inputs.ripple_pass,
            ripple_stop=self.inputs.ripple_stop
        )
        return runtime


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


def _unfold_image(img, mask=None):
    """Unfold a four-dimensional time series into two dimensions.

    Parameters
    ----------
    img: nibabel NIfTI object
        NIfTI object corresponding to the 4D time series to be unfolded.
    mask: nibabel NIfTI object
        Mask indicating the spatial extent of the unfolding. To unfold
        only brain voxels, for instance, this should be a brain mask.

    Returns
    numpy array
        2-dimensional numpy array with rows equal to frames in the time
        series and columns equal to voxels in the mask..
    """
    if mask is not None:
        return img.get_fdata()[mask.get_data().astype('bool')]
    else:
        return img.get_fdata().reshape([-1, img.shape[3]])


def _fold_image(data, template, mask=None):
    """Fold a 2D numpy array into a 4D NIfTI time series.

    Parameters
    ----------
    data: numpy array
        2-dimensional numpy array to be folded.
    template: nibabel NIfTI object
        NIfTI object that provides header and affine information for the
        folded dataset. This might, for instance, be the original 4D time
        series that was previously unfolded into the 2D data array.
    mask: nibabel NIfTI object
        Mask indicating the spatial extent of the unfolded 2D data.

    Returns
    -------
    nibabel NIfTI object
    """
    if mask is not None:
        data_folded = np.zeros(shape=template.shape)
        data_folded[mask.get_data().astype('bool')] = data
    else:
        data_folded = data.reshape(template.shape)
    return nib.Nifti1Image(dataobj=data_folded,
                           affine=template.affine,
                           header=template.header)


def general_filter(data,
                   sampling_rate,
                   filter_type='butterworth',
                   filter_order=1,
                   passband=(0.01, 0.08),
                   ripple_pass=5,
                   ripple_stop=20):
    """Temporal filtering for any data array.

    Parameters
    ----------
    data: numpy array
        2D data array to be filtered. The input should be transformed so that
        the axis to be filtered (typically time) is the second axis and all
        remaining dimensions are unfolded into the first axis.
    sampling_rate: float
        Repetition time or sampling rate of the data along the filter axis.
    filter_type: str
        Filter class: one of `butterworth`, `chebyshev1`, `chebyshev2`, and
        `elliptic`. Note that Chebyshev and elliptic filters require
        specification of appropriate ripples.
    filter_order: int
        Order of the filter. Note that the output filter has double the
        specified order; this prevents phase-shifting of the data.
    passband: tuple
        2-tuple indicating high-pass and low-pass cutoffs for the filter.
    ripple_pass: float
        Passband ripple parameter. Required for elliptic and type I Chebyshev
        filters.
    ripple_stop: float
        Stopband ripple parameter. Required for elliptic and type II Chebyshev
        filters.

    Returns
    -------
    numpy array
        The filtered input data array.
    """
    passband_norm, filter_pass = _get_norm_passband(passband, sampling_rate)

    if filter_type == 'butterworth':
        filt = signal.butter(N=filter_order,
                             Wn=passband_norm,
                             btype=filter_pass)
    elif filter_type == 'chebyshev1':
        filt = signal.cheby1(N=filter_order,
                             rp=ripple_pass,
                             Wn=passband_norm,
                             btype=filter_pass)
    elif filter_type == 'chebyshev2':
        filt = signal.cheby2(N=filter_order,
                             rs=ripple_stop,
                             Wn=passband_norm,
                             btype=filter_pass)
    elif filter_type == 'elliptic':
        filt = signal.ellip(N=filter_order,
                            rp=ripple_pass,
                            rs=ripple_stop,
                            Wn=passband_norm,
                            btype=filter_pass)
    ##########################################################################
    #TODO this block needs some work.
    # Mask nans and filter the data. Should probably interpolate over those
    # nans though.
    #TODO need to do something more reasonable to the nans  as this will
    #     distort the filter like hell.
    #     serial transposition is demeaning the data so that the filter is
    #     well-behaved. mean is added back at the end.
    #     also necessary to broadcast arrays.
    #TODO also need to decide what to do about means. filter will be incorrect
    #     unless data are demeaned, but the mean should potentially omit
    #     censored values
    ##########################################################################
    mask = np.isnan(data)
    data[mask] =   0
    colmeans = data.mean(1)
    data = signal.filtfilt(filt[0],filt[1],(data.T - colmeans).T,
                           method='gust')
    data[mask] = np.nan
    return (data.T + colmeans).T


def general_filter_4d(timeseries_4d,
                      timeseries_4d_out,
                      brain_mask=None,
                      t_rep=None,
                      filter_type='butterworth',
                      filter_order=1,
                      passband=(0.01, 0.08),
                      ripple_pass=5,
                      ripple_stop=20):

    """Temporally filter a 4D NIfTI dataset.

    Parameters
    ----------
    timeseries_4d: str
        Path to the 4-dimensional NIfTI time series dataset that is to be
        filtered. The filter is applied over the fourth (temporal) axis.
    timeseries_4d_out: str
        Path where the filtered time series dataset will be saved.
    brain_mask: str
        Binary-valued NIfTI image wherein the value of each voxel indicates
        whether that voxel is part of the brain (and should consequently be
        filtered). Providing a mask substantially speeds the filtering
        procedure.
    t_rep: float
        Repetition time of the dataset. If this isn't explicitly provided,
        then it will be automatically inferred from the image header.
    filter_type: str
        Filter class: one of `butterworth`, `chebyshev1`, `chebyshev2`, and
        `elliptic`. Note that Chebyshev and elliptic filters require
        specification of appropriate ripples.
    filter_order: int
        Order of the filter. Note that the output filter has double the
        specified order; this prevents phase-shifting of the data.
    passband: tuple
        2-tuple indicating high-pass and low-pass cutoffs for the filter.
    ripple_pass: float
        Passband ripple parameter. Required for elliptic and type I Chebyshev
        filters.
    ripple_stop: float
        Stopband ripple parameter. Required for elliptic and type II Chebyshev
        filters.

    Returns
    -------
    str
        Path to the saved and filtered NIfTI time series.
    """
    img = nb.load(timeseries_4d)
    t_rep = t_rep or img.header.get_zooms()[3]
    img_data = _unfold_image(img, brain_mask)

    img_data = general_filter(data=img_data,
                              sampling_rate=1/t_rep,
                              filter_type=filter_type,
                              passband=passband,
                              ripple_pass=ripple_pass,
                              ripple_stop=ripple_stop)

    img_filtered = _fold_image(img_data, img, brain_mask)
    nib.save(img_filtered, timeseries_4d_out)
    return timeseries_4d_out


def general_filter_2d(timeseries_2d,
                      timeseries_2d_out,
                      t_rep,
                      filter_type='butterworth',
                      filter_order=1,
                      passband=[0.01, 0.08],
                      ripple_pass=5,
                      ripple_stop=20):
    """Temporally filter a 2D TSV dataset.

    Parameters
    ----------
    timeseries_2d: str
        Path to the 2-dimensional TSV time series dataset that is to be
        filtered. The filter is applied to each column, over rows.
    timeseries_2d_out: str
        Path where the filtered time series dataset will be saved.
    t_rep: float
        Repetition time of the dataset.For 2-dimensional data, this argument
        is required.
    filter_type: str
        Filter class: one of `butterworth`, `chebyshev1`, `chebyshev2`, and
        `elliptic`. Note that Chebyshev and elliptic filters require
        specification of appropriate ripples.
    filter_order: int
        Order of the filter. Note that the output filter has double the
        specified order; this prevents phase-shifting of the data.
    passband: tuple
        2-tuple indicating high-pass and low-pass cutoffs for the filter.
    ripple_pass: float
        Passband ripple parameter. Required for elliptic and type I Chebyshev
        filters.
    ripple_stop: float
        Stopband ripple parameter. Required for elliptic and type II Chebyshev
        filters.

    Returns
    -------
    str
        Path to the saved and filtered TSV time series.
    """
    tsv_data = pd.read_csv(timeseries_2d, sep='\t')

    tsv_data[:] = general_filter(data=tsv_data.values.T,
                                 sampling_rate=1/t_rep,
                                 filter_type=filter_type,
                                 passband=passband,
                                 ripple_pass=ripple_pass,
                                 ripple_stop=ripple_stop).T

    tsv_data.to_csv(timeseries_2d_out, sep='\t', index=False, na_rep='n/a')
    return timeseries_2d_out
