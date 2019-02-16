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
    output_file = File(
        exists=True, desc='Temporally filtered TSV time series')


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

        self._results['output_file'] = general_filter_2d(
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



class Interpolate4DInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='4D NIfTI time series to be transformed')
    tmask = File(exists=True, mandatory=True,
                 desc='Temporal mask')
    mask = File(exists=True,
                desc='Spatial mask')
    t_rep = traits.Float(usedefault=False,
                         desc='Repetition time (T_R)')
    os_freq = traits.Int(8, usedefault=True,
                         desc='Oversampling frequency')
    max_freq = traits.Float(
        1, usedefault=True, desc='Max frequency, as a fraction of Nyquist')
    vox_bin = traits.Int(
        5000, usedefault=True, desc='Number of voxels to transform at a time')
    output_file = File(desc='Output path')


class Interpolate4DOutputSpec(TraitedSpec):
    output_file = File(exists=True, desc='Interpolated 4D NIfTI file')


class Interpolate4D(SimpleInterface):
    """Interpolate a 4D NIfTI time series.
    """
    input_spec = Interpolate4DInputSpec
    output_spec = Interpolate4DOutputSpec

    def _run_interface(self, runtime):
        if isdefined(self.inputs.output_file):
            out_file = self.inputs.output_file
        else:
            out_file = fname_presuffix(self.inputs.in_file,
                                       suffix='_interpolate4D',
                                       newpath=runtime.cwd)

        self._results['output_file'] = interpolate_lombscargle_4d(
            timeseries_4d=self.inputs.in_file,
            timeseries_4d_out=out_file,
            brain_mask=self.inputs.mask,
            temporal_mask=self.inputs.tmask,
            t_rep=self.inputs.t_rep,
            oversampling_frequency=self.inputs.os_freq,
            maximum_frequency=self.inputs.max_freq,
            voxel_bin_size=self.inputs.vox_bin
        )
        return runtime


class Interpolate2DInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='1D or 2D TSV time series to be transformed')
    tmask = File(exists=True, mandatory=True,
                 desc='Temporal mask')
    t_rep = traits.Float(usedefault=False, mandatory=True,
                         desc='Repetition time (T_R)')
    os_freq = traits.Int(8, usedefault=True,
                         desc='Oversampling frequency')
    max_freq = traits.Float(
        1, usedefault=True, desc='Max frequency, as a fraction of Nyquist')
    output_file = File(desc='Output path')


class Interpolate2DOutputSpec(TraitedSpec):
    output_file = File(exists=True, desc='Interpolated TSV file')


class Interpolate2D(SimpleInterface):
    """Interpolate a 2D TSV time series."""
    input_spec = Interpolate2DInputSpec
    output_spec = Interpolate2DOutputSpec

    def _run_interface(self, runtime):
        if isdefined(self.inputs.output_file):
            out_file = self.inputs.output_file
        else:
            out_file = fname_presuffix(self.inputs.in_file,
                                       suffix='_interpolate2D',
                                       newpath=runtime.cwd)

        self._results['output_file'] = interpolate_lombscargle_2d(
            timeseries_2d=self.inputs.in_file,
            timeseries_2d_out=out_file,
            temporal_mask=self.inputs.tmask,
            t_rep=self.inputs.t_rep,
            oversampling_frequency=self.inputs.os_freq,
            maximum_frequency=self.inputs.max_freq
        )
        return runtime


class DemeanDetrend4DInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='4D NIfTI time series to be demeaned and detrended')
    tmask = File(exists=True, mandatory=True,
                 desc='Temporal mask')
    mask = File(exists=True,
                desc='Spatial mask')
    detrend_order = traits.Int(0, usedefault=True,
                    desc='Order of polynomial detrend (0 for demean only)')
    output_file = File(desc='Output path')
    output_mean = File(desc='Output path for mean image')


class DemeanDetrend4DOutputSpec(TraitedSpec):
    output_file = File(exists=True, desc='Demeaned and detrended NIfTI file')
    output_mean = File(exists=True, desc='Mean value computed voxelwise')


class DemeanDetrend4D(SimpleInterface):
    """Demean/detrend a 4D NIfTI time series."""
    input_spec = DemeanDetrend4DInputSpec
    output_spec = DemeanDetrend4DOutputSpec

    def _run_interface(self, runtime):
        if isdefined(self.inputs.output_file):
            out_file = self.inputs.output_file
        else:
            out_file = fname_presuffix(self.inputs.in_file,
                                       suffix='_dmdt4D',
                                       newpath=runtime.cwd)
        if isdefined(self.inputs.output_mean):
            out_mean = self.inputs.output_mean
        else:
            out_mean = fname_presuffix(self.inputs.in_file,
                                       suffix='_dmdt4Dmean',
                                       newpath=runtime.cwd)

        (self._results['output_file'],
         self._results['output_mean']) = demean_detrend_4d(
            timeseries_4d=self.inputs.in_file,
            detrend_order=self.inputs.detrend_order,
            brain_mask=self.inputs.mask,
            temporal_mask=self.inputs.tmask,
            save_mean=True,
            timeseries_4d_out=out_file,
            mean_out=out_mean
        )
        return runtime


class DemeanDetrend2DInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='1D or 2D TSV time series to be demeaned/detrended')
    tmask = File(exists=True, mandatory=True,
                 desc='Temporal mask')
    detrend_order = traits.Int(0, usedefault=True,
                    desc='Order of polynomial detrend (0 for demean only)')
    output_file = File(desc='Output path')


class DemeanDetrend2DOutputSpec(TraitedSpec):
    output_file = File(exists=True, desc='Demeaned and detrended TSV file')


class DemeanDetrend2D(SimpleInterface):
    """Demean/detrend a 2D TSV time series."""
    input_spec = DemeanDetrend2DInputSpec
    output_spec = DemeanDetrend2DOutputSpec

    def _run_interface(self, runtime):
        if isdefined(self.inputs.output_file):
            out_file = self.inputs.output_file
        else:
            out_file = fname_presuffix(self.inputs.in_file,
                                       suffix='_dmdt2D',
                                       newpath=runtime.cwd)

        self._results['output_file'] = demean_detrend_2d(
            timeseries_2d=self.inputs.in_file,
            detrend_order=self.inputs.detrend_order,
            temporal_mask=self.inputs.tmask,
            timeseries_2d_out=out_file
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
        Sampling rate (1/T_R) of the data along the filter axis.
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
        Repetition time of the dataset. For 2-dimensional data, this argument
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


def periodogram_cfg(temporal_mask_file,
                    sampling_period,
                    flag=1,
                    oversampling_frequency=8,
                    maximum_frequency=1):
    """Configure inputs for interpolate_lombscargle.

    Parameters
    ----------
    temporal_mask_file: str
        File indicating whether the value in each frame should be
        interpolated.
    sampling_period: float
        The sampling period or repetition time.
    flag: 1 or 0
        Value in the temporal_mask_file that indicates a frame should be
        interpolated.
    oversampling_frequency: int
        Oversampling frequency for the periodogram.
    maximum_frequency: float
        The maximum frequency in the dataset, as a fraction of Nyquist.
        Default 1 (Nyquist).

    Returns
    -------
    sine_term: numpy array
        Sine basis term for the periodogram.
    cosine_term: numpy array
        Cosine basis term for the periodogram.
    angular_frequencies: numpy array
        Angular frequencies for computing the periodogram.
    all_samples: numpy array
        Temporal indices of all observations, seen and unseen.
    n_samples_seen: int
        The number of seen samples (i.e., samples not flagged for
        interpolation).
    tmask: numpy array
        Boolean-valued numpy array indicating whether the value in each frame
        should be interpolated.
    """
    tmask = pd.read_csv(temporal_mask_file, sep='\t').values.astype('bool')
    n_samples = len(tmask)

    seen_samples =(np.where(tmask)[0] + 1) * sampling_period
    timespan = max(seen_samples) - min(seen_samples)
    n_samples_seen = seen_samples.shape[0]
    if n_samples_seen == n_samples:
        raise ValueError('No interpolation is necessary for this dataset.')

    all_samples = np.arange(start=sampling_period,
                            stop=sampling_period * (n_samples + 1),
                            step=sampling_period)
    sampling_frequencies = np.arange(
        start=1/(timespan * oversampling_frequency),
        step=1/(timespan * oversampling_frequency),
        stop=(maximum_frequency * n_samples_seen
              / (2 * timespan)
              + 1 / (timespan * oversampling_frequency)))
    angular_frequencies = 2 * np.pi * sampling_frequencies

    offsets = np.arctan2(
        np.sum(
            np.sin(2 * np.outer(angular_frequencies, seen_samples)),
            1),
        np.sum(
            np.cos(2 * np.outer(angular_frequencies, seen_samples)),
            1)
        ) / (2 * angular_frequencies)

    cosine_term = np.cos(
        np.outer(angular_frequencies,
                 seen_samples) -
        np.matlib.repmat(angular_frequencies * offsets,
                         n_samples_seen, 1).T)
    sine_term = np.sin(
        np.outer(angular_frequencies,
                 seen_samples) -
        np.matlib.repmat(angular_frequencies * offsets,
                         n_samples_seen, 1).T)

    return (sine_term, cosine_term, angular_frequencies, all_samples,
            n_samples_seen, tmask)


def interpolate_lombscargle(data,
                            sine_term,
                            cosine_term,
                            angular_frequencies,
                            all_samples,
                            n_samples_seen,
                            n_samples):
    """Temporally interpolate over unseen (masked) values in a dataset using
    an approach based on the Lomb-Scargle periodogram. Follows code originally
    written in MATLAB by Anish Mitra and Jonathan Power:
    https://www.ncbi.nlm.nih.gov/pubmed/23994314

    The original code can be found in the function `getTransform` here:
        https://github.com/MidnightScanClub/MSC_Gratton2018_Codebase/blob/ ...
        master/FCProcessing/FCPROCESS_MSC_task.m

    Parameters
    ----------
    data: numpy array
        Seen data to use as a reference for reconstruction of unseen data.
    sine_term: numpy array
        Sine basis term for the periodogram.
    cosine_term: numpy array
        Cosine basis term for the periodogram.
    angular_frequencies: numpy array
        Angular frequencies for computing the periodogram.
    all_samples: numpy array
        Temporal indices of all samples, seen and unseen.
    n_samples_seen: int
        The number of seen samples (i.e., samples not flagged for
        interpolation).
    n_samples: int
        The total number of samples.

    Returns
    -------
    recon: numpy array
        Input data with unseen frames reconstructed via interpolation based on
        the Lomb-Scargle periodogram.
    """
    n_features = data.shape[0]

    def _compute_term(term):
        """Compute the transform from seen data as follows for sin and cos
        terms:
        termfinal = sum(termmult,2)./sum(term.^2,2)
        Compute numerators and denominators, then divide
        """
        mult = np.zeros(shape=(angular_frequencies.shape[0],
                               n_samples_seen,
                               n_features))
        for obs in range(0,n_samples_seen):
            mult[:,obs,:] = np.outer(term[:,obs],data[:,obs])
        numerator = np.sum(mult,1)
        denominator = np.sum(term**2,1)
        term = (numerator.T/denominator).T
        return term

    def _reconstruct_term(term, fn):
        """Interpolate over unseen epochs, reconstruct the time series
        """
        term_prod = fn(np.outer(angular_frequencies, all_samples))
        term_recon = np.zeros(shape=(angular_frequencies.shape[0],
                                     n_samples,
                                     n_features))
        for i in range(angular_frequencies.shape[0]):
            term_recon[i,:,:] = np.outer(term_prod[i,:],term[i,:])
        term_recon = np.sum(term_recon,0)
        return term_recon

    c = _compute_term(cosine_term)
    s = _compute_term(sine_term)

    s_recon = _reconstruct_term(s, np.sin)
    c_recon = _reconstruct_term(c, np.cos)

    recon = (c_recon + s_recon).T
    del c_recon, s_recon

    # Normalise the reconstructed spectrum. This is necessary when the
    # oversampling frequency exceeds 1.
    std_recon = np.std(recon, 1, ddof=1)
    std_orig = np.std(data, 1, ddof=1)
    norm_fac = std_recon/std_orig
    recon = (recon.T/norm_fac).T

    return recon


def interpolate_lombscargle_4d(timeseries_4d,
                               timeseries_4d_out,
                               brain_mask=None,
                               temporal_mask=None,
                               t_rep=None,
                               oversampling_frequency=8,
                               maximum_frequency=1,
                               voxel_bin_size=5000):
    """Interpolation for 4D NIfTI time series data.

    Temporally interpolate over unseen (masked) values in a dataset using an
    approach based on the Lomb-Scargle periodogram. Follows code originally
    written in MATLAB by Anish Mitra and Jonathan Power:
    https://www.ncbi.nlm.nih.gov/pubmed/23994314

    The original code can be found in the function `getTransform` here:
        https://github.com/MidnightScanClub/MSC_Gratton2018_Codebase/blob/ ...
        master/FCProcessing/FCPROCESS_MSC_task.m

    Parameters
    ----------
    timeseries_4d: str
        Path to the 4-dimensional NIfTI time series dataset that is to be
        interpolated. The interpolation is applied over the fourth (temporal)
        axis.
    timeseries_4d_out: str
        Path where the interpolated time series dataset will be saved.
    brain_mask: str
        Binary-valued NIfTI image wherein the value of each voxel indicates
        whether that voxel is part of the brain (and should consequently be
        interpolated). Providing a mask substantially speeds the interpolation
        procedure.
    temporal_mask: str
        Temporal mask file indicating whether the value in each frame should
        be interpolated.
    t_rep: float
        Repetition time of the dataset. If this isn't explicitly provided,
        then it will be automatically inferred from the image header.
    oversampling_frequency: int
        Oversampling frequency for the periodogram.
    maximum_frequency: float
        The maximum frequency in the dataset, as a fraction of Nyquist.
        Default 1 (Nyquist).
    voxel_bin_size: int
        Because interpolation is a highly memory-intensive process, it might
        be necessary to split the dataset into several voxel bins. This will
        limit memory usage at a significant cost to processing speed.

    Returns
    -------
    str
        Path to the saved and interpolated NIfTI time series.
    """
    img = nb.load(timeseries_4d)
    t_rep = t_rep or img.header.get_zooms()[3]
    img_data = _unfold_image(img, brain_mask)
    nvox = img_data.shape[0]
    nvol_total = img_data.shape[-1]

    (sine_term, cosine_term, angular_frequencies, all_samples, nvol, tmask
        ) = periodogram_cfg(
        temporal_mask_file=temporal_mask,
        sampling_period=t_rep,
        oversampling_frequency=oversampling_frequency,
        maximum_frequency=maximum_frequency)

    n_voxel_bins = int(np.ceil(nvox / voxel_bin_size))

    for current_bin in range(0, n_voxel_bins):
        print('Voxel bin {} out of {}'.format(current_bin + 1, n_voxel_bins))

        bin_index = np.arange(start=(current_bin)*voxel_bin_size-1,
                              stop=(current_bin+1)*voxel_bin_size)
        bin_index = np.intersect1d(bin_index, range(0,nvox))
        voxel_bin = img_data[bin_index, :][:, tmask]
        recon = interpolate_lombscargle(
            data=voxel_bin,
            sine_term=sine_term,
            cosine_term=cosine_term,
            angular_frequencies=angular_frequencies,
            all_samples=all_samples,
            n_samples_seen=nvol,
            n_samples=nvol_total)

        img_data[np.ix_(bin_index, np.logical_not(tmask))] = (
            recon[:, tmask.negation()])
        del recon

    img_interpolated = _fold_image(img_data, img, brain_mask)
    nib.save(img_interpolated, timeseries_4d_out)
    return timeseries_4d_out


def interpolate_lombscargle_2d(timeseries_2d,
                               timeseries_2d_out,
                               t_rep,
                               temporal_mask=None,
                               oversampling_frequency=8,
                               maximum_frequency=1):
    
    """Interpolation for 2D TSV time series data.

    Temporally interpolate over unseen (masked) values in a dataset using an
    approach based on the Lomb-Scargle periodogram. Follows code originally
    written in MATLAB by Anish Mitra and Jonathan Power:
    https://www.ncbi.nlm.nih.gov/pubmed/23994314

    The original code can be found in the function `getTransform` here:
        https://github.com/MidnightScanClub/MSC_Gratton2018_Codebase/blob/ ...
        master/FCProcessing/FCPROCESS_MSC_task.m

    Parameters
    ----------
    timeseries_2d: str
        Path to the 2-dimensional TSV time series dataset that is to be
        interpolated. The interpolation is applied to each column, over rows.
    timeseries_2d_out: str
        Path where the interpolated time series dataset will be saved.
    t_rep: float
        Repetition time of the dataset. For 2-dimensional data, this argument
        is required.
    temporal_mask: str
        Temporal mask file indicating whether the value in each frame should
        be interpolated.
    oversampling_frequency: int
        Oversampling frequency for the periodogram.
    maximum_frequency: float
        The maximum frequency in the dataset, as a fraction of Nyquist.
        Default 1 (Nyquist).

    Returns
    -------
    str
        Path to the saved and interpolated TSV time series.
    """
    tsv_data = read_tsv(timeseries_2d, sep='\t')
    nobs_total = tsv_data.shape[0]

    (sine_term, cosine_term, angular_frequencies, all_samples, nobs, tmask
        ) = periodogram_cfg(
        temporal_mask_file=temporal_mask,
        sampling_period=t_rep,
        oversampling_frequency=oversampling_frequency,
        maximum_frequency=maximum_frequency)
    tsv_data_recon = interpolate_lombscargle(
        data=tsv_data.values[tmask.data,:].T,
        sine_term=sine_term,
        cosine_term=cosine_term,
        angular_frequencies=angular_frequencies,
        all_samples=all_samples,
        n_samples_seen=nobs,
        n_samples=nobs_total
    ).T
    tsv_data.values[np.logical_not(tmask), :] = (
        tsv_data_recon[np.logical_not(tmask), :])

    tsv_data.to_csv(timeseries_2d_out, sep='\t', index=False, na_rep='n/a')
    return timeseries_2d_out


def demean_detrend(data, detrend_order, temporal_mask=None, save_mean=True):
    """Demean and detrend the input data using a polynomial or Legendre
    polynomial fit.

    Parameters
    ----------
    data: numpy array
        Data to be detrended. The detrend is applied across rows.
    detrend_order: int
        The degree of polynomial to be fit as part of the detrend protocol.
        (order 0: demean; order 1: linear; order 2: quadratic; . . .)
    temporal_mask: str
        Temporal mask file indicating whether the value in each frame should
        be considered in the demean step.
    save_mean
        Return the fit's mean value separately.

    Returns
    -------
    numpy array
        The demeaned and detrended dataset.
    numpy array or None
        The voxelwise mean. None if save_mean is False.
    """
    data_to_fit = data
    indices = np.arange(data.shape[-1])
    if temporal_mask is not None:
        indices = np.where(tmask)[0]
        tmask = pd.read_csv(temporal_mask, sep='\t').values.astype('bool')
        data_to_fit = data.take(
            indices=np.where(tmask)[0],
            axis=-1)

    fit_coef = np.polynomial.legendre.legfit(x=indices,
                                             y=data_to_fit.T,
                                             deg=detrend_order)
    fit = np.polynomial.legendre.legval(x=indices_all,
                                        c=fit_coef)
    fit_res = data - fit
    fit_mean = fit_coef[0]

    if save_mean:
        return fit_res, fit_mean
    else:
        return fit_res, None


def demean_detrend_4d(timeseries_4d,
                      timeseries_4d_out,
                      detrend_order,
                      brain_mask=None,
                      temporal_mask=None,
                      save_mean=False,
                      mean_out=None):
    """Demean and detrend a 4D NIfTI data set.

    Parameters
    ----------
    timeseries_4d: str
        Path to the 4-dimensional NIfTI time series dataset that is to be
        demeaned and detrended via polynomial fit. The fit is applied over
        the fourth (temporal) axis.
    timeseries_4d_out: str
        Path where the detrended time series dataset will be saved.
    detrend_order: int
        The degree of polynomial to be fit as part of the detrend protocol.
        (order 0: demean; order 1: linear; order 2: quadratic; . . .)
    brain_mask: str
        Binary-valued NIfTI image wherein the value of each voxel indicates
        whether that voxel is part of the brain (and should consequently be
        detrended). Providing a mask substantially speeds the detrend
        procedure.
    temporal_mask: str
        Temporal mask file indicating whether the value in each frame should
        be considered in the polynomial fit.
    save_mean: bool
        Return the fit's mean value separately. If this is true, then mean_out
        must be defined appropriately. This can be useful if you later wish
        to add back the mean.
    mean_out: str
        Path where the mean image will be saved.

    Returns
    -------
    str
        Path to the saved and detrended NIfTI time series.
    str
        Path to the saved mean image.
    """
    img = nb.load(timeseries_4d)
    img_data = _unfold_image(img, brain_mask)

    (residuals, mean_vals) = demean_detrend(data=img_data,
                                            detrend_order=detrend_order,
                                            temporal_mask=temporal_mask,
                                            save_mean=save_mean)
    img_dmdt = _fold_image(residuals, img, brain_mask)
    nib.save(img_dmdt, timeseries_4d_out)

    if save_mean:
        if brain_mask is not None:
            data_mean = np.zeros(shape=img.shape[:-1])
            data_mean[brain_mask.get_data().astype('bool')] = mean_vals
        else:
            data_folded = mean_vals.reshape(img.shape[:-1])
        img_mean = nib.Nifti1Image(dataobj=data_mean,
                                   affine=img.affine,
                                   header=img.header)
        nib.save(img_mean, mean_out)

    return timeseries_4d_out, mean_out


def demean_detrend_2d(timeseries_2d,
                      timeseries_2d_out,
                      detrend_order,
                      temporal_mask=None):

    """Demean and detrend a 2D TSV dataset.

    Parameters
    ----------
    timeseries_2d: str
        Path to the 2-dimensional TSV time series dataset that is to be
        demeaned and detrended via polynomial fit. The fit is applied over
        the fourth (temporal) axis.
    timeseries_2d_out: str
        Path where the detrended time series dataset will be saved.
    detrend_order: int
        The degree of polynomial to be fit as part of the detrend protocol.
        (order 0: demean; order 1: linear; order 2: quadratic; . . .)
    temporal_mask: str
        Temporal mask file indicating whether the value in each frame should
        be considered in the polynomial fit.

    Returns
    -------
    str
        Saved and detrended TSV time series.
    """
    tsv_data = read_csv(timeseries_2d, sep='\t')
    tsv_data[:] = demean_detrend(data=tsv_data.values.T,
                                 detrend_order=detrend_order,
                                 temporal_mask=temporal_mask,
                                 save_mean=False)
    tsv_data.to_csv(timeseries_2d_out, sep='\t', index=False, na_rep='n/a')
    return timeseries_2d_out
