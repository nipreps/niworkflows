#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Temporal filtering workflow based on a Nipype translation of:
    https://github.com/MidnightScanClub/MSC_Gratton2018_Codebase/blob/ ...
    master/FCProcessing/FCPROCESS_MSC_task.m

Original paper: https://www.ncbi.nlm.nih.gov/pubmed/23994314
"""

from multiprocessing import cpu_count
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu, fsl, afni
from ..interfaces.filter import (
    TemporalFilter4D, TemporalFilter2D,
    Interpolate4D, Interpolate2D,
    DemeanDetrend4D, DemeanDetrend2D,
    RestoreNaN
)

def init_temporal_filter_wf(t_rep,
                            detrend=True,
                            interpolate=True,
                            name='temporal_filter_wf',
                            omp_nthreads=None,
                            mem_gb=3.0,
                            passband=(0.01, 0.08),
                            filter_type="butterworth",
                            filter_order=1,
                            detrend_order=0,
                            ripple_pass=5,
                            ripple_stop=20):
    """
    This workflow synchronously filters the principal 4-dimensional analyte
    image and any 2-dimensional and 4-dimensional confound files.

    This workflow follows the recommendations from Power et al. (2014):
    https://www.ncbi.nlm.nih.gov/pubmed/23994314

      1. Demeaning and polynomial detrending of all inputs so as to ensure a
         well-behaved filter.
      2. Interpolation over any frames flagged for data quality and any
         missing (NaN or n/a) values using an approach based on the
         Lomb-Scargle periodogram for unevenly sampled data.
      3. Temporal filtering of all interpolated data using the user-specified
         filter. For Gaussian and Fourier filters, this calls a subroutine to
         process any 2-dimensional datasets.
      4. Restoration of missing values to the dataset.
      5. Adding back the mean from the demean/detrend step.


    .. workflow::
        :graph2use: orig
        :simple_form: yes
        from niworkflows.func import init_temporal_filter_wf
        wf = init_temporal_filter_wf()


    **Parameters**
        
        t_rep: float
            Repetition time of all data to be filtered.
        detrend: bool
            Indicates whether the data should be detrended prior to filtering.
            Detrending is required to ensure well-behaved Butterworth,
            Chebyshev, and elliptic filters, but may be disabled for other
            filter classes.
        interpolate: bool
            Indicates whether the data should be interpolated prior to
            filtering on the basis of the provided temporal mask.
        name : str
            Name of workflow (default: ``temporal_filter_wf``)
        omp_nthreads : int
            Maximum number of threads an individual process may use
        mem_gb : float
            Estimated peak memory consumption of the most hungry nodes
            in the workflow
        passband : tuple
            A 2-tuple of frequencies, in Hertz. The first element of the tuple
            corresponds to the high-pass cutoff frequency for the temporal
            filter, while the second corresponds to the low-pass cutoff
            frequency for the temporal filter.
            * Default behaviour implements a bandpass filter.
            * To implement a high-pass filter, set the low-pass frequency to
              the string value ``nyquist``.
            * To implement a low-pass filter, set the high-pass frequency to
              0.
            * Butterworth, Chebyshev, and elliptic filters support bandstop
              filters; these can be implemented by setting the high-pass
              frequency higher than the low-pass frequency.
        filter_type : str
            The type of temporal filter to be applied. Valid options include
            ``butterworth`` (default), ``fourier``, ``gaussian``,
            ``chebyshev1``, ``chebyshev2``, and ``elliptic``.
        filter_order : str
            The order of the filter. Has no effect for ``gaussian`` or
            ``fourier`` filters. In practice, the actual filter order is
            double the order specified by the input parameter because the
            filter is applied first forward and then backward to ensure zero
            phase shift.
        detrend_order: int
            The degree of polynomial to be fit as part of the detrend
            protocol.
            (order 0: demean; order 1: linear; order 2: quadratic; . . .)
        ripple_pass : str
            The ripple in the passband. Affects only ``chebyshev1`` and
            ``elliptic`` filters.
        ripple_stop : str
            The attenuation factor in the stopband. Affects only ``elliptic``
            and ``chebyshev2`` filters.


    **Inputs**
        
        timeseries_4d
            List of 4D time series formatted as a NIfTI file.
        timeseries_2d
            List of 1D or 2D time series files, for instance matrices of
            confound regressors.
        brain_mask
            (optional) A binary-valued NIfTI image indicating, for each voxel,
            whether it contains brain tissue. Used to limit computation to a
            specific region: substantially accelerates processing.
        temporal_mask
            (optional) A binary valued text file equal in length to the
            temporal dimension of ``timeseries_4d`` indicating, for each time
            point, whether it should be included in processing.


    **Outputs**
        
        timeseries_4d_filtered
            The filtered and fully processed ``timeseries_4d``.
        timeseries_2d_filtered
            The filtered and fully processed ``timeseries_2d``.


    """
    workflow = pe.Workflow(name=name)

    if omp_nthreads is None or omp_nthreads < 1:
        omp_nthreads = cpu_count()

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['timeseries_4d', 'timeseries_2d',
                'brain_mask', 'temporal_mask']),
        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['timeseries_4d_filtered', 'timeseries_2d_filtered']),
        name='outputnode')

    src_4d = (inputnode, 'timeseries_4d')
    src_2d = (inputnode, 'timeseries_2d')
    dst_4d = (outputnode, 'timeseries_4d_filtered')
    dst_2d = (outputnode, 'timeseries_2d_filtered')

    if detrend:
        dmdt_4d = pe.MapNode(
            DemeanDetrend4D(detrend_order=detrend_order),
            name='dmdt_4d', iterfield=['in_file'],
            n_procs=omp_nthreads, mem_gb=mem_gb)
        dmdt_2d = pe.MapNode(
            DemeanDetrend2D(detrend_order=detrend_order),
            name='dmdt_2d', iterfield=['in_file'])
        remean = pe.MapNode(
            fsl.maths.BinaryMaths(operation='add'),
            name='remean', iterfield=['in_file', 'operand_file'],
            n_procs=omp_nthreads, mem_gb=mem_gb)

        workflow.connect([
            (inputnode, dmdt_4d, [('brain_mask', 'mask'),
                                  ('temporal_mask', 'tmask')]),
            (inputnode, dmdt_2d, [('temporal_mask', 'tmask')]),
            (src_4d[0], dmdt_4d, [(src_4d[1], 'in_file')]),
            (src_2d[0], dmdt_2d, [(src_2d[1], 'in_file')]),
            (dmdt_4d, remean, [('output_mean', 'operand_file')]),
            (remean, dst_4d[0], [('out_file', dst_4d[1])]),
        ])
        src_4d = (dmdt_4d, 'output_file')
        src_2d = (dmdt_2d, 'output_file')
        dst_4d = (remean, 'in_file')

    if interpolate:
        interp_4d = pe.MapNode(
            Interpolate4D(t_rep=t_rep),
            name='interp_4d', iterfield=['in_file'],
            n_procs=omp_nthreads, mem_gb=mem_gb)
        interp_2d = pe.MapNode(
            Interpolate2D(t_rep=t_rep),
            name='interp_2d', iterfield=['in_file'])
        restore_nan = pe.MapNode(RestoreNaN(), name='restore_nan',
                                 iterfield=['in_file', 'idx_nan'])

        workflow.connect([
            (inputnode, interp_4d, [('brain_mask', 'mask'),
                                    ('temporal_mask', 'tmask')]),
            (inputnode, interp_2d, [('temporal_mask', 'tmask')]),
            (src_4d[0], interp_4d, [(src_4d[1], 'in_file')]),
            (src_2d[0], interp_2d, [(src_2d[1], 'in_file')]),
            (interp_2d, restore_nan, [('idx_nan', 'idx_nan')]),
            (restore_nan, dst_2d[0], [('output_file', dst_2d[1])]),
        ])
        src_4d = (interp_4d, 'output_file')
        src_2d = (interp_2d, 'output_file')
        dst_2d = (restore_nan, 'in_file')

    if filter_type == 'gaussian':
        _validate_passband(filter_type, passband)
        passband = get_fsl_passband(passband, t_rep)
        gaussian_4d = pe.MapNode(fsl.maths.TemporalFilter(
            highpass_sigma=passband[0],
            lowpass_sigma=passband[1],
            output_type='NIFTI_GZ'),
            name='gaussian_4d', iterfield=['in_file'],
            n_procs=omp_nthreads, mem_gb=mem_gb)
        gaussian_2d = pe.MapNode(fsl.maths.TemporalFilter(
            highpass_sigma=passband[0],
            lowpass_sigma=passband[1],
            output_type='NIFTI_GZ'),
            name='gaussian_2d', iterfield=['in_file'],
            n_procs=omp_nthreads, mem_gb=mem_gb)
    elif filter_type == 'fourier':
        _validate_passband(filter_type, passband)
        fourier_4d = pe.MapNode(afni.Bandpass(
            highpass=passband[0],
            lowpass=passband[1],
            outputtype='NIFTI_GZ'),
            name='fourier_4d', iterfield=['in_file'],
            n_procs=omp_nthreads, mem_gb=mem_gb)
        fourier_2d = pe.MapNode(afni.Bandpass(
            highpass=passband[0],
            lowpass=passband[1],
            outputtype='NIFTI',
            tr=t_rep),
            name='fourier_2d', iterfield=['in_file'],
            n_procs=omp_nthreads, mem_gb=mem_gb)
    else:
        signal_4d = pe.MapNode(TemporalFilter4D(
            t_rep=t_rep,
            filter_type=filter_type,
            filter_order=filter_order,
            passband=passband,
            ripple_pass=ripple_pass,
            ripple_stop=ripple_stop),
            name='signal_4d', iterfield=['in_file'],
            n_procs=omp_nthreads, mem_gb=mem_gb)
        signal_2d = pe.MapNode(TemporalFilter2D(
            t_rep=t_rep,
            filter_type=filter_type,
            filter_order=filter_order,
            passband=passband,
            ripple_pass=ripple_pass,
            ripple_stop=ripple_stop),
            name='signal_2d', iterfield=['in_file'],
            n_procs=omp_nthreads, mem_gb=mem_gb)

        workflow.connect([
            (inputnode, signal_4d, [('brain_mask', 'mask')]),
            (src_4d[0], signal_4d, [(src_4d[1], 'in_file')]),
            (src_2d[0], signal_2d, [(src_2d[1], 'in_file')]),
        ])
        src_4d = (signal_4d, 'output_file')
        src_2d = (signal_2d, 'output_file')

    workflow.connect([
        (src_4d[0], dst_4d[0], [(src_4d[1], dst_4d[1])]),
        (src_2d[0], dst_2d[0], [(src_2d[1], dst_2d[1])]),
    ])
    return workflow


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
