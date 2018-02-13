# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Plotting tools shared across MRIQC and FMRIPREP"""

import numpy as np
import matplotlib.pyplot as plt

from nilearn import _utils
from nilearn.input_data import NiftiMasker
from nilearn.signal import clean
from nilearn._utils import check_niimg_4d
from nilearn._utils.niimg import _safe_get_data


def plot_series(tsv_file, output_file=None, figure=None, axes=None, title=None,
                tr=1):
    """
    Plots time-series in a file (e.g. a confounds file)

    """

    # Prepare figure and axes
    if not figure:
        figure = axes.figure if axes else plt.gcf()

    if not axes:
        axes = plt.gca()


    # Read the tsv_file

    # Create layout (number of signals and timepoints)

    # Plot series

    # Add markers (if any)

    return figure



def fmricarpetplot(img, segmentation, detrend=True, tr=None, nskip=0, long_cutoff=800,
                   axes=None, title=None, output_file=None):
    """
    Plot an image representation of voxel intensities across time also know
    as the "carpet plot" or "Power plot". See Jonathan Power Neuroimage
    2017 Jul 1; 154:150-158.

    Parameters
    ----------

        img : Niimg-like object
            See http://nilearn.github.io/manipulating_images/input_output.html
            4D input image
        mask_img : Niimg-like object, optional
            See http://nilearn.github.io/manipulating_images/input_output.html
            Limit plotted voxels to those inside the provided mask. If not
            specified a new mask will be derived from data.
        detrend : boolean, optional
            Detrend and standardize the data prior to plotting.
        output_file : string, or None, optional
            The name of an image file to export the plot to. Valid extensions
            are .png, .pdf, .svg. If output_file is not None, the plot
            is saved to a file, and the display is closed.
        figure : matplotlib figure, optional
            Matplotlib figure used. If None is given, a
            new figure is created.
        axes : matplotlib axes, optional
            The axes used to display the plot. If None, the complete
            figure is used.
        title : string, optional
            The title displayed on the figure.
    """
    img_nii = check_niimg_4d(img, dtype='auto')
    func_data = _safe_get_data(img_nii, ensure_finite=True)

    # Define TR and number of frames
    tr = img_nii.header.get_zooms()[-1]
    ntsteps = func_data.shape[-1]

    data = func_data[segmentation > 0].reshape(-1, ntsteps)
    if detrend:  # Detrend data
        data = clean(data.T, t_r=tr).T

    # Order following segmentation labels
    seg = segmentation[segmentation > 0].reshape(-1)
    seg_labels = np.unique(seg)

    # Labels meaning
    cort_gm = seg_labels[(seg_labels > 100) & (seg_labels < 200)].tolist()
    deep_gm = seg_labels[(seg_labels > 30) & (seg_labels < 100)].tolist()
    cerebellum = [255]
    wm_csf = seg_labels[seg_labels < 10].tolist()
    seg_labels = cort_gm + deep_gm + cerebellum + wm_csf

    label_id = 0
    newsegm = np.zeros_like(seg)
    for _lab in seg_labels:
        newsegm[seg == _lab] = label_id
        label_id += 1
    order = np.argsort(newsegm)

    # Avoid segmentation faults for long acquisitions by decimating the input data
    decimation = 1 + data.shape[1] // long_cutoff
    if decimation:
        data = data[order, ::decimation]

    # Define nested GridSpec
    gs = mgs.GridSpecFromSubplotSpec(1, 2, subplot_spec=axes,
                                     width_ratios=[1, 100], wspace=0.0)

    # Segmentation colorbar
    ax0 = plt.subplot(gs[0])
    ax0.set_yticks([])
    ax0.set_xticks([])

    colors1 = plt.cm.summer(np.linspace(0., 1., len(cort_gm)))
    colors2 = plt.cm.autumn(np.linspace(0., 1., len(deep_gm) + 1))[::-1, ...]
    colors3 = plt.cm.winter(np.linspace(0., .5, len(wm_csf)))[::-1, ...]
    cmap = LinearSegmentedColormap.from_list('my_colormap', np.vstack((colors1, colors2, colors3)))

    ax0.imshow(newsegm[order, np.newaxis], interpolation='nearest', aspect='auto',
               cmap=cmap, vmax=len(seg_labels) - 1, vmin=0)
    ax0.grid(False)
    ax0.set_ylabel('voxels')

    # Carpet plot
    ax1 = plt.subplot(gs[1])
    ax1.imshow(data, interpolation='nearest',
               aspect='auto', cmap='gray', vmin=-2, vmax=2)

    ax1.grid(False)
    ax1.set_yticks([])
    ax1.set_yticklabels([])

    # Set 10 frame markers in X axis
    interval = max((int(data.shape[-1] + 1) // 10, int(data.shape[-1] + 1) // 5, 1))
    xticks = list(range(0, data.shape[-1])[::interval])
    ax1.set_xticks(xticks)
    ax1.set_xlabel('time (s)')
    labels = tr * (np.array(xticks)) * decimation
    ax1.set_xticklabels(['%.02f' % t for t in labels.tolist()])

    # Remove and redefine spines
    for side in ["top", "right"]:
        # Toggle the spine objects
        ax0.spines[side].set_color('none')
        ax0.spines[side].set_visible(False)
        ax1.spines[side].set_color('none')
        ax1.spines[side].set_visible(False)

    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.spines["bottom"].set_position(('outward', 20))
    ax1.spines["left"].set_color('none')
    ax1.spines["left"].set_visible(False)

    ax0.spines["left"].set_position(('outward', 20))
    ax0.spines["bottom"].set_color('none')
    ax0.spines["bottom"].set_visible(False)

    return [ax0, ax1], gs

def plot_carpet(img, mask_img=None, detrend=True, output_file=None,
                figure=None, axes=None, title=None):
    """
    Plot an image representation of voxel intensities across time also know
    as the "carpet plot" or "Power plot". See Jonathan Power Neuroimage
    2017 Jul 1; 154:150-158.

    Parameters
    ----------

        img : Niimg-like object
            See http://nilearn.github.io/manipulating_images/input_output.html
            4D input image
        mask_img : Niimg-like object, optional
            See http://nilearn.github.io/manipulating_images/input_output.html
            Limit plotted voxels to those inside the provided mask. If not
            specified a new mask will be derived from data.
        detrend : boolean, optional
            Detrend and standardize the data prior to plotting.
        output_file : string, or None, optional
            The name of an image file to export the plot to. Valid extensions
            are .png, .pdf, .svg. If output_file is not None, the plot
            is saved to a file, and the display is closed.
        figure : matplotlib figure, optional
            Matplotlib figure used. If None is given, a
            new figure is created.
        axes : matplotlib axes, optional
            The axes used to display the plot. If None, the complete
            figure is used.
        title : string, optional
            The title displayed on the figure.
    """
    img_nii = _utils.check_niimg_4d(img, dtype='auto')
    img_data = _safe_get_data(img_nii, ensure_finite=True)

    # Define TR and number of frames
    tr = img_nii.header.get_zooms()[-1]
    ntsteps = img_nii.shape[-1]

    if not mask_img:
        nifti_masker = NiftiMasker(mask_strategy='epi', standardize=False)
        mask_data = nifti_masker.mask_img_.get_data().astype(bool)
    else:
        mask_nii = _utils.check_niimg_3d(img, dtype='auto')
        mask_data = _safe_get_data(mask_nii, ensure_finite=True)

    data = img_data[mask_data > 0].reshape(-1, ntsteps)
    # Detrend data
    if detrend:
        data = clean(data.T, t_r=tr).T

    if not figure:
        if not axes:
            figure = plt.figure()
        else:
            figure = axes.figure

    if not axes:
        axes = figure.add_subplot(1, 1, 1)
    else:
        assert axes.figure is figure, ("The axes passed are not "
                                       "in the figure")

    # Avoid segmentation faults for long acquisitions by decimating the input
    # data
    long_cutoff = 800
    if data.shape[1] > long_cutoff:
        data = data[:, ::2]
    else:
        data = data[:, :]

    axes.imshow(data, interpolation='nearest',
                aspect='auto', cmap='gray', vmin=-2, vmax=2)

    axes.grid(False)
    axes.set_yticks([])
    axes.set_yticklabels([])

    # Set 10 frame markers in X axis
    interval = max(
        (int(data.shape[-1] + 1) // 10, int(data.shape[-1] + 1) // 5, 1))
    xticks = list(range(0, data.shape[-1])[::interval])
    axes.set_xticks(xticks)

    axes.set_xlabel('time (s)')
    axes.set_ylabel('voxels')
    if title:
        axes.set_title(title)
    labels = tr * (np.array(xticks))
    if data.shape[1] > long_cutoff:
        labels *= 2
    axes.set_xticklabels(['%.02f' % t for t in labels.tolist()])

    # Remove and redefine spines
    for side in ["top", "right"]:
        # Toggle the spine objects
        axes.spines[side].set_color('none')
        axes.spines[side].set_visible(False)

    axes.yaxis.set_ticks_position('left')
    axes.xaxis.set_ticks_position('bottom')
    axes.spines["bottom"].set_position(('outward', 20))
    axes.spines["left"].set_position(('outward', 20))

    if output_file is not None:
        figure.savefig(output_file)
        figure.close()
        figure = None

    return figure
