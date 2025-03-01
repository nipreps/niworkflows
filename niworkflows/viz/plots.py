# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Plotting tools shared across MRIQC and fMRIPrep."""

import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import pandas as pd
from matplotlib import cm, colormaps
from matplotlib import gridspec as mgs
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

DINA4_LANDSCAPE = (11.69, 8.27)


class fMRIPlot:
    """Generates the fMRI Summary Plot."""

    __slots__ = (
        'confounds',
        'nskip',
        'paired_carpet',
        'segments',
        'sort_carpet',
        'spikes',
        'timeseries',
        'tr',
    )

    def __init__(
        self,
        timeseries,
        segments,
        confounds=None,
        conf_file=None,
        tr=None,
        usecols=None,
        units=None,
        vlines=None,
        spikes_files=None,
        nskip=0,
        sort_carpet=True,
        paired_carpet=False,
    ):
        self.timeseries = timeseries
        self.segments = segments
        self.tr = tr
        self.nskip = nskip
        self.sort_carpet = sort_carpet
        self.paired_carpet = paired_carpet

        if units is None:
            units = {}
        if vlines is None:
            vlines = {}
        self.confounds = {}
        if confounds is None and conf_file:
            confounds = pd.read_csv(conf_file, sep=r'[\t\s]+', usecols=usecols, index_col=False)

        if confounds is not None:
            for name in confounds.columns:
                self.confounds[name] = {
                    'values': confounds[[name]].values.squeeze().tolist(),
                    'units': units.get(name),
                    'cutoff': vlines.get(name),
                }

        self.spikes = []
        if spikes_files:
            for sp_file in spikes_files:
                self.spikes.append((np.loadtxt(sp_file), None, False))

    def plot(self, figure=None):
        """Main plotter"""
        import seaborn as sns

        sns.set_style('whitegrid')
        sns.set_context('paper', font_scale=0.8)

        if figure is None:
            figure = plt.gcf()

        nconfounds = len(self.confounds)
        nspikes = len(self.spikes)
        nrows = 1 + nconfounds + nspikes

        # Create grid
        grid = mgs.GridSpec(
            nrows, 1, wspace=0.0, hspace=0.05, height_ratios=[1] * (nrows - 1) + [5]
        )

        grid_id = 0
        for tsz, name, iszs in self.spikes:
            spikesplot(tsz, title=name, outer_gs=grid[grid_id], tr=self.tr, zscored=iszs)
            grid_id += 1

        if self.confounds:
            from seaborn import color_palette

            palette = color_palette('husl', nconfounds)

        for i, (name, kwargs) in enumerate(self.confounds.items()):
            tseries = kwargs.pop('values')
            confoundplot(tseries, grid[grid_id], tr=self.tr, color=palette[i], name=name, **kwargs)
            grid_id += 1

        plot_carpet(
            self.timeseries,
            segments=self.segments,
            subplot=grid[-1],
            tr=self.tr,
            sort_rows=self.sort_carpet,
            drop_trs=self.nskip,
            cmap='paired' if self.paired_carpet else None,
        )
        return figure


def plot_carpet(
    data,
    segments=None,
    cmap=None,
    tr=None,
    detrend=True,
    subplot=None,
    title=None,
    output_file=None,
    size=(900, 1200),
    sort_rows='ward',
    drop_trs=0,
    legend=True,
):
    """
    Plot an image representation of voxel intensities across time.

    This kind of plot is known as "carpet plot" or "Power plot".
    See Jonathan Power Neuroimage 2017 Jul 1; 154:150-158.

    Parameters
    ----------
    data : N x T :obj:`numpy.array`
        The functional data to be plotted (*N* sampling locations by *T* timepoints).
    segments: :obj:`dict`, optional
        A mapping between segment labels (e.g., `"Left Cortex"`) and list of indexes
        in the data array.
    cmap : colormap
        Overrides the generation of an automated colormap.
    tr : float , optional
        Specify the TR, if specified it uses this value. If left as None,
        # of frames is plotted instead of time.
    detrend : :obj:`bool`, optional
        Detrend and standardize the data prior to plotting.
    subplot : matplotlib subplot, optional
        Subplot to plot figure on.
    title : string, optional
        The title displayed on the figure.
    output_file : string, or None, optional
        The name of an image file to export the plot to. Valid extensions
        are .png, .pdf, .svg. If output_file is not None, the plot
        is saved to a file, and the display is closed.
    size : :obj:`tuple`
        Maximum number of samples to plot (voxels, timepoints)
    sort_rows : :obj:`str` or :obj:`False` or :obj:`None`
        Apply a clustering algorithm to reorganize the rows of the carpet.
        ``""``, ``False``, and ``None`` skip clustering sorting.
        ``"linkage"`` uses linkage hierarchical clustering
        :obj:`scipy.cluster.hierarchy.linkage`.
        Any other value that Python evaluates to ``True`` will use the
        default clustering, which is :obj:`sklearn.cluster.ward_tree`.

    """
    if segments is None:
        segments = {'whole brain (voxels)': list(range(data.shape[0]))}

    if len(segments) == 1:
        legend = False

    if cmap is None:
        colors = colormaps['tab10'].colors
    elif cmap == 'paired':
        colors = list(colormaps['Paired'].colors)
        colors[0], colors[1] = colors[1], colors[0]
        colors[2], colors[7] = colors[7], colors[2]

    if detrend:
        from nilearn.signal import clean

        data = clean(data.T, t_r=tr, filter=False).T

    # We want all subplots to have the same dynamic range
    vminmax = (np.percentile(data, 2), np.percentile(data, 98))

    # Decimate number of time-series before clustering
    n_dec = int((1.8 * data.shape[0]) // size[0])
    if n_dec > 1:
        segments = {
            lab: idx[::n_dec] for lab, idx in segments.items() if np.array(idx).shape >= (1,)
        }

    # Cluster segments (if argument enabled)
    if sort_rows:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from sklearn.cluster import ward_tree

        for seg_label, seg_idx in segments.items():
            # In debugging cases, we might have ROIs too small to have enough rows to sort
            if len(seg_idx) < 2:
                continue
            roi_data = data[seg_idx]
            if isinstance(sort_rows, str) and sort_rows.lower() == 'linkage':
                linkage_matrix = linkage(
                    roi_data, method='average', metric='euclidean', optimal_ordering=True
                )
            else:
                children, _, n_leaves, _, distances = ward_tree(roi_data, return_distance=True)
                linkage_matrix = _ward_to_linkage(children, n_leaves, distances)

            dn = dendrogram(linkage_matrix, no_plot=True)
            # Override the ordering of the indices in this segment
            segments[seg_label] = np.array(seg_idx)[np.array(dn['leaves'])]

    # If subplot is not defined
    if subplot is None:
        subplot = mgs.GridSpec(1, 1)[0]

    # Length before decimation
    n_trs = data.shape[-1] - drop_trs

    # Calculate time decimation factor
    t_dec = max(int((1.8 * n_trs) // size[1]), 1)
    data = data[:, drop_trs::t_dec]

    # Define nested GridSpec
    gs = mgs.GridSpecFromSubplotSpec(
        len(segments),
        1,
        subplot_spec=subplot,
        hspace=0.05,
        height_ratios=[len(v) for v in segments.values()],
    )

    for i, indices in enumerate(segments.values()):
        # Carpet plot
        ax = plt.subplot(gs[i])

        ax.imshow(
            data[indices, :],
            interpolation='nearest',
            aspect='auto',
            cmap='gray',
            vmin=vminmax[0],
            vmax=vminmax[1],
        )

        # Toggle the spine objects
        ax.spines['top'].set_color('none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_color('none')
        ax.spines['right'].set_visible(False)

        # Make colored left axis
        ax.spines['left'].set_linewidth(3)
        ax.spines['left'].set_color(colors[i])
        ax.spines['left'].set_capstyle('butt')
        ax.spines['left'].set_position(('outward', 2))

        # Make all subplots have same xticks
        xticks = np.linspace(0, data.shape[-1], endpoint=True, num=7)
        ax.set_xticks(xticks)
        ax.set_yticks([])
        ax.grid(False)

        if i == (len(segments) - 1):
            xlabel = 'time-points (index)'
            xticklabels = (xticks * n_trs / data.shape[-1]).astype('uint32') + drop_trs
            if tr is not None:
                xlabel = 'time (mm:ss)'
                xticklabels = [
                    f'{int(t // 60):02d}:{(t % 60).round(0).astype(int):02d}'
                    for t in (tr * xticklabels)
                ]

            ax.set_xlabel(xlabel)
            ax.set_xticklabels(xticklabels)
            ax.spines['bottom'].set_position(('outward', 5))
            ax.spines['bottom'].set_color('k')
            ax.spines['bottom'].set_linewidth(0.8)
        else:
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.spines['bottom'].set_color('none')
            ax.spines['bottom'].set_visible(False)

        if title and i == 0:
            ax.set_title(title)

    if len(segments) == 1:
        ax.set_ylabel(next(iter(segments)))

    if legend:
        from matplotlib.patches import Patch
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        axlegend = inset_axes(
            ax,
            width='100%',
            height=0.01,
            loc='lower center',
            borderpad=-4.1,
        )
        axlegend.grid(False)
        axlegend.set_xticks([])
        axlegend.set_yticks([])
        axlegend.patch.set_alpha(0.0)
        for loc in ('top', 'bottom', 'left', 'right'):
            axlegend.spines[loc].set_color('none')
            axlegend.spines[loc].set_visible(False)

        axlegend.legend(
            handles=[Patch(color=colors[i], label=label) for i, label in enumerate(segments)],
            loc='upper center',
            bbox_to_anchor=(0.5, 0),
            shadow=False,
            fancybox=False,
            ncol=min(len(segments.keys()), 5),
            frameon=False,
            prop={'size': 8},
        )

    if output_file is not None:
        figure = plt.gcf()
        figure.savefig(output_file, bbox_inches='tight')
        plt.close(figure)
        figure = None
        return output_file

    return gs


def spikesplot(
    ts_z,
    outer_gs=None,
    tr=None,
    zscored=True,
    spike_thresh=6.0,
    title='Spike plot',
    ax=None,
    cmap='viridis',
    hide_x=True,
    nskip=0,
):
    """
    A spikes plot. Thanks to Bob Dogherty (this docstring needs be improved with proper ack)
    """

    if ax is None:
        ax = plt.gca()

    if outer_gs is not None:
        gs = mgs.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer_gs, width_ratios=[1, 100], wspace=0.0
        )
        ax = plt.subplot(gs[1])

    # Define TR and number of frames
    if tr is None:
        tr = 1.0

    # Load timeseries, zscored slice-wise
    nslices = ts_z.shape[0]
    ntsteps = ts_z.shape[1]

    # Load a colormap
    my_cmap = colormaps[cmap]
    norm = Normalize(vmin=0, vmax=float(nslices - 1))
    colors = [my_cmap(norm(sl)) for sl in range(nslices)]

    stem = len(np.unique(ts_z).tolist()) == 2
    # Plot one line per axial slice timeseries
    for sl in range(nslices):
        if not stem:
            ax.plot(ts_z[sl, :], color=colors[sl], lw=0.5)
        else:
            markerline, stemlines, baseline = ax.stem(ts_z[sl, :])
            plt.setp(markerline, 'markerfacecolor', colors[sl])
            plt.setp(baseline, 'color', colors[sl], 'linewidth', 1)
            plt.setp(stemlines, 'color', colors[sl], 'linewidth', 1)

    # Handle X, Y axes
    ax.grid(False)

    # Handle X axis
    last = ntsteps - 1
    ax.set_xlim(0, last)
    xticks = list(range(last)[::20]) + [last] if not hide_x else []
    ax.set_xticks(xticks)

    if not hide_x:
        if tr is None:
            ax.set_xlabel('time (frame #)')
        else:
            ax.set_xlabel('time (s)')
            ax.set_xticklabels([f'{t:.2f}' for t in (tr * np.array(xticks)).tolist()])

    # Handle Y axis
    ylabel = 'slice-wise noise average on background'
    if zscored:
        ylabel += ' (z-scored)'
        zs_max = np.abs(ts_z).max()
        ax.set_ylim(
            (
                -(np.abs(ts_z[:, nskip:]).max()) * 1.05,
                (np.abs(ts_z[:, nskip:]).max()) * 1.05,
            )
        )

        ytick_vals = np.arange(0.0, zs_max, float(np.floor(zs_max / 2.0)))
        yticks = list(reversed((-1.0 * ytick_vals[ytick_vals > 0]).tolist())) + ytick_vals.tolist()

        # TODO plot min/max or mark spikes
        # yticks.insert(0, ts_z.min())
        # yticks += [ts_z.max()]
        for val in ytick_vals:
            ax.plot((0, ntsteps - 1), (-val, -val), 'k:', alpha=0.2)
            ax.plot((0, ntsteps - 1), (val, val), 'k:', alpha=0.2)

        # Plot spike threshold
        if zs_max < spike_thresh:
            ax.plot((0, ntsteps - 1), (-spike_thresh, -spike_thresh), 'k:')
            ax.plot((0, ntsteps - 1), (spike_thresh, spike_thresh), 'k:')
    else:
        yticks = [
            ts_z[:, nskip:].min(),
            np.median(ts_z[:, nskip:]),
            ts_z[:, nskip:].max(),
        ]
        ax.set_ylim(0, max(yticks[-1] * 1.05, (yticks[-1] - yticks[0]) * 2.0 + yticks[-1]))
        # ax.set_ylim(ts_z[:, nskip:].min() * 0.95,
        #             ts_z[:, nskip:].max() * 1.05)

    ax.annotate(
        ylabel,
        xy=(0.0, 0.7),
        xycoords='axes fraction',
        xytext=(0, 0),
        textcoords='offset points',
        va='center',
        ha='left',
        color='gray',
        size=4,
        bbox={
            'boxstyle': 'round',
            'fc': 'w',
            'ec': 'none',
            'color': 'none',
            'lw': 0,
            'alpha': 0.8,
        },
    )
    ax.set_yticks([])
    ax.set_yticklabels([])

    # if yticks:
    #     # ax.set_yticks(yticks)
    #     # ax.set_yticklabels(['%.02f' % y for y in yticks])
    #     # Plot maximum and minimum horizontal lines
    #     ax.plot((0, ntsteps - 1), (yticks[0], yticks[0]), 'k:')
    #     ax.plot((0, ntsteps - 1), (yticks[-1], yticks[-1]), 'k:')

    for side in ['top', 'right']:
        ax.spines[side].set_color('none')
        ax.spines[side].set_visible(False)

    if not hide_x:
        ax.spines['bottom'].set_position(('outward', 10))
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.spines['bottom'].set_color('none')
        ax.spines['bottom'].set_visible(False)

    # ax.spines["left"].set_position(('outward', 30))
    # ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_visible(False)
    ax.spines['left'].set_color(None)

    # labels = [label for label in ax.yaxis.get_ticklabels()]
    # labels[0].set_weight('bold')
    # labels[-1].set_weight('bold')
    if title:
        ax.set_title(title)
    return ax


def spikesplot_cb(position, cmap='viridis', fig=None):
    # Add colorbar
    if fig is None:
        fig = plt.gcf()

    cax = fig.add_axes(position)
    cb = ColorbarBase(
        cax,
        cmap=colormaps[cmap],
        spacing='proportional',
        orientation='horizontal',
        drawedges=False,
    )
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(['Inferior', '(axial slice)', 'Superior'])
    cb.outline.set_linewidth(0)
    cb.ax.xaxis.set_tick_params(width=0)
    return cax


def confoundplot(
    tseries,
    gs_ts,
    gs_dist=None,
    name=None,
    units=None,
    tr=None,
    hide_x=True,
    color='b',
    nskip=0,
    cutoff=None,
    ylims=None,
):
    import seaborn as sns

    # Define TR and number of frames
    notr = False
    if tr is None:
        notr = True
        tr = 1.0
    ntsteps = len(tseries)
    tseries = np.array(tseries)

    # Define nested GridSpec
    gs = mgs.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_ts, width_ratios=[1, 100], wspace=0.0)

    ax_ts = plt.subplot(gs[1])
    ax_ts.grid(False)

    # Set 10 frame markers in X axis
    interval = max((ntsteps // 10, ntsteps // 5, 1))
    xticks = list(range(ntsteps)[::interval])
    ax_ts.set_xticks(xticks)

    if not hide_x:
        if notr:
            ax_ts.set_xlabel('time (frame #)')
        else:
            ax_ts.set_xlabel('time (s)')
            labels = tr * np.array(xticks)
            ax_ts.set_xticklabels([f'{t:.2f}' for t in labels.tolist()])
    else:
        ax_ts.set_xticklabels([])

    if name is not None:
        if units is not None:
            name += f' [{units}]'

        ax_ts.annotate(
            name,
            xy=(0.0, 0.7),
            xytext=(0, 0),
            xycoords='axes fraction',
            textcoords='offset points',
            va='center',
            ha='left',
            color=color,
            size=8,
            bbox={
                'boxstyle': 'round',
                'fc': 'w',
                'ec': 'none',
                'color': 'none',
                'lw': 0,
                'alpha': 0.8,
            },
        )

    for side in ['top', 'right']:
        ax_ts.spines[side].set_color('none')
        ax_ts.spines[side].set_visible(False)

    if not hide_x:
        ax_ts.spines['bottom'].set_position(('outward', 20))
        ax_ts.xaxis.set_ticks_position('bottom')
    else:
        ax_ts.spines['bottom'].set_color('none')
        ax_ts.spines['bottom'].set_visible(False)

    # ax_ts.spines["left"].set_position(('outward', 30))
    ax_ts.spines['left'].set_color('none')
    ax_ts.spines['left'].set_visible(False)
    # ax_ts.yaxis.set_ticks_position('left')

    ax_ts.set_yticks([])
    ax_ts.set_yticklabels([])

    nonnan = tseries[~np.isnan(tseries)]
    if nonnan.size > 0:
        # Calculate Y limits
        valrange = nonnan.max() - nonnan.min()
        def_ylims = [nonnan.min() - 0.1 * valrange, nonnan.max() + 0.1 * valrange]
        if ylims is not None:
            if ylims[0] is not None:
                def_ylims[0] = min([def_ylims[0], ylims[0]])
            if ylims[1] is not None:
                def_ylims[1] = max([def_ylims[1], ylims[1]])

        # Add space for plot title and mean/SD annotation
        def_ylims[0] -= 0.1 * (def_ylims[1] - def_ylims[0])

        ax_ts.set_ylim(def_ylims)

        # Annotate stats
        maxv = nonnan.max()
        mean = nonnan.mean()
        stdv = nonnan.std()
        p95 = np.percentile(nonnan, 95.0)
    else:
        maxv = 0
        mean = 0
        stdv = 0
        p95 = 0

    stats_label = (
        r'max: {max:.3f}{units} $\bullet$ mean: {mean:.3f}{units} '
        r'$\bullet$ $\sigma$: {sigma:.3f}'
    ).format(max=maxv, mean=mean, units=units or '', sigma=stdv)
    ax_ts.annotate(
        stats_label,
        xy=(0.98, 0.7),
        xycoords='axes fraction',
        xytext=(0, 0),
        textcoords='offset points',
        va='center',
        ha='right',
        color=color,
        size=4,
        bbox={
            'boxstyle': 'round',
            'fc': 'w',
            'ec': 'none',
            'color': 'none',
            'lw': 0,
            'alpha': 0.8,
        },
    )

    # Annotate percentile 95
    ax_ts.plot((0, ntsteps - 1), [p95] * 2, linewidth=0.1, color='lightgray')
    ax_ts.annotate(
        f'{p95:.2f}',
        xy=(0, p95),
        xytext=(-1, 0),
        textcoords='offset points',
        va='center',
        ha='right',
        color='lightgray',
        size=3,
    )

    if cutoff is None:
        cutoff = []

    for thr in cutoff:
        ax_ts.plot((0, ntsteps - 1), [thr] * 2, linewidth=0.2, color='dimgray')

        ax_ts.annotate(
            f'{thr:.2f}',
            xy=(0, thr),
            xytext=(-1, 0),
            textcoords='offset points',
            va='center',
            ha='right',
            color='dimgray',
            size=3,
        )

    ax_ts.plot(tseries, color=color, linewidth=0.8)
    ax_ts.set_xlim((0, ntsteps - 1))

    if gs_dist is not None:
        ax_dist = plt.subplot(gs_dist)
        sns.displot(tseries, vertical=True, ax=ax_dist)
        ax_dist.set_xlabel('Timesteps')
        ax_dist.set_ylim(ax_ts.get_ylim())
        ax_dist.set_yticklabels([])

        return [ax_ts, ax_dist], gs
    return ax_ts, gs


def compcor_variance_plot(
    metadata_files,
    metadata_sources=None,
    output_file=None,
    varexp_thresh=(0.5, 0.7, 0.9),
    fig=None,
):
    """
    Parameters
    ----------
    metadata_files: list
        List of paths to files containing component metadata. If more than one
        decomposition has been performed (e.g., anatomical and temporal
        CompCor decompositions), then all metadata files can be provided in
        the list. However, each metadata file should have a corresponding
        entry in `metadata_sources`.
    metadata_sources: list or None
        List of source names (e.g., ['aCompCor']) for decompositions. This
        list should be of the same length as `metadata_files`.
    output_file: str or None
        Path where the output figure should be saved. If this is not defined,
        then the plotting axes will be returned instead of the saved figure
        path.
    varexp_thresh: tuple
        Set of variance thresholds to include in the plot (default 0.5, 0.7,
        0.9).
    fig: figure or None
        Existing figure on which to plot.

    Returns
    -------
    ax: axes
        Plotting axes. Returned only if the `output_file` parameter is None.
    output_file: str
        The file where the figure is saved.
    """
    metadata = {}
    if metadata_sources is None:
        if len(metadata_files) == 1:
            metadata_sources = ['CompCor']
        else:
            metadata_sources = [f'Decomposition {i:d}' for i in range(len(metadata_files))]
    for file, source in zip(metadata_files, metadata_sources):
        metadata[source] = pd.read_csv(str(file), sep=r'\s+')
        metadata[source]['source'] = source
    metadata = pd.concat(list(metadata.values()))
    bbox_txt = {
        'boxstyle': 'round',
        'fc': 'white',
        'ec': 'none',
        'color': 'none',
        'linewidth': 0,
        'alpha': 0.8,
    }

    decompositions = []
    data_sources = list(metadata.groupby(['source', 'mask']).groups.keys())
    for source, mask in data_sources:
        if not np.isnan(
            metadata.loc[(metadata['source'] == source) & (metadata['mask'] == mask)][
                'singular_value'
            ].values[0]
        ):
            decompositions.append((source, mask))

    if fig is not None:
        ax = [fig.add_subplot(1, len(decompositions), i + 1) for i in range(len(decompositions))]
    elif len(decompositions) > 1:
        fig, ax = plt.subplots(1, len(decompositions), figsize=(5 * len(decompositions), 5))
    else:
        ax = [plt.axes()]

    for m, (source, mask) in enumerate(decompositions):
        components = metadata[(metadata['mask'] == mask) & (metadata['source'] == source)]
        if len([m for s, m in decompositions if s == source]) > 1:
            title_mask = f' ({mask} mask)'
        else:
            title_mask = ''
        fig_title = f'{source}{title_mask}'

        ax[m].plot(
            np.arange(components.shape[0] + 1),
            [0] + list(100 * components['cumulative_variance_explained']),
            color='purple',
            linewidth=2.5,
        )
        ax[m].grid(False)
        ax[m].set_xlabel('number of components in model')
        ax[m].set_ylabel('cumulative variance explained (%)')
        ax[m].set_title(fig_title)

        varexp = {}

        for i, thr in enumerate(varexp_thresh):
            varexp[thr] = (
                np.atleast_1d(np.searchsorted(components['cumulative_variance_explained'], thr))
                + 1
            )
            ax[m].axhline(y=100 * thr, color='lightgrey', linewidth=0.25)
            ax[m].axvline(x=varexp[thr], color=f'C{i}', linewidth=2, linestyle=':')
            ax[m].text(
                0,
                100 * thr,
                f'{100 * thr:.0f}',
                fontsize='x-small',
                bbox=bbox_txt,
            )
            ax[m].text(
                varexp[thr][0],
                25,
                f'{varexp[thr][0]} components explain\n{100 * thr:.0f}% of variance',
                rotation=90,
                horizontalalignment='center',
                fontsize='xx-small',
                bbox=bbox_txt,
            )

        ax[m].set_yticks([])
        ax[m].set_yticklabels([])
        for label in ax[m].xaxis.get_majorticklabels():
            label.set_fontsize('x-small')
            label.set_rotation('vertical')
        for side in ['top', 'right', 'left']:
            ax[m].spines[side].set_color('none')
            ax[m].spines[side].set_visible(False)

    if output_file is not None:
        figure = plt.gcf()
        figure.savefig(output_file, bbox_inches='tight')
        plt.close(figure)
        figure = None
        return output_file
    return ax


def confounds_correlation_plot(
    confounds_file,
    columns=None,
    figure=None,
    max_dim=20,
    output_file=None,
    reference='global_signal',
    ignore_initial_volumes=0,
):
    """
    Generate a bar plot with the correlation of confounds.

    Parameters
    ----------
    confounds_file: :obj:`str`
        File containing all confound regressors to be included in the
        correlation plot.
    figure: figure or None
        Existing figure on which to plot.
    columns: :obj:`list` or :obj:`None`.
        Select a list of columns from the dataset.
    max_dim: :obj:`int`
        The maximum number of regressors to be included in the output plot.
        Reductions (e.g., CompCor) of high-dimensional data can yield so many
        regressors that the correlation structure becomes obfuscated. This
        criterion selects the ``max_dim`` regressors that have the largest
        correlation magnitude with ``reference`` for inclusion in the plot.
    output_file: :obj:`str` or :obj:`None`
        Path where the output figure should be saved. If this is not defined,
        then the plotting axes will be returned instead of the saved figure
        path.
    reference: :obj:`str`
        ``confounds_correlation_plot`` prepares a bar plot of the correlations
        of each confound regressor with a reference column. By default, this
        is the global signal (so that collinearities with the global signal
        can readily be assessed).
    ignore_initial_volumes : :obj:`int`
        Number of non-steady-state volumes at the beginning of the scan to ignore.

    Returns
    -------
    axes and gridspec
        Plotting axes and gridspec. Returned only if ``output_file`` is ``None``.
    output_file: :obj:`str`
        The file where the figure is saved.
    """
    import seaborn as sns

    confounds_data = pd.read_table(confounds_file)

    if columns:
        columns = dict.fromkeys(columns)  # Drop duplicates
        columns[reference] = None  # Make sure the reference is included
        confounds_data = confounds_data[list(columns)]

    confounds_data = confounds_data.loc[
        ignore_initial_volumes:,
        np.logical_not(np.isclose(confounds_data.var(skipna=True), 0)),
    ]
    corr = confounds_data.corr()

    gscorr = corr.copy()
    gscorr['index'] = gscorr.index
    gscorr[reference] = np.abs(gscorr[reference])
    gs_descending = gscorr.sort_values(by=reference, ascending=False)['index']
    n_vars = corr.shape[0]
    max_dim = min(n_vars, max_dim)

    gs_descending = gs_descending[:max_dim]
    features = [p for p in corr.columns if p in gs_descending]
    corr = corr.loc[features, features]

    # Modifying in-place is no longer supported
    corr_vals = corr.to_numpy(copy=True)
    np.fill_diagonal(corr_vals, 0)
    corr.iloc[:, :] = corr_vals

    if figure is None:
        plt.figure(figsize=(15, 5))
    gs = mgs.GridSpec(1, 21)
    ax0 = plt.subplot(gs[0, :10])
    ax1 = plt.subplot(gs[0, 11:])

    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, linewidths=0.5, cmap='coolwarm', center=0, square=True, ax=ax0)
    ax0.tick_params(axis='both', which='both', width=0)

    for label in ax0.xaxis.get_majorticklabels():
        label.set_fontsize('small')
    for label in ax0.yaxis.get_majorticklabels():
        label.set_fontsize('small')
    sns.barplot(
        data=gscorr,
        x='index',
        y=reference,
        ax=ax1,
        order=gs_descending,
        palette='Reds_d',
        saturation=0.5,
    )

    ax1.set_xlabel('Confound time series')
    ax1.set_ylabel(f'Magnitude of correlation with {reference}')
    ax1.tick_params(axis='x', which='both', width=0)
    ax1.tick_params(axis='y', which='both', width=5, length=5)

    for label in ax1.xaxis.get_majorticklabels():
        label.set_fontsize('small')
        label.set_rotation('vertical')
    for label in ax1.yaxis.get_majorticklabels():
        label.set_fontsize('small')
    for side in ['top', 'right', 'left']:
        ax1.spines[side].set_color('none')
        ax1.spines[side].set_visible(False)

    if output_file is not None:
        figure = plt.gcf()
        figure.savefig(output_file, bbox_inches='tight')
        plt.close(figure)
        figure = None
        return output_file
    return [ax0, ax1], gs


def cifti_surfaces_plot(
    in_cifti,
    density='32k',
    surface_type='inflated',
    clip_range=(0, None),
    output_file=None,
    **kwargs,
):
    """
    Plots a CIFTI-2 dense timeseries onto left/right mesh surfaces.

    Parameters
    ----------
    in_cifti : str
        CIFTI-2 dense timeseries (.dtseries.nii)
    density : str
        Surface density
    surface_type : str
        Inflation level of mesh surfaces. Supported: midthickness, inflated, veryinflated
    clip_range : tuple or None
        Range to clip `in_cifti` data prior to plotting.
        If not None, two values must be provided as lower and upper bounds.
        If values are None, no clipping is performed for that bound.
    output_file: :obj:`str` or :obj:`None`
        Path where the output figure should be saved. If this is not defined,
        then the figure will be returned.
    kwargs : dict
        Keyword arguments for :obj:`nilearn.plotting.plot_surf`

    Outputs
    -------
    figure : matplotlib.pyplot.figure
        Surface plot figure. Returned only if ``output_file`` is ``None``.
    output_file: :obj:`str`
        The file where the figure is saved.
    """
    from nilearn.plotting import plot_surf

    def get_surface_meshes(density, surface_type):
        import templateflow.api as tf

        lh, rh = tf.get('fsLR', density=density, suffix=surface_type, extension=['.surf.gii'])
        return str(lh), str(rh)

    if density != '32k':
        raise NotImplementedError('Only 32k density is currently supported.')

    img = nb.cifti2.load(in_cifti)
    if img.nifti_header.get_intent()[0] != 'ConnDenseSeries':
        raise TypeError(f'{in_cifti} is not a dense timeseries CIFTI file')

    geo = img.header.get_index_map(1)
    left_cortex, right_cortex = None, None
    for bm in geo.brain_models:
        if bm.brain_structure == 'CIFTI_STRUCTURE_CORTEX_LEFT':
            left_cortex = bm
        elif bm.brain_structure == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
            right_cortex = bm

    if left_cortex is None or right_cortex is None:
        raise RuntimeError('CIFTI is missing cortex information')

    # calculate an average of the BOLD data, excluding the first 5 volumes
    # as potential nonsteady states
    data = img.dataobj[5:20].mean(axis=0)

    counts = (left_cortex.index_count, right_cortex.index_count)
    if density == '32k' and counts != (29696, 29716):
        raise ValueError('Cortex data is not in fsLR space')

    # medial wall needs to be added back in
    lh_data = np.full(left_cortex.surface_number_of_vertices, np.nan)
    rh_data = np.full(right_cortex.surface_number_of_vertices, np.nan)
    lh_data[left_cortex.vertex_indices] = _concat_brain_struct_data([left_cortex], data)
    rh_data[right_cortex.vertex_indices] = _concat_brain_struct_data([right_cortex], data)

    if clip_range:
        lh_data = np.clip(lh_data, clip_range[0], clip_range[1], out=lh_data)
        rh_data = np.clip(rh_data, clip_range[0], clip_range[1], out=rh_data)
        mn, mx = clip_range
    else:
        mn, mx = None, None

    if mn is None:
        mn = np.min(data)
    if mx is None:
        mx = np.max(data)

    cmap = kwargs.pop('cmap', 'YlOrRd_r')
    cbar_map = cm.ScalarMappable(norm=Normalize(mn, mx), cmap=cmap)

    # Make background maps that rescale to a medium gray
    lh_bg = np.zeros(lh_data.shape, 'int8')
    rh_bg = np.zeros(rh_data.shape, 'int8')
    lh_bg[:2] = [3, -2]
    rh_bg[:2] = [3, -2]

    lh_mesh, rh_mesh = get_surface_meshes(density, surface_type)
    lh_kwargs = {'surf_mesh': lh_mesh, 'surf_map': lh_data, 'bg_map': lh_bg}
    rh_kwargs = {'surf_mesh': rh_mesh, 'surf_map': rh_data, 'bg_map': rh_bg}

    # Build the figure
    figure = plt.figure(figsize=plt.figaspect(0.25), constrained_layout=True)
    for i, view in enumerate(('lateral', 'medial')):
        for j, hemi in enumerate(('left', 'right')):
            title = f'{hemi.title()} - {view.title()}'
            ax = figure.add_subplot(1, 4, i * 2 + j + 1, projection='3d', rasterized=True)
            hemi_kwargs = (lh_kwargs, rh_kwargs)[j]
            plot_surf(
                hemi=hemi,
                view=view,
                title=title,
                cmap=cmap,
                vmin=mn,
                vmax=mx,
                axes=ax,
                **hemi_kwargs,
                **kwargs,
            )
            # plot_surf sets this to 8, which seems a little far out, but 6 starts clipping
            ax.dist = 7

    figure.colorbar(cbar_map, shrink=0.2, ax=figure.axes, location='bottom')

    if output_file is not None:
        figure.savefig(output_file, bbox_inches='tight', dpi=400)
        plt.close(figure)
        return output_file

    return figure


def _concat_brain_struct_data(structs, data):
    concat_data = np.array([], dtype=data.dtype)
    for struct in structs:
        struct_upper_bound = struct.index_offset + struct.index_count
        struct_data = data[struct.index_offset : struct_upper_bound]
        concat_data = np.concatenate((concat_data, struct_data))
    return concat_data


def _ward_to_linkage(children, n_leaves, distances):
    """Create linkage matrix from the output of Ward clustering."""
    # create the counts of samples under each node
    counts = np.zeros(children.shape[0])
    n_samples = n_leaves
    for i, merge in enumerate(children):
        current_count = 0
        for child_idx in merge:
            current_count += 1 if child_idx < n_samples else counts[child_idx - n_samples]
        counts[i] = current_count

    return np.column_stack([children, distances, counts]).astype(float)
