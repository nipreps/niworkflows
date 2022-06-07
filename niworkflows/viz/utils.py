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
"""Helper tools for visualization purposes."""
from pathlib import Path
from shutil import which
from tempfile import TemporaryDirectory
import subprocess
import base64
import re
from uuid import uuid4
from io import StringIO

import numpy as np
import nibabel as nb

from nipype.utils import filemanip
from .. import NIWORKFLOWS_LOG
from ..utils.images import rotation2canonical, rotate_affine


SVGNS = "http://www.w3.org/2000/svg"


def robust_set_limits(data, plot_params, percentiles=(15, 99.8)):
    """Set (vmax, vmin) based on percentiles of the data."""
    plot_params["vmin"] = plot_params.get("vmin", np.percentile(data, percentiles[0]))
    plot_params["vmax"] = plot_params.get("vmax", np.percentile(data, percentiles[1]))
    return plot_params


def svg_compress(image, compress="auto"):
    """Generate a blob SVG from a matplotlib figure, may perform compression."""
    # Check availability of svgo and cwebp
    has_compress = all((which("svgo"), which("cwebp")))
    if compress is True and not has_compress:
        raise RuntimeError(
            "Compression is required, but svgo or cwebp are not installed"
        )
    else:
        compress = (compress is True or compress == "auto") and has_compress

    # Compress the SVG file using SVGO
    if compress:
        cmd = "svgo -i - -o - -q -p 3 --pretty"
        try:
            pout = subprocess.run(
                cmd,
                input=image.encode("utf-8"),
                stdout=subprocess.PIPE,
                shell=True,
                check=True,
                close_fds=True,
            ).stdout
        except OSError as e:
            from errno import ENOENT

            if compress is True and e.errno == ENOENT:
                raise e
        else:
            image = pout.decode("utf-8")

    # Convert all of the rasters inside the SVG file with 80% compressed WEBP
    if compress:
        new_lines = []
        with StringIO(image) as fp:
            for line in fp:
                if "image/png" in line:
                    tmp_lines = [line]
                    while "/>" not in line:
                        line = fp.readline()
                        tmp_lines.append(line)
                    content = "".join(tmp_lines).replace("\n", "").replace(",  ", ",")

                    left = content.split("base64,")[0] + "base64,"
                    left = left.replace("image/png", "image/webp")
                    right = content.split("base64,")[1]
                    png_b64 = right.split('"')[0]
                    right = '"' + '"'.join(right.split('"')[1:])

                    cmd = "cwebp -quiet -noalpha -q 80 -o - -- -"
                    pout = subprocess.run(
                        cmd,
                        input=base64.b64decode(png_b64),
                        shell=True,
                        stdout=subprocess.PIPE,
                        check=True,
                        close_fds=True,
                    ).stdout
                    webpimg = base64.b64encode(pout).decode("utf-8")
                    new_lines.append(left + webpimg + right)
                else:
                    new_lines.append(line)
        lines = new_lines
    else:
        lines = image.splitlines()

    svg_start = 0
    for i, line in enumerate(lines):
        if "<svg " in line:
            svg_start = i
            continue

    image_svg = lines[svg_start:]  # strip out extra DOCTYPE, etc headers
    return "".join(image_svg)  # straight up giant string


def svg2str(display_object, dpi=300):
    """Serialize a nilearn display object to string."""
    from io import StringIO

    image_buf = StringIO()
    display_object.frame_axes.figure.savefig(
        image_buf, dpi=dpi, format="svg", facecolor="k", edgecolor="k"
    )
    return image_buf.getvalue()


def extract_svg(display_object, dpi=300, compress="auto"):
    """Remove the preamble of the svg files generated with nilearn."""
    image_svg = svg2str(display_object, dpi)
    if compress is True or compress == "auto":
        image_svg = svg_compress(image_svg, compress)
    image_svg = re.sub(' height="[0-9]+[a-z]*"', "", image_svg, count=1)
    image_svg = re.sub(' width="[0-9]+[a-z]*"', "", image_svg, count=1)
    image_svg = re.sub(
        " viewBox", ' preseveAspectRation="xMidYMid meet" viewBox', image_svg, count=1
    )
    start_tag = "<svg "
    start_idx = image_svg.find(start_tag)
    end_tag = "</svg>"
    end_idx = image_svg.rfind(end_tag)
    if start_idx == -1 or end_idx == -1:
        NIWORKFLOWS_LOG.info("svg tags not found in extract_svg")
    # rfind gives the start index of the substr. We want this substr
    # included in our return value so we add its length to the index.
    end_idx += len(end_tag)
    return image_svg[start_idx:end_idx]


def cuts_from_bbox(mask_nii, cuts=3):
    """Find equi-spaced cuts for presenting images."""
    mask_data = np.asanyarray(mask_nii.dataobj) > 0.0

    # First, project the number of masked voxels on each axes
    ijk_counts = [
        mask_data.sum(2).sum(1),  # project sagittal planes to transverse (i) axis
        mask_data.sum(2).sum(0),  # project coronal planes to to longitudinal (j) axis
        mask_data.sum(1).sum(0),  # project axial planes to vertical (k) axis
    ]

    # If all voxels are masked in a slice (say that happens at k=10),
    # then the value for ijk_counts for the projection to k (ie. ijk_counts[2])
    # at that element of the orthogonal axes (ijk_counts[2][10]) is
    # the total number of voxels in that slice (ie. Ni x Nj).
    # Here we define some thresholds to consider the plane as "masked"
    # The thresholds vary because of the shape of the brain
    # I have manually found that for the axial view requiring 30%
    # of the slice elements to be masked drops almost empty boxes
    # in the mosaic of axial planes (and also addresses #281)
    ijk_th = np.ceil([
        (mask_data.shape[1] * mask_data.shape[2]) * 0.2,  # sagittal
        (mask_data.shape[0] * mask_data.shape[2]) * 0.1,  # coronal
        (mask_data.shape[0] * mask_data.shape[1]) * 0.3,  # axial
    ]).astype(int)

    vox_coords = np.zeros((4, cuts), dtype=np.float32)
    vox_coords[-1, :] = 1.0
    for ax, (c, th) in enumerate(zip(ijk_counts, ijk_th)):
        # Start with full plane if mask is seemingly empty
        smin, smax = (0, mask_data.shape[ax] - 1)

        B = np.argwhere(c > th)
        if B.size < cuts:  # Threshold too high
            B = np.argwhere(c > 0)
        if B.size:
            smin, smax = B.min(), B.max()

        vox_coords[ax, :] = np.linspace(smin, smax, num=cuts + 2)[1:-1]

    ras_coords = mask_nii.affine.dot(vox_coords)[:3, ...]
    return {k: list(v) for k, v in zip(["x", "y", "z"], np.around(ras_coords, 3))}


def _3d_in_file(in_file):
    """ if self.inputs.in_file is 3d, return it.
    if 4d, pick an arbitrary volume and return that.

    if in_file is a list of files, return an arbitrary file from
    the list, and an arbitrary volume from that file
    """
    from nilearn import image as nlimage

    in_file = filemanip.filename_to_list(in_file)[0]

    try:
        in_file = nb.load(in_file)
    except AttributeError:
        in_file = in_file

    if len(in_file.shape) == 3:
        return in_file

    return nlimage.index_img(in_file, 0)


def plot_segs(
    image_nii,
    seg_niis,
    out_file,
    bbox_nii=None,
    masked=False,
    colors=None,
    compress="auto",
    **plot_params
):
    """
    Generate a static mosaic with ROIs represented by their delimiting contour.

    Plot segmentation as contours over the image (e.g. anatomical).
    seg_niis should be a list of files. mask_nii helps determine the cut
    coordinates. plot_params will be passed on to nilearn plot_* functions. If
    seg_niis is a list of size one, it behaves as if it was plotting the mask.
    """
    from svgutils.transform import fromstring
    from nilearn import image as nlimage

    plot_params = {} if plot_params is None else plot_params

    image_nii = _3d_in_file(image_nii)
    canonical_r = rotation2canonical(image_nii)
    image_nii = rotate_affine(image_nii, rot=canonical_r)
    seg_niis = [rotate_affine(_3d_in_file(f), rot=canonical_r) for f in seg_niis]
    data = image_nii.get_fdata()

    plot_params = robust_set_limits(data, plot_params)

    bbox_nii = (
        image_nii if bbox_nii is None
        else rotate_affine(_3d_in_file(bbox_nii), rot=canonical_r)
    )

    if masked:
        bbox_nii = nlimage.threshold_img(bbox_nii, 1e-3)

    cuts = cuts_from_bbox(bbox_nii, cuts=7)
    plot_params["colors"] = colors or plot_params.get("colors", None)
    out_files = []
    for d in plot_params.pop("dimensions", ("z", "x", "y")):
        plot_params["display_mode"] = d
        plot_params["cut_coords"] = cuts[d]
        svg = _plot_anat_with_contours(
            image_nii, segs=seg_niis, compress=compress, **plot_params
        )
        # Find and replace the figure_1 id.
        svg = svg.replace("figure_1", "segmentation-%s-%s" % (d, uuid4()), 1)
        out_files.append(fromstring(svg))

    return out_files


def _plot_anat_with_contours(image, segs=None, compress="auto", **plot_params):
    from nilearn.plotting import plot_anat

    nsegs = len(segs or [])
    plot_params = plot_params or {}
    # plot_params' values can be None, however they MUST NOT
    # be None for colors and levels from this point on.
    colors = plot_params.pop("colors", None) or []
    levels = plot_params.pop("levels", None) or []
    missing = nsegs - len(colors)
    if missing > 0:  # missing may be negative
        from seaborn import color_palette

        colors = colors + color_palette("husl", missing)

    colors = [[c] if not isinstance(c, list) else c for c in colors]

    if not levels:
        levels = [[0.5]] * nsegs

    # anatomical
    display = plot_anat(image, **plot_params)

    # remove plot_anat -specific parameters
    plot_params.pop("display_mode")
    plot_params.pop("cut_coords")

    plot_params["linewidths"] = 0.5
    for i in reversed(range(nsegs)):
        plot_params["colors"] = colors[i]
        display.add_contours(segs[i], levels=levels[i], **plot_params)

    svg = extract_svg(display, compress=compress)
    display.close()
    return svg


def plot_registration(
    anat_nii,
    div_id,
    plot_params=None,
    order=("z", "x", "y"),
    cuts=None,
    estimate_brightness=False,
    label=None,
    contour=None,
    compress="auto",
    dismiss_affine=False,
):
    """
    Plots the foreground and background views
    Default order is: axial, coronal, sagittal
    """
    from svgutils.transform import fromstring
    from nilearn.plotting import plot_anat
    from nilearn import image as nlimage

    plot_params = {} if plot_params is None else plot_params

    # Use default MNI cuts if none defined
    if cuts is None:
        raise NotImplementedError  # TODO

    out_files = []
    if estimate_brightness:
        plot_params = robust_set_limits(anat_nii.get_fdata().reshape(-1), plot_params)

    # FreeSurfer ribbon.mgz
    ribbon = contour is not None and np.array_equal(
        np.unique(contour.get_fdata()), [0, 2, 3, 41, 42]
    )

    if ribbon:
        contour_data = contour.get_fdata() % 39
        white = nlimage.new_img_like(contour, contour_data == 2)
        pial = nlimage.new_img_like(contour, contour_data >= 2)

    if dismiss_affine:
        canonical_r = rotation2canonical(anat_nii)
        anat_nii = rotate_affine(anat_nii, rot=canonical_r)
        if ribbon:
            white = rotate_affine(white, rot=canonical_r)
            pial = rotate_affine(pial, rot=canonical_r)
        if contour:
            contour = rotate_affine(contour, rot=canonical_r)

    # Plot each cut axis
    for i, mode in enumerate(list(order)):
        plot_params["display_mode"] = mode
        plot_params["cut_coords"] = cuts[mode]
        if i == 0:
            plot_params["title"] = label
        else:
            plot_params["title"] = None

        # Generate nilearn figure
        display = plot_anat(anat_nii, **plot_params)
        if ribbon:
            kwargs = {"levels": [0.5], "linewidths": 0.5}
            display.add_contours(white, colors="b", **kwargs)
            display.add_contours(pial, colors="r", **kwargs)
        elif contour is not None:
            display.add_contours(contour, colors="r", levels=[0.5], linewidths=0.5)

        svg = extract_svg(display, compress=compress)
        display.close()

        # Find and replace the figure_1 id.
        svg = svg.replace("figure_1", "%s-%s-%s" % (div_id, mode, uuid4()), 1)
        out_files.append(fromstring(svg))

    return out_files


def compose_view(bg_svgs, fg_svgs, ref=0, out_file="report.svg"):
    """Compose the input svgs into one standalone svg with CSS flickering animation."""
    out_file = Path(out_file).absolute()
    out_file.write_text("\n".join(_compose_view(bg_svgs, fg_svgs, ref=ref)))
    return str(out_file)


def _compose_view(bg_svgs, fg_svgs, ref=0):
    from svgutils.compose import Unit
    from svgutils.transform import SVGFigure, GroupElement

    if fg_svgs is None:
        fg_svgs = []

    # Merge SVGs and get roots
    svgs = bg_svgs + fg_svgs
    roots = [f.getroot() for f in svgs]

    # Query the size of each
    sizes = []
    for f in svgs:
        viewbox = [float(v) for v in f.root.get("viewBox").split(" ")]
        width = int(viewbox[2])
        height = int(viewbox[3])
        sizes.append((width, height))
    nsvgs = len(bg_svgs)

    sizes = np.array(sizes)

    # Calculate the scale to fit all widths
    width = sizes[ref, 0]
    scales = width / sizes[:, 0]
    heights = sizes[:, 1] * scales

    # Compose the views panel: total size is the width of
    # any element (used the first here) and the sum of heights
    fig = SVGFigure(Unit(f"{width}px"), Unit(f"{heights[:nsvgs].sum()}px"))

    yoffset = 0
    for i, r in enumerate(roots):
        r.moveto(0, yoffset, scale_x=scales[i])
        if i == (nsvgs - 1):
            yoffset = 0
        else:
            yoffset += heights[i]

    # Group background and foreground panels in two groups
    if fg_svgs:
        newroots = [
            GroupElement(roots[:nsvgs], {"class": "background-svg"}),
            GroupElement(roots[nsvgs:], {"class": "foreground-svg"}),
        ]
    else:
        newroots = roots
    fig.append(newroots)
    fig.root.attrib.pop("width", None)
    fig.root.attrib.pop("height", None)
    fig.root.set("preserveAspectRatio", "xMidYMid meet")

    with TemporaryDirectory() as tmpdirname:
        out_file = Path(tmpdirname) / "tmp.svg"
        fig.save(str(out_file))
        # Post processing
        svg = out_file.read_text().splitlines()

    # Remove <?xml... line
    if svg[0].startswith("<?xml"):
        svg = svg[1:]

    # Add styles for the flicker animation
    if fg_svgs:
        svg.insert(
            2,
            """\
<style type="text/css">
@keyframes flickerAnimation%s { 0%% {opacity: 1;} 100%% { opacity: 0; }}
.foreground-svg { animation: 1s ease-in-out 0s alternate none infinite paused flickerAnimation%s;}
.foreground-svg:hover { animation-play-state: running;}
</style>"""
            % tuple([uuid4()] * 2),
        )

    return svg


def transform_to_2d(data, max_axis):
    """
    Projects 3d data cube along one axis using maximum intensity with
    preservation of the signs. Adapted from nilearn.
    """
    import numpy as np

    # get the shape of the array we are projecting to
    new_shape = list(data.shape)
    del new_shape[max_axis]

    # generate a 3D indexing array that points to max abs value in the
    # current projection
    a1, a2 = np.indices(new_shape)
    inds = [a1, a2]
    inds.insert(max_axis, np.abs(data).argmax(axis=max_axis))

    # take the values where the absolute value of the projection
    # is the highest
    maximum_intensity_data = data[tuple(inds)]

    return np.rot90(maximum_intensity_data)


def plot_melodic_components(
    melodic_dir,
    in_file,
    tr=None,
    out_file="melodic_reportlet.svg",
    compress="auto",
    report_mask=None,
    noise_components_file=None,
):
    """
    Plots the spatiotemporal components extracted by FSL MELODIC
    from functional MRI data.

    Parameters
    ----------
    melodic_dir : str
        Path pointing to the outputs of MELODIC
    in_file :  str
        Path pointing to the reference fMRI dataset. This file
        will be used to extract the TR value, if the ``tr`` argument
        is not set. This file will be used to calculate a mask
        if ``report_mask`` is not provided.
    tr : float
        Repetition time in seconds
    out_file : str
        Path where the resulting SVG file will be stored
    compress : ``'auto'`` or bool
        Whether SVG should be compressed. If ``'auto'``, compression
        will be executed if dependencies are installed (SVGO)
    report_mask : str
        Path to a brain mask corresponding to ``in_file``
    noise_components_file : str
        A CSV file listing the indexes of components classified as noise
        by some manual or automated (e.g. ICA-AROMA) procedure. If a
        ``noise_components_file`` is provided, then components will be
        plotted with red/green colors (correspondingly to whether they
        are in the file -noise components, red-, or not -signal, green-).
        When all or none of the components are in the file, a warning
        is printed at the top.

    """
    from nilearn.image import index_img, iter_img
    import nibabel as nb
    import numpy as np
    import pylab as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    import os

    sns.set_style("white")
    current_palette = sns.color_palette()
    in_nii = nb.load(in_file)
    if not tr:
        tr = in_nii.header.get_zooms()[3]
        units = in_nii.header.get_xyzt_units()
        if units:
            if units[-1] == "msec":
                tr = tr / 1000.0
            elif units[-1] == "usec":
                tr = tr / 1000000.0
            elif units[-1] != "sec":
                NIWORKFLOWS_LOG.warning(
                    "Unknown repetition time units " "specified - assuming seconds"
                )
        else:
            NIWORKFLOWS_LOG.warning(
                "Repetition time units not specified - assuming seconds"
            )

    from nilearn.input_data import NiftiMasker
    from nilearn.plotting import cm

    if not report_mask:
        nifti_masker = NiftiMasker(mask_strategy="epi")
        nifti_masker.fit(index_img(in_nii, range(2)))
        mask_img = nifti_masker.mask_img_
    else:
        mask_img = nb.load(report_mask)

    mask_sl = []
    for j in range(3):
        mask_sl.append(transform_to_2d(mask_img.get_fdata(), j))

    timeseries = np.loadtxt(os.path.join(melodic_dir, "melodic_mix"))
    power = np.loadtxt(os.path.join(melodic_dir, "melodic_FTmix"))
    stats = np.loadtxt(os.path.join(melodic_dir, "melodic_ICstats"))
    n_components = stats.shape[0]
    Fs = 1.0 / tr
    Ny = Fs / 2
    f = Ny * (np.array(list(range(1, power.shape[0] + 1)))) / (power.shape[0])

    # Set default colors
    color_title = "k"
    color_time = current_palette[0]
    color_power = current_palette[1]
    classified_colors = None

    warning_row = 0  # Do not allocate warning row
    # Only if the components file has been provided, a warning banner will
    # be issued if all or none of the components were classified as noise
    if noise_components_file:
        noise_components = np.loadtxt(
            noise_components_file, dtype=int, delimiter=",", ndmin=1
        )
        # Activate warning row if pertinent
        warning_row = int(noise_components.size in (0, n_components))
        classified_colors = {True: "r", False: "g"}

    n_rows = int((n_components + (n_components % 2)) / 2)
    fig = plt.figure(figsize=(6.5 * 1.5, (n_rows + warning_row) * 0.85))
    gs = GridSpec(
        n_rows * 2 + warning_row,
        9,
        width_ratios=[1, 1, 1, 4, 0.001, 1, 1, 1, 4],
        height_ratios=[5] * warning_row + [1.1, 1] * n_rows,
    )

    if warning_row:
        ax = fig.add_subplot(gs[0, :])
        ncomps = "NONE of the"
        if noise_components.size == n_components:
            ncomps = "ALL"
        ax.annotate(
            "WARNING: {} components were classified as noise".format(ncomps),
            xy=(0.0, 0.5),
            xycoords="axes fraction",
            xytext=(0.01, 0.5),
            textcoords="axes fraction",
            size=12,
            color="#ea8800",
            bbox=dict(boxstyle="round", fc="#f7dcb7", ec="#FC990E"),
        )
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    titlefmt = "C{id:d}{noise}: Tot. var. expl. {var:.2g}%".format
    for i, img in enumerate(iter_img(os.path.join(melodic_dir, "melodic_IC.nii.gz"))):

        col = i % 2
        row = i // 2
        l_row = row * 2 + warning_row
        is_noise = False

        if classified_colors:
            # If a noise components list is provided, assign red/green
            is_noise = (i + 1) in noise_components
            color_title = color_time = color_power = classified_colors[is_noise]

        data = img.get_fdata()
        for j in range(3):
            ax1 = fig.add_subplot(gs[l_row:l_row + 2, j + col * 5])
            sl = transform_to_2d(data, j)
            m = np.abs(sl).max()
            ax1.imshow(
                sl, vmin=-m, vmax=+m, cmap=cm.cold_white_hot, interpolation="nearest"
            )
            ax1.contour(mask_sl[j], levels=[0.5], colors="k", linewidths=0.5)
            plt.axis("off")
            ax1.autoscale_view("tight")
            if j == 0:
                ax1.set_title(
                    titlefmt(id=i + 1, noise=" [noise]" * is_noise, var=stats[i, 1]),
                    x=0,
                    y=1.18,
                    fontsize=7,
                    horizontalalignment="left",
                    verticalalignment="top",
                    color=color_title,
                )

        ax2 = fig.add_subplot(gs[l_row, 3 + col * 5])
        ax3 = fig.add_subplot(gs[l_row + 1, 3 + col * 5])

        ax2.plot(
            np.arange(len(timeseries[:, i])) * tr,
            timeseries[:, i],
            linewidth=min(200 / len(timeseries[:, i]), 1.0),
            color=color_time,
        )
        ax2.set_xlim([0, len(timeseries[:, i]) * tr])
        ax2.axes.get_yaxis().set_visible(False)
        ax2.autoscale_view("tight")
        ax2.tick_params(axis="both", which="major", pad=0)
        sns.despine(left=True, bottom=True)
        for tick in ax2.xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
            tick.label.set_color(color_time)

        ax3.plot(
            f[0:],
            power[0:, i],
            color=color_power,
            linewidth=min(100 / len(power[0:, i]), 1.0),
        )
        ax3.set_xlim([f[0], f.max()])
        ax3.axes.get_yaxis().set_visible(False)
        ax3.autoscale_view("tight")
        ax3.tick_params(axis="both", which="major", pad=0)
        for tick in ax3.xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
            tick.label.set_color(color_power)
        sns.despine(left=True, bottom=True)

    plt.subplots_adjust(hspace=0.5)
    fig.savefig(
        out_file,
        dpi=300,
        format="svg",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.01,
    )
    fig.clf()
