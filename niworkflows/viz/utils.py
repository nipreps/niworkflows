# -*- coding: utf-8 -*-
"""Helper tools for visualization purposes"""
from __future__ import absolute_import, division, print_function
import os.path as op
from sys import version_info

from uuid import uuid4
import numpy as np
from lxml import etree
import nibabel as nb
from nilearn.plotting import plot_anat, find_cut_slices
from nilearn import image as nlimage

from io import open
import jinja2
from pkg_resources import resource_filename as pkgrf

from nipype.utils import filemanip

SVGNS = "http://www.w3.org/2000/svg"
PY3 = version_info[0] > 2

def save_html(template, report_file_name, unique_string, **kwargs):
    ''' save an actual html file with name report_file_name. unique_string is
    used to uniquely identify the html/css/js/etc generated for this report. For
    limitations on unique_string, check
    http://stackoverflow.com/questions/70579/what-are-valid-values-for-the-id-attribute-in-html '''

    searchpath = pkgrf('niworkflows', '/')
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(searchpath=searchpath),
        trim_blocks=True, lstrip_blocks=True
    )
    report_tpl = env.get_template('viz/' + template)
    kwargs['unique_string'] = unique_string
    report_render = report_tpl.render(kwargs)

    with open(report_file_name, 'w' if PY3 else 'wb') as handle:
        handle.write(report_render)

def as_svg(image):
    ''' takes an image as created by nilearn.plotting and returns a blob svg.
    A bit hacky. '''
    filename = 'temp.svg'

    image.savefig(filename)
    with open(filename, 'r' if PY3 else 'rb') as file_obj:
        image_svg = file_obj.readlines()

    svg_start = 0
    for i, line in enumerate(image_svg):
        if '<svg ' in line:
            svg_start = i
            continue

    image_svg = image_svg[svg_start:] # strip out extra DOCTYPE, etc headers
    return '\n'.join(image_svg) # straight up giant string

def _3d_in_file(in_file):
    ''' if self.inputs.in_file is 3d, return it.
    if 4d, pick an arbitrary volume and return that.

    if in_file is a list of files, return an arbitrary file from
    the list, and an arbitrary volume from that file
    '''

    in_file = filemanip.filename_to_list(in_file)[0]

    try:
        in_file = nb.load(in_file)
    except AttributeError:
        in_file = in_file

    if in_file.get_data().ndim == 3:
        return in_file

    return nlimage.index_img(in_file, 0)

def plot_segs(image_nii, seg_niis, mask_nii, out_file, masked=False, ifinputs=None, ifoutputs=None, title=None, **plot_params):
    """ plot segmentation as contours over the image (e.g. anatomical). seg_niis should be a list of files. mask_nii helps determine the cut coordinates. plot_params will be passed on to nilearn plot_* functions """

    def _plot_anat_with_contours(image, segs=None, **plot_params):
        assert not segs is None
        assert len(segs) <= 3
        plot_params = {} if plot_params is None else plot_params

        # anatomical
        plot_params['alpha'] = .7
        svg = plot_anat(image, **plot_params)

        # segment contours
        for seg, color in zip(segs, ['b', 'r', 'y']):
            plot_params['colors'] = color
            plot_params['levels'] = [0.5] if not 'levels' in plot_params else plot_params['levels']
            plot_params['alpha'] = 1
            svg.add_contours(seg, **plot_params)

        svgs_list.append(as_svg(svg))

    image_nii = _3d_in_file(image_nii)
    seg_niis = filemanip.filename_to_list(seg_niis)
    mask_nii = nb.load(mask_nii) if masked else nlimage.threshold_img(mask_nii, 1e-3)

    cuts = {k: find_cut_slices(mask_nii, direction=k, n_cuts=3) for k in ['x', 'y', 'z']}

    svgs_list = []
    plot_xyz(image_nii, _plot_anat_with_contours, cuts, segs=seg_niis)

    save_html(template='segmentation.tpl',
              report_file_name=out_file,
              unique_string='seg' + str(uuid4()),
              base_image='<br />'.join(svgs_list),
              title=title)

def plot_xyz(image, plot_func, cuts, plot_params=None, dimensions=['z', 'x', 'y'], **kwargs):
    """
    plot_func must be a function that more-or-less conforms to nilearn's plot_* signature
    """
    plot_params = {} if plot_params is None else plot_params

    for dimension in dimensions:
        plot_params['display_mode'] = dimension
        plot_params['cut_coords'] = cuts[dimension]
        plot_func(image, **plot_params, **kwargs)

def plot_registration(anat_img, div_id, plot_params=None,
                      order=('z', 'x', 'y'), cuts=None,
                      estimate_brightness=False):
    """
    Plots the foreground and background views
    Default order is: axial, coronal, sagittal
    """

    plot_params = {} if plot_params is None else plot_params

    # Use default MNI cuts if none defined
    if cuts is None:
        raise NotImplementedError  # TODO

    out_files = []
    if estimate_brightness:
        from nibabel import load as loadnii
        data = loadnii(anat_img).get_data().reshape(-1)
        vmin = np.percentile(data, 15)
        if plot_params.get('vmin', None) is None:
            plot_params['vmin'] = vmin
        if plot_params.get('vmax', None) is None:
            plot_params['vmax'] = np.percentile(data[data > vmin], 99.8)

    # Plot each cut axis
    for mode in list(order):
        out_file = '{}_{}.svg'.format(div_id, mode)
        plot_params['display_mode'] = mode
        plot_params['cut_coords'] = cuts[mode]
        plot_params['output_file'] = out_file

        # Generate nilearn figure
        plot_anat(anat_img, **plot_params)
        out_files.append(out_file)

        # Open generated svg file and fix id
        with open(out_file, 'rb') as f:
            svg = f.read()

        # Find and replace the figure_1 id.
        xml_data = etree.fromstring(svg)
        find_text = etree.ETXPath("//{%s}g[@id='figure_1']" % (SVGNS))
        find_text(xml_data)[0].set('id', '%s-%s-%s' % (div_id, mode, uuid4()))

        with open(out_file, 'wb') as f:
            f.write(etree.tostring(xml_data))
    return out_files


def compose_view(bg_svgs, fg_svgs, ref=0, out_file='report.svg'):
    """
    Composes the input svgs into one standalone svg and inserts
    the CSS code for the flickering animation
    """
    import svgutils.transform as svgt
    import svgutils.compose as svgc

    # Read all svg files and get roots
    svgs = [svgt.fromfile(f) for f in bg_svgs + fg_svgs]
    roots = [f.getroot() for f in svgs]
    nsvgs = len(svgs) // 2
    # Query the size of each
    sizes = [(int(f.width[:-2]), int(f.height[:-2])) for f in svgs]

    # Calculate the scale to fit all widths
    scales = [1.0] * len(svgs)
    if not all([width[0] == sizes[0][0] for width in sizes[1:]]):
        ref_size = sizes[ref]
        for i, els in enumerate(sizes):
            scales[i] = ref_size[0]/els[0]

    newsizes = [tuple(size)
                for size in np.array(sizes) * np.array(scales)[..., np.newaxis]]

    # Compose the views panel: total size is the width of
    # any element (used the first here) and the sum of heights
    totalsize = [newsizes[0][0], np.sum(newsizes[:3], axis=0)[1]]
    fig = svgt.SVGFigure(totalsize[0], totalsize[1])

    yoffset = 0
    for i, r in enumerate(roots):
        size = newsizes[i]
        r.moveto(0, yoffset, scale=scales[i])
        yoffset += size[1]
        if i == (nsvgs - 1):
            yoffset = 0

    # Group background and foreground panels in two groups
    newroots = [
        svgt.GroupElement(roots[:3], {'class': 'background-svg'}),
        svgt.GroupElement(roots[3:], {'class': 'foreground-svg'})
    ]
    fig.append(newroots)
    out_file = op.abspath(out_file)
    fig.save(out_file)

    # Add styles for the flicker animation
    with open(out_file, 'r' if PY3 else 'rb') as f:
        svg = f.read().split('\n')

    svg.insert(2, """\
<style type="text/css">
@keyframes flickerAnimation%s { 0%% {opacity: 1;} 100%% { opacity: 0; }}
.foreground-svg { animation: 1s ease-in-out 0s alternate none infinite running flickerAnimation%s;}
.foreground-svg:hover { animation-play-state: paused;}
</style>""" % tuple([uuid4()] * 2))
    with open(out_file, 'w' if PY3 else 'wb') as f:
        f.write('\n'.join(svg))
    return out_file
