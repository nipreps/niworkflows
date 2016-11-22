# -*- coding: utf-8 -*-
# @Author: shoshber
"""Helper tools for visualization purposes"""

from __future__ import absolute_import, division, print_function

from io import open
import jinja2
from pkg_resources import resource_filename as pkgrf

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

    with open(report_file_name, 'w') as handle:
        handle.write(report_render)

def as_svg(image):
    ''' takes an image as created by nilearn.plotting and returns a blob svg.
    A bit hacky. '''
    filename = 'temp.svg'

    image.savefig(filename)

    with open(filename, 'r') as file_obj:
        image_svg = file_obj.readlines()
    image_svg = image_svg[4:] # strip out extra DOCTYPE, etc headers
    image_svg = ''.join(image_svg) # straight up giant string

    return image_svg
