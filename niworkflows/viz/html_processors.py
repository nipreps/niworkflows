# -*- coding: utf-8 -*-
"""Helper tools for cleaning up html. """
from __future__ import absolute_import, division, print_function, unicode_literals
from sys import version_info

from bs4 import BeautifulSoup

PY3 = version_info[0] > 2

def as_svg(image, filename='temp.svg'):
    ''' takes an image as created by nilearn.plotting and returns a blob svg.
    A bit hacky. '''
    image.savefig(filename)
    with open(filename, 'r' if PY3 else 'rb') as file_obj:
        image_svg = file_obj.readlines()

    svg_start = 0
    for i, line in enumerate(image_svg):
        if '<svg ' in line:
            svg_start = i
            continue

    image_svg = image_svg[svg_start:]  # strip out extra DOCTYPE, etc headers
    return '\n'.join(image_svg)  # straight up giant string


def uniquify(html_string):
    ''' Make HTML concatenable. To see the rules for valid concatenable HTML, see validators.py '''
    soup = BeautifulSoup(html_string, 'html.parser')
    scope_style_tags(soup)
    return str(soup)

def scope_style_tags(soup):
    ''' take a BeautifulSoup object, returns the same object with all style tags scoped '''
    for style_tag in soup.find_all('style'):
        style_tag['scoped'] = True
    return soup
