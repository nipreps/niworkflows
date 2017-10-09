#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Nipype interfaces for ANTs commands
"""


from ..nipype.interfaces import base
from ..nipype.interfaces.ants.base import ANTSCommandInputSpec, ANTSCommand
from ..nipype.interfaces.base import traits


class ImageMathInputSpec(ANTSCommandInputSpec):
    dimension = traits.Int(3, usedefault=True, position=1, argstr='%d',
                           desc='dimension of output image')
    output_image = base.File(position=2, argstr='%s', name_source=['op1'],
                             name_template='%s_maths', desc='output image file')
    operation = base.Str(mandatory=True, position=2, argstr='%s',
                         desc='operations and intputs')
    op1 = base.File(exists=True, mandatory=True, position=-2, argstr='%s',
                    desc='first operator')
    op2 = traits.Either(base.File(exists=True), base.Str, mandatory=True, position=-1,
                        argstr='%s', desc='second operator')


class ImageMathOuputSpec(base.TraitedSpec):
    output_image = base.File(exists=True, desc='output image file')


class ImageMath(ANTSCommand):
    """
    Operations over images

    Example:
    --------

    """

    _cmd = 'ImageMaths'
    input_spec = ImageMathInputSpec
    output_spec = ImageMathOuputSpec


class ResampleImageBySpacingInputSpec(ANTSCommandInputSpec):
    dimension = traits.Int(3, usedefault=True, position=1, argstr='%d',
                           desc='dimension of output image')
    input_image = base.File(mandatory=True, position=2, argstr='%s',
                            desc='input image file')
    output_image = base.File(position=3, argstr='%s', name_source=['input_image'],
                             name_template='%s_resampled', desc='output image file')

class ResampleImageBySpacing(ANTSCommand):
    """
    ResampleImageBySpacing  ImageDimension inputImageFile  outputImageFile outxspc outyspc {outzspacing}  {dosmooth?}  {addvox} {nn-interp?}
 addvox pads each dimension by addvox
    """
    _cmd = 'ResampleImageBySpacing'
    input_spec = ResampleImageBySpacingInputSpec

