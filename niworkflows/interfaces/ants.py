# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Patch ImageMath until https://github.com/nipy/nipype/pull/3210 is merged."""
from nipype.interfaces.base import Str
from nipype.interfaces.ants.utils import ImageMath as _ImageMath


ImageMath = _ImageMath
ImageMath.input_spec.operation = Str(
    mandatory=True, position=3, argstr="%s", desc="operations and intputs"
)
