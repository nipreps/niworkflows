# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Connectome workbench tools interfaces
=====================================

"""

from nipype.interfaces.base import (
    TraitedSpec, File, traits, CommandLineInputSpec)
from nipype.interfaces.workbench.base import WBCommand
from nipype import logging

iflogger = logging.getLogger('nipype.interface')


class ShortcutsInputSpec(CommandLineInputSpec):
    fs_white = File(
        exists=True, mandatory=True, argstr='%s', position=0,
        desc='FreeSurfer\'s {lh,rh}.white file')
    fs_pial = File(
        exists=True, mandatory=True, argstr='%s', position=1,
        desc='FreeSurfer\'s {lh,rh}.pial file')
    fs_sphere = File(
        exists=True, mandatory=True, argstr='%s', position=3,
        desc='FreeSurfer\'s {lh,rh}.sphere.reg file')
    out_sphere = File(
        argstr='%s', position=4, name_source=['fs_sphere'],
        name_template='fs_LR-deformed_to-fsaverage.%s.32k_fs_LR.surf.gii',
        keep_ext=False, desc='fs_LR sphere deformed to fsaverage')
    out_sphere_gii = File(
        argstr='%s', position=7, name_source=['fs_sphere'],
        name_template='%s.reg.surf.gii', keep_ext=False,
        desc='output FS sphere in GIfTI format'
        )
    hemi = traits.Enum('lh', 'rh', mandatory=True, desc='set hemisphere')
    out_fs_midthickness = File(
        argstr='%s', position=5, name_source=['hemi'],
        name_template='%s.midthickness.surf.gii',
        desc='output FS midthickness surface')
    out_midthickness = File(
        argstr='%s', position=6, name_source=['hemi'],
        name_template='%s.midthickness.32k_fs_LR.surf.gii',
        desc='output fs_LR midthickness surface')


class ShortcutsOutputSpec(TraitedSpec):
    out_sphere = File(exists=True, desc="fs_LR sphere deformed onto fsaverage")
    out_sphere_gii = File(exists=True, desc="FreeSurfer sphere in GIfTI format")
    out_midthickness = File(exists=True, desc="fs_LR midthickness")
    out_fs_midthickness = File(exists=True, desc="FreeSurfer midthickness in GIfTI format")


class Shortcuts(WBCommand):
    """
    Create GIfTI shortcuts
    """
    input_spec = ShortcutsInputSpec
    output_spec = ShortcutsOutputSpec
    _cmd = 'wb_shortcuts -freesurfer-resample-prep'
