# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" A robust ANTs T1-to-MNI registration workflow with fallback retry """

from __future__ import print_function, division, absolute_import, unicode_literals
from os import path as op

import pkg_resources as pkgr
from multiprocessing import cpu_count
from packaging.version import Version
import nibabel as nb
import numpy as np

from ..nipype.interfaces.ants.registration import RegistrationOutputSpec
from ..nipype.interfaces.ants import AffineInitializer
from ..nipype.interfaces.base import (
    traits, isdefined, BaseInterface, BaseInterfaceInputSpec, File)

from ..data import getters
from .. import NIWORKFLOWS_LOG, __version__
from .fixes import (
    FixHeaderApplyTransforms as ApplyTransforms,
    FixHeaderRegistration as Registration
)


niworkflows_version = Version(__version__)


class RobustMNINormalizationInputSpec(BaseInterfaceInputSpec):
    # Enable deprecation
    package_version = niworkflows_version

    moving_image = File(exists=True, mandatory=True, desc='image to apply transformation to')
    reference_image = File(exists=True, desc='override the reference image')
    moving_mask = File(exists=True, desc='moving image mask')
    reference_mask = File(exists=True, desc='reference image mask')
    num_threads = traits.Int(cpu_count(), usedefault=True, nohash=True,
                             desc="Number of ITK threads to use")
    flavor = traits.Enum('precise', 'testing', 'fast', usedefault=True,
                         desc='registration settings parameter set')
    orientation = traits.Enum('RAS', 'LAS', mandatory=True, usedefault=True,
                              desc='modify template orientation (should match input image)')
    reference = traits.Enum('T1', 'T2', 'PD', mandatory=True, usedefault=True,
                            desc='set the reference modality for registration')
    moving = traits.Enum('T1', 'EPI', usedefault=True, mandatory=True,
                         desc='registration type')
    template = traits.Enum(
        'mni_icbm152_linear',
        'mni_icbm152_nlin_asym_09c',
        usedefault=True, desc='define the template to be used')
    settings = traits.List(File(exists=True), desc='pass on the list of settings files')
    template_resolution = traits.Enum(1, 2, mandatory=True, usedefault=True,
                                      desc='template resolution')
    explicit_masking = traits.Bool(True, usedefault=True,
                                   desc="""\
Set voxels outside the masks to zero thus creating an artificial border
that can drive the registration. Requires reliable and accurate masks.
See https://sourceforge.net/p/advants/discussion/840261/thread/27216e69/#c7ba\
""")
    initial_moving_transform = File(exists=True, desc='transform for initialization')
    float = traits.Bool(False, usedefault=True, desc='use single precision calculations')


class RobustMNINormalization(BaseInterface):
    """
    An interface to robustly run T1-to-MNI spatial normalization.
    Several settings are sequentially tried until some work.
    """
    input_spec = RobustMNINormalizationInputSpec
    output_spec = RegistrationOutputSpec

    def _list_outputs(self):
        return self._results

    def __init__(self, **inputs):
        self.norm = None
        self.retry = 1
        self._results = {}
        self.terminal_output = 'file'
        super(RobustMNINormalization, self).__init__(**inputs)

    def _get_settings(self):
        if isdefined(self.inputs.settings):
            NIWORKFLOWS_LOG.info('User-defined settings, overriding defaults')
            return self.inputs.settings

        filestart = '{}-mni_registration_{}_'.format(
            self.inputs.moving.lower(), self.inputs.flavor)

        filenames = [i for i in pkgr.resource_listdir('niworkflows', 'data')
                     if i.startswith(filestart) and i.endswith('.json')]
        return [pkgr.resource_filename('niworkflows.data', f)
                for f in sorted(filenames)]

    def _run_interface(self, runtime):
        settings_files = self._get_settings()

        ants_args = self._get_ants_args()

        if not isdefined(self.inputs.initial_moving_transform):
            NIWORKFLOWS_LOG.info('Estimating initial transform using AffineInitializer')
            init = AffineInitializer(
                fixed_image=ants_args['fixed_image'],
                moving_image=ants_args['moving_image'],
                num_threads=self.inputs.num_threads)
            init.resource_monitor = False
            init.terminal_output = 'allatonce'
            init_result = init.run()
            # Save outputs (if available)
            init_out = _write_outputs(init_result.runtime, '.nipype-init')
            if init_out:
                NIWORKFLOWS_LOG.info(
                    'Terminal outputs of initialization saved (%s).',
                    ', '.join(init_out))

            ants_args['initial_moving_transform'] = init_result.outputs.out_file

        for ants_settings in settings_files:

            NIWORKFLOWS_LOG.info('Loading settings from file %s.',
                                 ants_settings)
            self.norm = Registration(from_file=ants_settings,
                                     **ants_args)
            self.norm.resource_monitor = False
            self.norm.terminal_output = self.terminal_output

            NIWORKFLOWS_LOG.info(
                'Retry #%d, commandline: \n%s', self.retry, self.norm.cmdline)
            self.norm.ignore_exception = True
            interface_result = self.norm.run()

            if interface_result.runtime.returncode != 0:
                NIWORKFLOWS_LOG.warning('Retry #%d failed.', self.retry)
                # Save outputs (if available)
                term_out = _write_outputs(interface_result.runtime,
                                          '.nipype-%04d' % self.retry)
                if term_out:
                    NIWORKFLOWS_LOG.warning(
                        'Log of failed retry saved (%s).', ', '.join(term_out))
            else:
                runtime.returncode = 0
                self._results.update(interface_result.outputs.get())
                if isdefined(self.inputs.moving_mask):
                    self._validate_results()
                NIWORKFLOWS_LOG.info(
                    'Successful spatial normalization (retry #%d).', self.retry)
                return runtime

            self.retry += 1

        raise RuntimeError(
            'Robust spatial normalization failed after %d retries.' % (self.retry - 1))

    def _get_ants_args(self):
        args = {'moving_image': self.inputs.moving_image,
                'num_threads': self.inputs.num_threads,
                'float': self.inputs.float,
                'terminal_output': 'file',
                'write_composite_transform': True,
                'initial_moving_transform': self.inputs.initial_moving_transform}

        if isdefined(self.inputs.moving_mask):
            if self.inputs.explicit_masking:
                args['moving_image'] = mask(
                    self.inputs.moving_image,
                    self.inputs.moving_mask,
                    "moving_masked.nii.gz")
            else:
                args['moving_image_mask'] = self.inputs.moving_mask

        if isdefined(self.inputs.reference_image):
            args['fixed_image'] = self.inputs.reference_image
            if isdefined(self.inputs.reference_mask):
                if self.inputs.explicit_masking:
                    args['fixed_image'] = mask(
                        self.inputs.reference_image,
                        self.inputs.mreference_mask,
                        "fixed_masked.nii.gz")
                else:
                    args['fixed_image_mask'] = self.inputs.reference_mask
        else:
            if self.inputs.orientation == 'LAS':
                raise NotImplementedError

            mni_template = getters.get_dataset(self.inputs.template)
            resolution = self.inputs.template_resolution

            if self.inputs.explicit_masking:
                args['fixed_image'] = mask(op.join(
                    mni_template, '%dmm_%s.nii.gz' % (resolution, self.inputs.reference)),
                    op.join(
                        mni_template, '%dmm_brainmask.nii.gz' % resolution),
                    "fixed_masked.nii.gz")
            else:
                args['fixed_image'] = op.join(
                    mni_template,
                    '%dmm_%s.nii.gz' % (resolution, self.inputs.reference))
                args['fixed_image_mask'] = op.join(
                    mni_template, '%dmm_brainmask.nii.gz' % resolution)

        return args

    def _validate_results(self):
        forward_transform = self._results['composite_transform']
        input_mask = self.inputs.moving_mask
        if isdefined(self.inputs.reference_mask):
            target_mask = self.inputs.reference_mask
        else:
            mni_template = getters.get_dataset(self.inputs.template)
            resolution = self.inputs.template_resolution
            target_mask = op.join(mni_template, '%dmm_brainmask.nii.gz' % resolution)

        res = ApplyTransforms(dimension=3,
                              input_image=input_mask,
                              reference_image=target_mask,
                              transforms=forward_transform,
                              interpolation='NearestNeighbor',
                              resource_monitor=False).run()
        input_mask_data = (nb.load(res.outputs.output_image).get_data() != 0)
        target_mask_data = (nb.load(target_mask).get_data() != 0)

        overlap_voxel_count = np.logical_and(input_mask_data, target_mask_data)

        overlap_perc = float(overlap_voxel_count.sum()) / float(input_mask_data.sum()) * 100

        assert overlap_perc > 50, \
            "Normalization failed: only %d%% of the normalized moving image " \
            "mask overlaps with the reference image mask." % overlap_perc


def mask(in_file, mask_file, new_name):
    import nibabel as nb
    import os

    in_nii = nb.load(in_file)
    mask_nii = nb.load(mask_file)
    data = in_nii.get_data()
    data[mask_nii.get_data() == 0] = 0
    new_nii = nb.Nifti1Image(data, in_nii.affine, in_nii.header)
    new_nii.to_filename(new_name)
    return os.path.abspath(new_name)


def _write_outputs(runtime, out_fname=None):
    if out_fname is None:
        out_fname = '.nipype'

    out_files = []
    for name in ['stdout', 'stderr', 'merged']:
        stream = getattr(runtime, name, '')
        if stream:
            out_file = op.join(runtime.cwd, name + out_fname)
            with open(out_file, 'w') as outf:
                print(stream, file=outf)
            out_files.append(out_file)
    return out_files
