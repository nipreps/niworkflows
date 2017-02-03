# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2016-07-21 11:28:52
""" A robust ANTs T1-to-MNI registration workflow with fallback retry """

from __future__ import print_function, division, absolute_import, unicode_literals
from os import path as op
import shutil
import pkg_resources as pkgr
from multiprocessing import cpu_count

from nipype.interfaces.ants.registration import Registration, RegistrationOutputSpec
from nipype.interfaces.base import (traits, isdefined, BaseInterface, BaseInterfaceInputSpec,
                                    File, InputMultiPath)

from niworkflows.data import getters
from niworkflows import __packagename__, NIWORKFLOWS_LOG

class RobustMNINormalizationInputSpec(BaseInterfaceInputSpec):
    moving_image = InputMultiPath(
        File(exists=True), mandatory=True, desc='image to apply transformation to')
    reference_image = InputMultiPath(
        File(exists=True), desc='override the reference image')
    moving_mask = File(exists=True, desc='moving image mask')
    reference_mask = File(exists=True, desc='reference image mask')
    num_threads = traits.Int(cpu_count(), usedefault=True, nohash=True,
                             desc="Number of ITK threads to use")
    testing = traits.Bool(False, usedefault=True, desc='use testing settings')
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
                                   desc="Set voxels outside the masks to zero"
                                        "thus creating an artificial border"
                                        "that can drive the registration. "
                                        "Requires reliable and accurate masks."
                                        "See https://sourceforge.net/p/advants/discussion/840261/thread/27216e69/#c7ba")


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
        self.retry = 0
        self._results = {}
        super(RobustMNINormalization, self).__init__(**inputs)

    def _get_settings(self):
        if isdefined(self.inputs.settings):
            NIWORKFLOWS_LOG.info('User-defined settings, overriding defaults')
            return self.inputs.settings

        filestart = '{}-mni_registration_'.format(self.inputs.moving.lower())
        if self.inputs.testing:
            filestart += 'testing_'

        filenames = [i for i in pkgr.resource_listdir('niworkflows', 'data')
                     if i.startswith(filestart) and i.endswith('.json')]
        return [pkgr.resource_filename('niworkflows.data', f)
                for f in sorted(filenames)]

    def _run_interface(self, runtime):
        settings_files = self._get_settings()

        for ants_settings in settings_files:
            interface_result = None

            self._config_ants(ants_settings)

            NIWORKFLOWS_LOG.info(
                'Retry #%d, commandline: \n%s', self.retry, self.norm.cmdline)
            try:
                interface_result = self.norm.run()
            except Exception as exc:
                NIWORKFLOWS_LOG.warn(
                        'Retry #%d failed: %s.', self.retry, exc)


            errfile = op.join(runtime.cwd, 'stderr.nipype')
            outfile = op.join(runtime.cwd, 'stdout.nipype')

            shutil.move(errfile, errfile + '.%03d' % self.retry)
            shutil.move(outfile, outfile + '.%03d' % self.retry)

            if interface_result is not None:
                runtime.returncode = 0
                self._results.update(interface_result.outputs.get())
                NIWORKFLOWS_LOG.info(
                    'Successful spatial normalization (retry #%d).', self.retry)
                return runtime

            self.retry += 1

        raise RuntimeError(
            'Robust spatial normalization failed after %d retries.' % (self.retry - 1))

    def _config_ants(self, ants_settings):
        NIWORKFLOWS_LOG.info('Loading settings from file %s.', ants_settings)
        self.norm = Registration(
            moving_image=self.inputs.moving_image,
            num_threads=self.inputs.num_threads,
            from_file=ants_settings,
            terminal_output='file'
        )
        if isdefined(self.inputs.moving_mask):
            if self.inputs.explicit_masking:
                self.norm.inputs.moving_image = mask(
                    self.inputs.moving_image[0],
                    self.inputs.moving_mask,
                    "moving_masked.nii.gz")
            else:
                self.norm.inputs.moving_image_mask = self.inputs.moving_mask


        if isdefined(self.inputs.reference_image):
            self.norm.inputs.fixed_image = self.inputs.reference_image
            if isdefined(self.inputs.reference_mask):
                if self.inputs.explicit_masking:
                    self.norm.inputs.fixed_image = mask(
                        self.inputs.reference_image[0],
                        self.inputs.mreference_mask,
                        "fixed_masked.nii.gz")
                else:
                    self.norm.inputs.fixed_image_mask = self.inputs.reference_mask
        else:
            get_template = getattr(getters, 'get_{}'.format(self.inputs.template))
            mni_template = get_template()

            if self.inputs.orientation == 'LAS':
                raise NotImplementedError

            resolution = self.inputs.template_resolution
            if self.inputs.testing:
                resolution = 2

            if self.inputs.explicit_masking:
                self.norm.inputs.fixed_image = mask(op.join(
                    mni_template, '%dmm_%s.nii.gz' % (resolution, self.inputs.reference)),
                    op.join(
                        mni_template, '%dmm_brainmask.nii.gz' % resolution),
                    "fixed_masked.nii.gz")
            else:
                self.norm.inputs.fixed_image = op.join(
                    mni_template,
                    '%dmm_%s.nii.gz' % (resolution, self.inputs.reference))
                self.norm.inputs.fixed_image_mask = op.join(
                    mni_template, '%dmm_brainmask.nii.gz' % resolution)



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

