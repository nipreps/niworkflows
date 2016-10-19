# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2016-07-21 11:28:52
""" A robust ANTs T1-to-MNI registration workflow with fallback retry """

from __future__ import print_function, division, absolute_import, unicode_literals
from os import path as op
import pkg_resources as pkgr

from nipype.interfaces.ants.registration import Registration, RegistrationOutputSpec
from nipype.interfaces.base import (traits, isdefined, BaseInterface, BaseInterfaceInputSpec,
                                    File, InputMultiPath)

from niworkflows.data import getters
from niworkflows import __packagename__, NIWORKFLOWS_LOG

MAX_RETRIES = 3


class RobustMNINormalizationInputSpec(BaseInterfaceInputSpec):
    moving_image = InputMultiPath(
        File(exists=True), mandatory=True, desc='image to apply transformation to')
    moving_mask = File(exists=True, desc='moving image mask')
    num_threads = traits.Int(1, usedefault=True, nohash=True,
                             desc="Number of ITK threads to use")
    testing = traits.Bool(False, usedefault=True, desc='use testing settings')
    orientation = traits.Enum('RAS', 'LAS', mandatory=True, usedefault=True,
                              desc='modify template orientation (should match input image)')
    template = traits.Enum(
        'mni_icbm152_linear',
        'mni_icbm152_nlin_asym_09c',
        usedefault=True, desc='define the template to be used')


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
        self.retry = 0
        self._results = {}
        super(RobustMNINormalization, self).__init__(**inputs)

    def _run_interface(self, runtime):
        settings_file = ''.join(['data/t1-mni_registration',
                                 '_testing' if self.inputs.testing else '',
                                 '_{0:03d}.json']).format


        while True:
            ants_settings = pkgr.resource_filename(
                'niworkflows', settings_file(self.retry))
            NIWORKFLOWS_LOG.info('Retry #%d, settings file "%s"', self.retry,
                                 ants_settings)
            norm = self._config_ants(ants_settings)
            try:
                interface_result = norm.run()
                break
            except Exception as exc:
                NIWORKFLOWS_LOG.warn('Retry #%d failed. Reason:\n%s', self.retry,
                                     exc)
                self.retry += 1
                if self.retry > MAX_RETRIES:
                    raise

        self._results.update(interface_result.outputs.get())
        return runtime

    def _config_ants(self, ants_settings):
        norm = Registration(
            moving_image=self.inputs.moving_image,
            num_threads=self.inputs.num_threads,
            from_file=ants_settings
        )
        if isdefined(self.inputs.moving_mask):
            norm.inputs.moving_image_mask = self.inputs.moving_mask

        get_template = getattr(getters, 'get_{}'.format(self.inputs.template))
        mni_template = get_template()

        if self.inputs.orientation == 'LAS':
            raise NotImplementedError

        resolution = 1
        if self.inputs.testing:
            resolution = 2

        norm.inputs.fixed_image = op.join(mni_template,
                                          '%dmm_T1.nii.gz' % resolution)
        norm.inputs.fixed_image_mask = op.join(mni_template,
                                               '%dmm_brainmask.nii.gz' % resolution)
        return norm
