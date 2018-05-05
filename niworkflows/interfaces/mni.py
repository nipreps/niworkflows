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

from nipype.interfaces.ants.registration import RegistrationOutputSpec
from nipype.interfaces.ants import AffineInitializer
from nipype.interfaces.base import (
    traits, isdefined, SimpleInterface, BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File)

from templateflow.api import get as get_template
from .. import NIWORKFLOWS_LOG, __version__
from .fixes import (
    FixHeaderApplyTransforms as ApplyTransforms,
    FixHeaderRegistration as Registration
)


niworkflows_version = Version(__version__)


class SelectReferenceInputSpec(BaseInterfaceInputSpec):
    reference_image = File(exists=True, desc='override the reference image')
    reference_mask = File(exists=True, requires=['reference_image'],
                          desc='reference image mask')
    template = traits.Enum(
        'MNI152NLin2009cAsym',
        'OASIS',
        'NKI',
        'mni_icbm152_linear',
        usedefault=True, desc='define the template to be used')
    orientation = traits.Enum('RAS', 'LAS', mandatory=True, usedefault=True,
                              desc='modify template orientation')
    reference = traits.Enum('T1w', 'T2w', 'boldref', 'PDw', mandatory=True, usedefault=True,
                            desc='set the reference modality for registration')
    template_resolution = traits.Enum(1, 2, mandatory=True, usedefault=True,
                                      desc='template resolution')


class SelectReferenceOutputSpec(TraitedSpec):
    reference_image = File(exists=True, desc='reference image file')
    reference_mask = File(exists=True, desc='reference mask file')


class SelectReference(SimpleInterface):
    input_spec = SelectReferenceInputSpec
    output_spec = SelectReferenceOutputSpec

    def _run_interface(self, runtime):
        if isdefined(self.inputs.reference_image):
            reference_image = self.inputs.reference_image
            reference_mask = self.inputs.reference_mask
        else:
            # Raise an error if the user specifies an unsupported image orientation.
            if self.inputs.orientation == 'LAS':
                raise NotImplementedError

            resolution = self.inputs.template_resolution

            reference_image = get_template(
                self.inputs.template,
                '_res-%02d_%s.nii.gz' % (resolution, self.inputs.reference))
            reference_mask = get_template(
                self.inputs.template,
                '_res-%02d_desc-brain_mask.nii.gz' % resolution)

        self._results = {'reference_image': reference_image,
                         'reference_mask': reference_mask}

        return runtime


class RobustMNINormalizationInputSpec(BaseInterfaceInputSpec):
    """
    Set inputs to RobustMNINormalization
    """
    # Enable deprecation
    package_version = niworkflows_version

    # Moving image.
    moving_image = File(exists=True, mandatory=True, desc='image to apply transformation to')
    # Reference image (optional).
    reference_image = File(exists=True, mandatory=True, desc='reference image')
    # Moving mask (optional).
    moving_mask = File(exists=True, desc='moving image mask')
    # Reference mask (optional).
    reference_mask = File(exists=True, desc='reference image mask')
    # Number of threads to use for ANTs/ITK processes.
    num_threads = traits.Int(cpu_count(), usedefault=True, nohash=True,
                             desc="Number of ITK threads to use")
    # ANTs parameter set to use.
    flavor = traits.Enum('precise', 'testing', 'fast', usedefault=True,
                         desc='registration settings parameter set')
    # T1 or EPI registration?
    moving = traits.Enum('T1w', 'bold', usedefault=True, mandatory=True,
                         desc='registration type')
    # Load other settings from file.
    settings = traits.List(File(exists=True), desc='pass on the list of settings files')
    # Use explicit masking?
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
        """
        Return any settings defined by the user, as well as any pre-defined
        settings files that exist for the image modalities to be registered.
        """
        # If user-defined settings exist...
        if isdefined(self.inputs.settings):
            # Note this in the log and return those settings.
            NIWORKFLOWS_LOG.info('User-defined settings, overriding defaults')
            return self.inputs.settings

        # Define a prefix for output files based on the modality of the moving image.
        filestart = '{}-mni_registration_{}_'.format(
            self.inputs.moving.lower(), self.inputs.flavor)

        # Get a list of settings files that match the flavor.
        filenames = [i for i in pkgr.resource_listdir('niworkflows', 'data')
                     if i.startswith(filestart) and i.endswith('.json')]
        # Return the settings files.
        return [pkgr.resource_filename('niworkflows.data', f)
                for f in sorted(filenames)]

    def _run_interface(self, runtime):
        # Get a list of settings files.
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

        # For each settings file...
        for ants_settings in settings_files:

            NIWORKFLOWS_LOG.info('Loading settings from file %s.',
                                 ants_settings)
            # Configure an ANTs run based on these settings.
            self.norm = Registration(from_file=ants_settings,
                                     **ants_args)
            self.norm.resource_monitor = False
            self.norm.terminal_output = self.terminal_output

            # Print the retry number and command line call to the log.
            NIWORKFLOWS_LOG.info(
                'Retry #%d, commandline: \n%s', self.retry, self.norm.cmdline)
            self.norm.ignore_exception = True
            # Try running registration.
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
                # Grab the outputs.
                self._results.update(interface_result.outputs.get())
                if isdefined(self.inputs.moving_mask):
                    self._validate_results()

                # Note this in the log.
                NIWORKFLOWS_LOG.info(
                    'Successful spatial normalization (retry #%d).', self.retry)
                # Break out of the retry loop.
                return runtime

            self.retry += 1

        # If all tries fail, raise an error.
        raise RuntimeError(
            'Robust spatial normalization failed after %d retries.' % (self.retry - 1))

    def _get_ants_args(self):
        args = {'moving_image': self.inputs.moving_image,
                'fixed_image': self.inputs.reference_image,
                'moving_image_masks': self.inputs.moving_mask,
                'fixed_image_masks': self.inputs.reference_mask,
                'num_threads': self.inputs.num_threads,
                'float': self.inputs.float,
                'terminal_output': 'file',
                'write_composite_transform': True,
                'initial_moving_transform': self.inputs.initial_moving_transform}

        return args

    def _validate_results(self):
        forward_transform = self._results['composite_transform']
        input_mask = self.inputs.moving_mask
        if isdefined(self.inputs.reference_mask):
            target_mask = self.inputs.reference_mask
        else:
            resolution = self.inputs.template_resolution
            target_mask = get_template(
                self.inputs.template,
                '_res-%02d_desc-brain_mask.nii.gz' % resolution)

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
