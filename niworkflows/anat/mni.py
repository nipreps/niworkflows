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
    # Set the input file to be used as the moving image.
    moving_image = InputMultiPath(
        File(exists=True), mandatory=True, desc='image to apply transformation to')
    # Set an input file to be used as the reference image (instead of the default template).
    reference_image = InputMultiPath(
        File(exists=True), desc='override the reference image')
    # Set the input file to be used as the moving mask.
    moving_mask = File(exists=True, desc='moving image mask')
    # Set the input file to be used as the reference mask.
    reference_mask = File(exists=True, desc='reference image mask')
    # Set the input file to be used as the lesion mask.
    lesion_mask = File(exists=True, desc='lesion mask image')
    # Number of threads to use for ANTs/ITK processes.
    num_threads = traits.Int(cpu_count(), usedefault=True, nohash=True,
                             desc="Number of ITK threads to use")
    # Run in test mode?
    testing = traits.Bool(False, usedefault=True, desc='use testing settings')
    # Set orientation of input and template images.
    orientation = traits.Enum('RAS', 'LAS', mandatory=True, usedefault=True,
                              desc='modify template orientation (should match input image)')
    # Set the modality of the reference image.
    reference = traits.Enum('T1', 'T2', 'PD', mandatory=True, usedefault=True,
                            desc='set the reference modality for registration')
    # T1 or EPI registration?
    moving = traits.Enum('T1', 'EPI', usedefault=True, mandatory=True,
                         desc='registration type')
    # Set the default template to use.
    template = traits.Enum(
        'mni_icbm152_linear',
        'mni_icbm152_nlin_asym_09c',
        usedefault=True, desc='define the template to be used')
    # Load other settings from file.
    settings = traits.List(File(exists=True), desc='pass on the list of settings files')
    # Set the resolution of the default template.
    template_resolution = traits.Enum(1, 2, mandatory=True, usedefault=True,
                                      desc='template resolution')
    # Use explicit masking?
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
        filestart = '{}-mni_registration_'.format(self.inputs.moving.lower())
        # If running in test mode, indicate this in the output prefix.
        if self.inputs.testing:
            filestart += 'testing_'
        
        # Get a list of settings files that match the output prefix. 
        filenames = [i for i in pkgr.resource_listdir('niworkflows', 'data')
                     if i.startswith(filestart) and i.endswith('.json')]
        # Return the settings files.
        return [pkgr.resource_filename('niworkflows.data', f)
                for f in sorted(filenames)]

    def _run_interface(self, runtime):
        # Get a list of settings files.
        settings_files = self._get_settings()

        # For each settings file...
        for ants_settings in settings_files:
            interface_result = None

            # Configure an ANTs run based on these settings.
            self._config_ants(ants_settings)
            
            # Print the retry number and command line call to the log.
            NIWORKFLOWS_LOG.info(
                'Retry #%d, commandline: \n%s', self.retry, self.norm.cmdline)
            try:
                # Try running registration.
                interface_result = self.norm.run()
            except Exception as exc:
                # If registration fails, note this in the log.
                NIWORKFLOWS_LOG.warn(
                        'Retry #%d failed: %s.', self.retry, exc)

            errfile = op.join(runtime.cwd, 'stderr.nipype')
            outfile = op.join(runtime.cwd, 'stdout.nipype')
 
            shutil.move(errfile, errfile + '.%03d' % self.retry)
            shutil.move(outfile, outfile + '.%03d' % self.retry)
            
            # If registration runs successfully...
            if interface_result is not None:
                runtime.returncode = 0
                # Grab the outputs.
                self._results.update(interface_result.outputs.get())
                # Note this in the log.
                NIWORKFLOWS_LOG.info(
                    'Successful spatial normalization (retry #%d).', self.retry)
                # Break out of the retry loop.
                return runtime

            # If registration failed, increment the retry counter.
            self.retry += 1

        # If all tries fail, raise an error.
        raise RuntimeError(
            'Robust spatial normalization failed after %d retries.' % (self.retry - 1))

    def _config_ants(self, ants_settings):
        """
        Configure RobustMNINormalization based on defaults and custom
        settings specified in RobustMNINormalizationInputSpec.
        """
        NIWORKFLOWS_LOG.info('Loading settings from file %s.', ants_settings)

        # Call the Registration class from nipype.interfaces.ants
        self.norm = Registration(
            moving_image=self.inputs.moving_image,
            num_threads=self.inputs.num_threads,
            from_file=ants_settings,
            terminal_output='file'
        )
       
        # If the settings specify a moving mask...
        if isdefined(self.inputs.moving_mask):
            # ...and explicit masking is turned on...
            if self.inputs.explicit_masking:
                # Mask the moving image;
                # Use the masked image as the moving image for Registration;
                # Do not use the moving mask during registration.
                self.norm.inputs.moving_image = mask(
                    self.inputs.moving_image[0],
                    self.inputs.moving_mask,
                    "moving_masked.nii.gz")
            else:
                # Use the moving mask during registration.
                self.norm.inputs.moving_image_mask = self.inputs.moving_mask

        # If the settings specify a reference image...
        if isdefined(self.inputs.reference_image):
            # ...set that reference image as the fixed image.
            self.norm.inputs.fixed_image = self.inputs.reference_image
            # if the settings specify a reference mask...
            if isdefined(self.inputs.reference_mask):
                # ...and explicit masking is turned on...
                if self.inputs.explicit_masking:
                    # Mask the moving image;
                    # Use the masked image as the moving image for Registration;
                    # Do not use the moving mask during registration.
                    self.norm.inputs.fixed_image = mask(
                        self.inputs.reference_image[0],
                        self.inputs.mreference_mask,
                        "fixed_masked.nii.gz")
                else:
                    # Use the moving mask during registration.
                    self.norm.inputs.fixed_image_mask = self.inputs.reference_mask
        else:
            # Get the template specified by the user.
            get_template = getattr(getters, 'get_{}'.format(self.inputs.template))
            mni_template = get_template()

            # Raise an error if the user specifies an unsupported image orientation.
            if self.inputs.orientation == 'LAS':
                raise NotImplementedError

            # Set the template resolution.
            resolution = self.inputs.template_resolution
            # Use a 2mm template when running in testing mode.
            if self.inputs.testing:
                resolution = 2

            # If explicit masking is turned on...
            if self.inputs.explicit_masking:
                # Mask the template image with the pre-computed template mask;
                # Use the masked image as the fixed image for Registration;
                # Do not use a fixed image mask during registration.
                self.norm.inputs.fixed_image = mask(op.join(
                    mni_template, '%dmm_%s.nii.gz' % (resolution, self.inputs.reference)),
                    op.join(mni_template, '%dmm_brainmask.nii.gz' % resolution),
                    "fixed_masked.nii.gz")
            else:
                # Use the raw template image for Registration.
                self.norm.inputs.fixed_image = op.join(
                    mni_template, '%dmm_%s.nii.gz' % (resolution, self.inputs.reference))
                # Use the pre-computed mask for Registration.
                self.norm.inputs.fixed_image_mask = op.join(
                    mni_template, '%dmm_brainmask.nii.gz' % resolution)



def mask(in_file, mask_file, new_name):
    """
    Applies a binary mask to an image.

    Parameters
    ----------
    in_file : str
              Path to a NIfTI file.
    mask_file : str
                Path to a NIfTI file.
    new_name : str
               Path/filename for the masked output image.

    Returns
    -------
    str
        Absolute path of the masked output image.

    Notes
    -----
    in_file and mask_file must be in the same
    image space and have the same dimensions.
    """
    import nibabel as nb
    import os
    # Load the input image
    in_nii = nb.load(in_file)
    # Load the mask image
    mask_nii = nb.load(mask_file)
    # Set all non-mask voxels in the input file to zero.
    data = in_nii.get_data()
    data[mask_nii.get_data() == 0] = 0
    # Save the new masked image.
    new_nii = nb.Nifti1Image(data, in_nii.affine, in_nii.header)
    new_nii.to_filename(new_name)
    return os.path.abspath(new_name)

