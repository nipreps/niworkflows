# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2016-07-21 11:28:52
""" A robust ANTs T1-to-MNI registration workflow with fallback retry """

from __future__ import print_function, division, absolute_import, unicode_literals
from os import path as op
import shutil
import pkg_resources as pkgr
from multiprocessing import cpu_count
from packaging.version import Version

from niworkflows.nipype.interfaces.ants.registration import Registration, RegistrationOutputSpec
from niworkflows.nipype.interfaces.ants.resampling import ApplyTransforms
from niworkflows.nipype.interfaces.ants import AffineInitializer
from niworkflows.nipype.interfaces.base import (
    traits, isdefined, BaseInterface, BaseInterfaceInputSpec, File)

from niworkflows.data import getters
from niworkflows import NIWORKFLOWS_LOG, __version__

import nibabel as nb
import numpy as np

niworkflows_version = Version(__version__)


class RobustMNINormalizationInputSpec(BaseInterfaceInputSpec):
    """
    Set inputs to RobustMNINormalization
    """
    # Enable deprecation
    package_version = niworkflows_version

    # Moving image.
    moving_image = File(exists=True, mandatory=True, desc='image to apply transformation to')
    # Reference image (optional).
    reference_image = File(exists=True, desc='override the reference image')
    # Moving mask (optional).
    moving_mask = File(exists=True, desc='moving image mask')
    # Reference mask (optional).
    reference_mask = File(exists=True, desc='reference image mask')
    # Lesion mask (optional).
    lesion_mask = File(exists=True, desc='lesion mask image')
    # Number of threads to use for ANTs/ITK processes.
    num_threads = traits.Int(cpu_count(), usedefault=True, nohash=True,
                             desc="Number of ITK threads to use")
    flavor = traits.Enum('precise', 'testing', 'fast', usedefault=True,
                         desc='registration settings parameter set')
    orientation = traits.Enum('RAS', 'LAS', mandatory=True, usedefault=True,
                              desc='modify template orientation (should match input image)')
    # Modality of the reference image.
    reference = traits.Enum('T1', 'T2', 'PD', mandatory=True, usedefault=True,
                            desc='set the reference modality for registration')
    # T1 or EPI registration?
    moving = traits.Enum('T1', 'EPI', usedefault=True, mandatory=True,
                         desc='registration type')
    # Template to use as the default reference image.
    template = traits.Enum(
        'mni_icbm152_linear',
        'mni_icbm152_nlin_asym_09c',
        usedefault=True, desc='define the template to be used')
    # Load other settings from file.
    settings = traits.List(File(exists=True), desc='pass on the list of settings files')
    # Resolution of the default template.
    template_resolution = traits.Enum(1, 2, mandatory=True, usedefault=True,
                                      desc='template resolution')
    # Use explicit masking?
    explicit_masking = traits.Bool(True, usedefault=True,
                                   desc="Set voxels outside the masks to zero"
                                        "thus creating an artificial border"
                                        "that can drive the registration. "
                                        "Requires reliable and accurate masks."
                                        "See https://sourceforge.net/p/advants/discussion/840261/thread/27216e69/#c7ba")
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
            ants_args['initial_moving_transform'] = AffineInitializer(
                fixed_image=ants_args['fixed_image'],
                moving_image=ants_args['moving_image'],
                num_threads=self.inputs.num_threads).run().outputs.out_file

        # For each settings file...
        for ants_settings in settings_files:
            interface_result = None

            NIWORKFLOWS_LOG.info('Loading settings from file %s.',
                                 ants_settings)
            # Configure an ANTs run based on these settings.
            self.norm = Registration(from_file=ants_settings, **ants_args)

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
                if isdefined(self.inputs.moving_mask):
                    self._validate_results()

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

    def _get_ants_args(self):
        args = {'moving_image': self.inputs.moving_image,
                'num_threads': self.inputs.num_threads,
                'float': self.inputs.float,
                'terminal_output': 'file',
                'write_composite_transform': True,
                'initial_moving_transform': self.inputs.initial_moving_transform}


        if isdefined(self.inputs.moving_mask):
            # If explicit masking is enabled...
            if self.inputs.explicit_masking:
                args['moving_image'] = mask(
                    self.inputs.moving_image,
                    self.inputs.moving_mask,
                    "moving_masked.nii.gz")
            # But explicit masking is disabled...
            else:
                args['moving_image_mask'] = self.inputs.moving_mask

            # If a lesion mask is also provided...
            if isdefined(self.inputs.lesion_mask):
                args['moving_image_mask'] = create_cfm(
                    self.inputs.moving_mask,
                    "moving_cfm.nii.gz",
                    self.inputs.lesion_mask,
                    global_mask=not self.inputs.explicit_masking)

        elif isdefined(self.inputs.lesion_mask):
            # If a lesion mask is also provided...
            args['moving_image_mask'] = create_cfm(
                self.inputs.moving_image,
                "moving_cfm.nii.gz",
                self.inputs.lesion_mask,
                global_mask=True)

        if isdefined(self.inputs.reference_image):
            args['fixed_image'] = self.inputs.reference_image
            if isdefined(self.inputs.reference_mask):
                # If explicit masking is enabled...
                if self.inputs.explicit_masking:
                    args['fixed_image'] = mask(
                        self.inputs.reference_image,
                        self.inputs.mreference_mask,
                        "fixed_masked.nii.gz")

                    # If a lesion mask is also provided...
                    if isdefined(self.inputs.lesion_mask):
                        # Create a cost function mask with the form: [global mask]
                        # Use this as the fixed mask.
                        args['fixed_image_mask'] = create_cfm(
                            self.inputs.reference_mask,
                            "fixed_cfm.nii.gz",
                            lesion_mask=None,
                            global_mask=True)

                # If a reference mask is provided...
                # But explicit masking is disabled...
                else:
                    args['fixed_image_mask'] = self.inputs.reference_mask

            # But a lesion mask "is" provided ...
            elif isdefined(self.inputs.lesion_mask):
                # Create a cost function mask with the form: [global mask]
                # Use this as the fixed mask
                args['fixed_image_mask'] = create_cfm(
                    self.inputs.reference_image,
                    "fixed_cfm.nii.gz",
                    lesion_mask=None,
                    global_mask=True)

        else:
            # Raise an error if the user specifies an unsupported image orientation
            if self.inputs.orientation == 'LAS':
                raise NotImplementedError

            # Get the template specified by the user.
            mni_template = getters.get_dataset(self.inputs.template)
            # Set the template resolution
            resolution = self.inputs.template_resolution

            # If explicit masking is enabled...
            if self.inputs.explicit_masking:
                args['fixed_image'] = mask(op.join(
                    mni_template, '%dmm_%s.nii.gz' % (resolution, self.inputs.reference)),
                    op.join(
                        mni_template, '%dmm_brainmask.nii.gz' % resolution),
                    "fixed_masked.nii.gz")

                if isdefined(self.inputs.lesion_mask):
                    # Create a cost function mask with the form: [global mask]
                    # Use this as the fixed mask.
                    args['fixed_image_mask'] = create_cfm(
                        op.join(mni_template, '%dmm_brainmask.nii.gz' % resolution),
                        "fixed_cfm.nii.gz",
                        lesion_mask=None,
                        global_mask=True)
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
                              interpolation='NearestNeighbor').run()
        input_mask_data = (nb.load(res.outputs.output_image).get_data() != 0)
        target_mask_data = (nb.load(target_mask).get_data() != 0)

        overlap_voxel_count = np.logical_and(input_mask_data, target_mask_data)

        overlap_perc = float(overlap_voxel_count.sum())/float(input_mask_data.sum())*100

        assert overlap_perc > 50, \
            "Normalization failed: only %d%% of the normalized moving image " \
            "mask overlaps with the reference image mask."%overlap_perc


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


def create_cfm(in_file, out_path, lesion_mask, global_mask=True):
    """
    Create a mask to constrain registration.

    Parameters
    ----------
    in_file : str
        Path to an existing image (usually a mask).
        If global_mask = True, this is used as a size/dimension reference.
    out_path : str
        Path/filename for the new cost function mask.
    lesion_mask : str, optional
        Path to an existing binary lesion mask.
    global_mask : bool
        Create a whole-image mask (True) or limit to reference mask (False)

    Returns
    -------
    str
        Absolute path of the new cost function mask.

    Notes
    -----
    in_file and lesion_mask must be in the same
    image space and have the same dimensions
    """
    import nibabel as nb
    import os
    import subprocess

    # Load the input image
    in_nii = nb.load(in_file)
    # If we want a global mask, create one based on the input image.
    if global_mask is True:
        in_data = in_nii.get_data()
        # Set all voxels in the input image to 1
        in_data[:] = 1
        # Replace the original input image.
        in_nii = nb.Nifti1Image(in_data, in_nii.affine, in_nii.header)

    # If a lesion mask was provided, combine it with the secondary mask.
    if lesion_mask is not None:
        # Reorient the lesion mask and write it to disk
        lm_nii = nb.as_closest_canonical(nb.load(lesion_mask))
        lm_nii.to_filename("lesion_mask_reorient.nii.gz")
        lm_reorient = os.path.abspath("lesion_mask_reorient.nii.gz")
        if global_mask is True:
            # Write the global mask to disk
            in_nii.to_filename("global_mask.nii.gz")
            # Use this as the secondary mask
            secondary_mask = os.path.abspath("global_mask.nii.gz")
        else:
            # Assume in_file is already a mask. Use this as the secondary mask.
            secondary_mask = in_file
        # Subtract the reoriented lesion mask from the secondary mask.
        res = subprocess.call(["ImageMath", "3", out_path, "-",
            secondary_mask, lm_reorient])
    else:
        assert (global_mask is True), "If no lesion mask is provided, global_mask must be True"
        # Write the global mask to disk.
        cfm_nii = in_nii
        cfm_nii.to_filename(out_path)

    return os.path.abspath(out_path)
