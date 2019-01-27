# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
ReportCapableInterfaces for masks tools

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import nibabel as nb
from nilearn.masking import compute_epi_mask
import scipy.ndimage as nd

from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces import fsl, ants
from nipype.interfaces.base import (
    SimpleInterface, BaseInterfaceInputSpec, TraitedSpec,
    traits, isdefined, InputMultiPath, File, Str)
from nipype.interfaces.mixins import reporting
from nipype.algorithms import confounds
from seaborn import color_palette
from .. import NIWORKFLOWS_LOG
from . import report_base as nrc


class BETInputSpecRPT(nrc.SVGReportCapableInputSpec,
                      fsl.preprocess.BETInputSpec):
    pass


class BETOutputSpecRPT(reporting.ReportCapableOutputSpec,
                       fsl.preprocess.BETOutputSpec):
    pass


class BETRPT(nrc.SegmentationRC, fsl.BET):
    input_spec = BETInputSpecRPT
    output_spec = BETOutputSpecRPT

    def _run_interface(self, runtime):
        if self.generate_report:
            self.inputs.mask = True

        return super(BETRPT, self)._run_interface(runtime)

    def _post_run_hook(self, runtime):
        ''' generates a report showing slices from each axis of an arbitrary
        volume of in_file, with the resulting binary brain mask overlaid '''

        self._anat_file = self.inputs.in_file
        self._mask_file = self.aggregate_outputs(runtime=runtime).mask_file
        self._seg_files = [self._mask_file]
        self._masked = self.inputs.mask

        NIWORKFLOWS_LOG.info('Generating report for BET. file "%s", and mask file "%s"',
                             self._anat_file, self._mask_file)

        return super(BETRPT, self)._post_run_hook(runtime)


class BrainExtractionInputSpecRPT(nrc.SVGReportCapableInputSpec,
                                  ants.segmentation.BrainExtractionInputSpec):
    pass


class BrainExtractionOutputSpecRPT(reporting.ReportCapableOutputSpec,
                                   ants.segmentation.BrainExtractionOutputSpec):
    pass


class BrainExtractionRPT(nrc.SegmentationRC, ants.segmentation.BrainExtraction):
    input_spec = BrainExtractionInputSpecRPT
    output_spec = BrainExtractionOutputSpecRPT

    def _post_run_hook(self, runtime):
        ''' generates a report showing slices from each axis '''

        brain_extraction_mask = self.aggregate_outputs(runtime=runtime).BrainExtractionMask

        if isdefined(self.inputs.keep_temporary_files) and self.inputs.keep_temporary_files == 1:
            self._anat_file = self.aggregate_outputs(runtime=runtime).N4Corrected0
        else:
            self._anat_file = self.inputs.anatomical_image
        self._mask_file = brain_extraction_mask
        self._seg_files = [brain_extraction_mask]
        self._masked = False

        NIWORKFLOWS_LOG.info('Generating report for ANTS BrainExtraction. file "%s", mask "%s"',
                             self._anat_file, self._mask_file)

        return super(BrainExtractionRPT, self)._post_run_hook(runtime)


# TODO: move this interface to nipype.interfaces.nilearn
class ComputeEPIMaskInputSpec(nrc.SVGReportCapableInputSpec,
                              BaseInterfaceInputSpec):
    in_file = File(exists=True, desc="3D or 4D EPI file")
    dilation = traits.Int(desc="binary dilation on the nilearn output")


class ComputeEPIMaskOutputSpec(reporting.ReportCapableOutputSpec):
    mask_file = File(exists=True, desc="Binary brain mask")


class ComputeEPIMask(nrc.SegmentationRC):
    input_spec = ComputeEPIMaskInputSpec
    output_spec = ComputeEPIMaskOutputSpec

    def _run_interface(self, runtime):
        orig_file_nii = nb.load(self.inputs.in_file)
        in_file_data = orig_file_nii.get_data()

        # pad the data to avoid the mask estimation running into edge effects
        in_file_data_padded = np.pad(in_file_data, (1, 1), 'constant',
                                     constant_values=(0, 0))

        padded_nii = nb.Nifti1Image(in_file_data_padded, orig_file_nii.affine,
                                    orig_file_nii.header)

        mask_nii = compute_epi_mask(padded_nii, exclude_zeros=True)

        mask_data = mask_nii.get_data()
        if isdefined(self.inputs.dilation):
            mask_data = nd.morphology.binary_dilation(mask_data).astype(np.uint8)

        # reverse image padding
        mask_data = mask_data[1:-1, 1:-1, 1:-1]

        # exclude zero and NaN voxels
        mask_data[in_file_data == 0] = 0
        mask_data[np.isnan(in_file_data)] = 0

        better_mask = nb.Nifti1Image(mask_data, orig_file_nii.affine,
                                     orig_file_nii.header)
        better_mask.set_data_dtype(np.uint8)
        better_mask.to_filename("mask_file.nii.gz")

        self._mask_file = os.path.join(runtime.cwd, "mask_file.nii.gz")

        runtime.returncode = 0
        return super(ComputeEPIMask, self)._run_interface(runtime)

    def _list_outputs(self):
        outputs = super(ComputeEPIMask, self)._list_outputs()
        outputs['mask_file'] = self._mask_file
        return outputs

    def _post_run_hook(self, runtime):
        ''' generates a report showing slices from each axis of an arbitrary
        volume of in_file, with the resulting binary brain mask overlaid '''

        self._anat_file = self.inputs.in_file
        self._mask_file = self.aggregate_outputs(runtime=runtime).mask_file
        self._seg_files = [self._mask_file]
        self._masked = True

        NIWORKFLOWS_LOG.info(
            'Generating report for nilearn.compute_epi_mask. file "%s", and mask file "%s"',
            self._anat_file, self._mask_file)

        return super(ComputeEPIMask, self)._post_run_hook(runtime)


class ACompCorInputSpecRPT(nrc.SVGReportCapableInputSpec,
                           confounds.CompCorInputSpec):
    pass


class ACompCorOutputSpecRPT(reporting.ReportCapableOutputSpec,
                            confounds.CompCorOutputSpec):
    pass


class ACompCorRPT(nrc.SegmentationRC, confounds.ACompCor):
    input_spec = ACompCorInputSpecRPT
    output_spec = ACompCorOutputSpecRPT

    def _post_run_hook(self, runtime):
        ''' generates a report showing slices from each axis '''

        assert len(self.inputs.mask_files) == 1, \
            "ACompCorRPT only supports a single input mask. " \
            "A list %s was found." % self.inputs.mask_files
        self._anat_file = self.inputs.realigned_file
        self._mask_file = self.inputs.mask_files[0]
        self._seg_files = self.inputs.mask_files
        self._masked = False

        NIWORKFLOWS_LOG.info('Generating report for aCompCor. file "%s", mask "%s"',
                             self.inputs.realigned_file, self._mask_file)

        return super(ACompCorRPT, self)._post_run_hook(runtime)


class PrepareRegistrationImagesInputSpec(BaseInterfaceInputSpec):
    fixed_file = File(exists=True, mandatory=True)
    moving_file = File(exists=True, mandatory=True)
    fixed_mask = File(exists=True)
    moving_mask = File(exists=True)
    lesion_mask = File(exists=True)
    explicit_masking = traits.Bool(
        True, usedefault=True,
        desc='Set voxels outside the masks to zero thus creating an artificial border that can '
             'that can drive the registration. Requires reliable and accurate masks. See '
             'See https://sourceforge.net/p/advants/discussion/840261/thread/27216e69/#c7ba')


class PrepareRegistrationImagesOutputSpec(TraitedSpec):
    fixed_file = File(exists=True)
    moving_file = File(exists=True)
    fixed_mask = File(exists=True)
    moving_mask = File(exists=True)


class PrepareRegistrationImages(SimpleInterface):
    input_spec = PrepareRegistrationImagesInputSpec
    output_spec = PrepareRegistrationImagesOutputSpec

    def _run_interface(self, runtime):
        fixed_file = self.inputs.fixed_file
        moving_file = self.inputs.moving_file
        fixed_mask = self.inputs.fixed_mask
        moving_mask = self.inputs.moving_mask
        lesion_mask = self.inputs.lesion_mask
        explicit_masking = self.inputs.explicit_masking

        # Default behavior: pass-through
        self._results = {
            'fixed_file': fixed_file,
            'moving_file': moving_file,
            }

        if explicit_masking:
            if fixed_mask:
                self._results['fixed_file'] = apply_mask(fixed_file, fixed_mask, cwd=runtime.cwd)
            if moving_mask:
                self._results['moving_file'] = apply_mask(moving_file, moving_mask,
                                                          cwd=runtime.cwd)
        else:
            self._results.update({
                'fixed_mask': fixed_mask,
                'moving_mask': moving_mask,
                })

        if lesion_mask:
            self._results['fixed_mask'] = create_cfm(
                fixed_mask or fixed_file,
                lesion_mask=None,
                global_mask=True,
                cwd=runtime.cwd)
            self._results['moving_mask'] = create_cfm(
                moving_mask or moving_file,
                lesion_mask=lesion_mask,
                global_mask=explicit_masking or not moving_mask,
                cwd=runtime.cwd)

        return runtime


def apply_mask(in_file, mask_file, out_filename=None, cwd=None):
    """
    Apply a binary mask to an image.

    Parameters
    ----------
    in_file : str
        Path to an image to mask
    mask_file : str
        Path to a binary mask
    out_filename : str (optional)
        Base filename for output file - if ``None``, ``in_file`` is used
    cwd : str (optional)
        Working directory to create CFM in - if ``None``, ``os.getcwd()`` is used

    Returns
    -------
    str
        Absolute path of the masked output image.

    Notes
    -----
    in_file and mask_file must be in the same
    image space and have the same dimensions.
    """
    if cwd is None:
        cwd = os.getcwd()

    out_filename = fname_presuffix(out_filename or in_file, suffix='_masked', newpath=cwd)

    img = nb.load(in_file)
    mask = nb.load(mask_file)
    try:
        _check_space(img, mask)
    except ValueError as err:
        msg = err.args[0]
        raise ValueError(msg + 'Files: {}, {}'.format(in_file, mask_file))
    data = img.get_data() * (mask.get_data() > 0)

    img.__class__(data, img.affine, img.header).to_filename(out_filename)

    return out_filename


def create_cfm(in_file, lesion_mask, global_mask, out_filename=None, cwd=None):
    """
    Create a cost function mask (CFM) to constrain registration.

    Parameters
    ----------
    in_file : str
        Path to an existing image (usually a mask).
        If global_mask = True, this is used as a size/dimension reference.
    lesion_mask : str, optional
        Path to an existing binary lesion mask.
    global_mask : bool
        Create a whole-image mask (True) or limit to reference mask (False)
        A whole image-mask is 1 everywhere
    out_filename : str (optional)
        Base filename for output file - if ``None``, ``in_file`` is used
    cwd : str (optional)
        Working directory to create CFM in - if ``None``, ``os.getcwd()`` is used

    Returns
    -------
    str
        Absolute path of the new cost function mask.

    Notes
    -----
    in_file and lesion_mask must be in the same
    image space and have the same dimensions
    """
    if cwd is None:
        cwd = os.getcwd()

    out_filename = fname_presuffix(out_filename or in_file, suffix='_cfm', newpath=cwd)

    if not global_mask and not lesion_mask:
        NIWORKFLOWS_LOG.warning(
            'No lesion mask was provided and global_mask not requested, '
            'therefore the original mask will not be modified.')

    # Load the input image
    img = nb.load(in_file)

    # If we want a global mask, create one based on the input image.
    mask = np.ones(img.shape, dtype=np.uint8) if global_mask else img.get_data()
    if not is_binary(mask):
        raise ValueError("`global_mask` must be true if `in_file` is not a binary mask")

    # If a lesion mask was provided, combine it with the secondary mask.
    if lesion_mask is not None:
        # Reorient the lesion mask and get the data.
        lm_img = nb.as_closest_canonical(nb.load(lesion_mask))

        try:
            _check_space(img, lm_img)
        except ValueError as err:
            msg = err.args[0]
            raise ValueError(msg + 'Files: {}, {}'.format(in_file, lesion_mask))

        if not is_binary(lm_img.get_data()):
            raise ValueError('`lesion_mask` must be a binary mask')

        # Subtract lesion mask from secondary mask, set negatives to 0
        mask = np.fmax(mask - lm_img.get_data(), 0)

    # Coerce to uint8, use np.asanyarray to avoid unnecessary copies
    mask = np.asanyarray(mask, dtype=np.uint8)

    # Use original image type to permit Nifti2, but check against the
    # Nifti1 base class before using the Nifti-specific set_data_dtype
    cfm_img = img.__class__(mask, img.affine, img.header)
    if isinstance(cfm_img, nb.Nifti1Pair):
        cfm_img.set_data_dtype(np.uint8)

    cfm_img.to_filename(out_filename)

    return out_filename


def _check_space(img_a, img_b):
    if img_a.shape != img_b.shape:
        raise ValueError('Incompatible shapes: {!r}, {!r}'.format(img_a.shape, img_b.shape))
    if not np.allclose(img_a.affine, img_b.affine):
        raise ValueError('Mismatched affines')


def is_binary(data):
    return set(np.unique(data)).issubset({0, 1})


class TCompCorInputSpecRPT(nrc.SVGReportCapableInputSpec,
                           confounds.TCompCorInputSpec):
    pass


class TCompCorOutputSpecRPT(reporting.ReportCapableOutputSpec,
                            confounds.TCompCorOutputSpec):
    pass


class TCompCorRPT(nrc.SegmentationRC, confounds.TCompCor):
    input_spec = TCompCorInputSpecRPT
    output_spec = TCompCorOutputSpecRPT

    def _post_run_hook(self, runtime):
        ''' generates a report showing slices from each axis '''

        high_variance_masks = self.aggregate_outputs(runtime=runtime).high_variance_masks

        assert not isinstance(high_variance_masks, list),\
            "TCompCorRPT only supports a single output high variance mask. " \
            "A list %s was found." % str(high_variance_masks)
        self._anat_file = self.inputs.realigned_file
        self._mask_file = high_variance_masks
        self._seg_files = [high_variance_masks]
        self._masked = False

        NIWORKFLOWS_LOG.info('Generating report for tCompCor. file "%s", mask "%s"',
                             self.inputs.realigned_file,
                             self.aggregate_outputs(runtime=runtime).high_variance_masks)

        return super(TCompCorRPT, self)._post_run_hook(runtime)


class SimpleShowMaskInputSpec(nrc.SVGReportCapableInputSpec):
    background_file = File(exists=True, mandatory=True, desc='file before')
    mask_file = File(exists=True, mandatory=True, desc='file before')


class SimpleShowMaskRPT(nrc.SegmentationRC, nrc.ReportingInterface):
    input_spec = SimpleShowMaskInputSpec

    def _post_run_hook(self, runtime):
        self._anat_file = self.inputs.background_file
        self._mask_file = self.inputs.mask_file
        self._seg_files = [self.inputs.mask_file]
        self._masked = True

        return super(SimpleShowMaskRPT, self)._post_run_hook(runtime)


class ROIsPlotInputSpecRPT(nrc.SVGReportCapableInputSpec):
    in_file = File(exists=True, mandatory=True, desc='the volume where ROIs are defined')
    in_rois = InputMultiPath(File(exists=True), mandatory=True,
                             desc='a list of regions to be plotted')
    in_mask = File(exists=True, desc='a special region, eg. the brain mask')
    masked = traits.Bool(False, usedefault=True, desc='mask in_file prior plotting')
    colors = traits.Either(None, traits.List(Str), usedefault=True,
                           desc='use specific colors for contours')
    levels = traits.Either(None, traits.List(traits.Float),
                           usedefault=True, desc='pass levels to nilearn.plotting')
    mask_color = Str('r', usedefault=True, desc='color for mask')


class ROIsPlot(nrc.ReportingInterface):
    input_spec = ROIsPlotInputSpecRPT

    def _generate_report(self):
        from niworkflows.viz.utils import plot_segs, compose_view
        seg_files = self.inputs.in_rois
        mask_file = None if not isdefined(self.inputs.in_mask) \
            else self.inputs.in_mask

        # Remove trait decoration and replace None with []
        levels = [l for l in self.inputs.levels or []]
        colors = [c for c in self.inputs.colors or []]

        if len(seg_files) == 1:  # in_rois is a segmentation
            nsegs = len(levels)
            if nsegs == 0:
                levels = np.unique(np.round(
                    nb.load(seg_files[0]).get_data()).astype(int))
                levels = (levels[levels > 0] - 0.5).tolist()
                nsegs = len(levels)

            levels = [levels]
            missing = nsegs - len(colors)
            if missing > 0:
                colors = colors + color_palette("husl", missing)
            colors = [colors]
        else:  # in_rois is a list of masks
            nsegs = len(seg_files)
            levels = [[0.5]] * nsegs
            missing = nsegs - len(colors)
            if missing > 0:
                colors = [[c] for c in colors + color_palette("husl", missing)]

        if mask_file:
            seg_files.insert(0, mask_file)
            if levels:
                levels.insert(0, [0.5])
            colors.insert(0, [self.inputs.mask_color])
            nsegs += 1

        self._out_report = os.path.abspath(self.inputs.out_report)
        compose_view(
            plot_segs(
                image_nii=self.inputs.in_file,
                seg_niis=seg_files,
                bbox_nii=mask_file,
                levels=levels,
                colors=colors,
                out_file=self.inputs.out_report,
                masked=self.inputs.masked,
                compress=self.inputs.compress_report,
            ),
            fg_svgs=None,
            out_file=self._out_report
        )
