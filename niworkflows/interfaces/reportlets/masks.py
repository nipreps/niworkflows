# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""ReportCapableInterfaces for masks tools."""
import os
import numpy as np
import nibabel as nb

from nipype.interfaces import fsl, ants
from nipype.interfaces.base import (
    File,
    traits,
    isdefined,
    InputMultiPath,
    Str,
)
from nipype.interfaces.mixins import reporting
from nipype.algorithms import confounds
from ... import NIWORKFLOWS_LOG
from . import base as nrb


class _BETInputSpecRPT(nrb._SVGReportCapableInputSpec, fsl.preprocess.BETInputSpec):
    pass


class _BETOutputSpecRPT(
    reporting.ReportCapableOutputSpec, fsl.preprocess.BETOutputSpec
):
    pass


class BETRPT(nrb.SegmentationRC, fsl.BET):
    input_spec = _BETInputSpecRPT
    output_spec = _BETOutputSpecRPT

    def _run_interface(self, runtime):
        if self.generate_report:
            self.inputs.mask = True

        return super(BETRPT, self)._run_interface(runtime)

    def _post_run_hook(self, runtime):
        """generates a report showing slices from each axis of an arbitrary
        volume of in_file, with the resulting binary brain mask overlaid"""

        self._anat_file = self.inputs.in_file
        self._mask_file = self.aggregate_outputs(runtime=runtime).mask_file
        self._seg_files = [self._mask_file]
        self._masked = self.inputs.mask

        NIWORKFLOWS_LOG.info(
            'Generating report for BET. file "%s", and mask file "%s"',
            self._anat_file,
            self._mask_file,
        )

        return super(BETRPT, self)._post_run_hook(runtime)


class _BrainExtractionInputSpecRPT(
    nrb._SVGReportCapableInputSpec, ants.segmentation.BrainExtractionInputSpec
):
    pass


class _BrainExtractionOutputSpecRPT(
    reporting.ReportCapableOutputSpec, ants.segmentation.BrainExtractionOutputSpec
):
    pass


class BrainExtractionRPT(nrb.SegmentationRC, ants.segmentation.BrainExtraction):
    input_spec = _BrainExtractionInputSpecRPT
    output_spec = _BrainExtractionOutputSpecRPT

    def _post_run_hook(self, runtime):
        """ generates a report showing slices from each axis """

        brain_extraction_mask = self.aggregate_outputs(
            runtime=runtime
        ).BrainExtractionMask

        if (
            isdefined(self.inputs.keep_temporary_files)
            and self.inputs.keep_temporary_files == 1
        ):
            self._anat_file = self.aggregate_outputs(runtime=runtime).N4Corrected0
        else:
            self._anat_file = self.inputs.anatomical_image
        self._mask_file = brain_extraction_mask
        self._seg_files = [brain_extraction_mask]
        self._masked = False

        NIWORKFLOWS_LOG.info(
            'Generating report for ANTS BrainExtraction. file "%s", mask "%s"',
            self._anat_file,
            self._mask_file,
        )

        return super(BrainExtractionRPT, self)._post_run_hook(runtime)


class _ACompCorInputSpecRPT(nrb._SVGReportCapableInputSpec, confounds.CompCorInputSpec):
    pass


class _ACompCorOutputSpecRPT(
    reporting.ReportCapableOutputSpec, confounds.CompCorOutputSpec
):
    pass


class ACompCorRPT(nrb.SegmentationRC, confounds.ACompCor):
    input_spec = _ACompCorInputSpecRPT
    output_spec = _ACompCorOutputSpecRPT

    def _post_run_hook(self, runtime):
        """ generates a report showing slices from each axis """

        if len(self.inputs.mask_files) != 1:
            raise ValueError(
                "ACompCorRPT only supports a single input mask. "
                "A list %s was found." % self.inputs.mask_files
            )
        self._anat_file = self.inputs.realigned_file
        self._mask_file = self.inputs.mask_files[0]
        self._seg_files = self.inputs.mask_files
        self._masked = False

        NIWORKFLOWS_LOG.info(
            'Generating report for aCompCor. file "%s", mask "%s"',
            self.inputs.realigned_file,
            self._mask_file,
        )

        return super(ACompCorRPT, self)._post_run_hook(runtime)


class _TCompCorInputSpecRPT(
    nrb._SVGReportCapableInputSpec, confounds.TCompCorInputSpec
):
    pass


class _TCompCorOutputSpecRPT(
    reporting.ReportCapableOutputSpec, confounds.TCompCorOutputSpec
):
    pass


class TCompCorRPT(nrb.SegmentationRC, confounds.TCompCor):
    input_spec = _TCompCorInputSpecRPT
    output_spec = _TCompCorOutputSpecRPT

    def _post_run_hook(self, runtime):
        """ generates a report showing slices from each axis """

        high_variance_masks = self.aggregate_outputs(
            runtime=runtime
        ).high_variance_masks

        if isinstance(high_variance_masks, list):
            raise ValueError(
                "TCompCorRPT only supports a single output high variance mask. "
                "A list %s was found." % high_variance_masks
            )
        self._anat_file = self.inputs.realigned_file
        self._mask_file = high_variance_masks
        self._seg_files = [high_variance_masks]
        self._masked = False

        NIWORKFLOWS_LOG.info(
            'Generating report for tCompCor. file "%s", mask "%s"',
            self.inputs.realigned_file,
            self.aggregate_outputs(runtime=runtime).high_variance_masks,
        )

        return super(TCompCorRPT, self)._post_run_hook(runtime)


class _SimpleShowMaskInputSpec(nrb._SVGReportCapableInputSpec):
    background_file = File(exists=True, mandatory=True, desc="file before")
    mask_file = File(exists=True, mandatory=True, desc="file before")


class SimpleShowMaskRPT(nrb.SegmentationRC, nrb.ReportingInterface):
    input_spec = _SimpleShowMaskInputSpec

    def _post_run_hook(self, runtime):
        self._anat_file = self.inputs.background_file
        self._mask_file = self.inputs.mask_file
        self._seg_files = [self.inputs.mask_file]
        self._masked = True

        return super(SimpleShowMaskRPT, self)._post_run_hook(runtime)


class _ROIsPlotInputSpecRPT(nrb._SVGReportCapableInputSpec):
    in_file = File(
        exists=True, mandatory=True, desc="the volume where ROIs are defined"
    )
    in_rois = InputMultiPath(
        File(exists=True), mandatory=True, desc="a list of regions to be plotted"
    )
    in_mask = File(exists=True, desc="a special region, eg. the brain mask")
    masked = traits.Bool(False, usedefault=True, desc="mask in_file prior plotting")
    colors = traits.Either(
        None, traits.List(Str), usedefault=True, desc="use specific colors for contours"
    )
    levels = traits.Either(
        None,
        traits.List(traits.Float),
        usedefault=True,
        desc="pass levels to nilearn.plotting",
    )
    mask_color = Str("r", usedefault=True, desc="color for mask")


class ROIsPlot(nrb.ReportingInterface):
    input_spec = _ROIsPlotInputSpecRPT

    def _generate_report(self):
        from seaborn import color_palette
        from niworkflows.viz.utils import plot_segs, compose_view

        seg_files = self.inputs.in_rois
        mask_file = None if not isdefined(self.inputs.in_mask) else self.inputs.in_mask

        # Remove trait decoration and replace None with []
        levels = [level for level in self.inputs.levels or []]
        colors = [c for c in self.inputs.colors or []]

        if len(seg_files) == 1:  # in_rois is a segmentation
            nsegs = len(levels)
            if nsegs == 0:
                levels = np.unique(
                    np.round(nb.load(seg_files[0]).get_fdata(dtype="float32"))
                )
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
            out_file=self._out_report,
        )
