# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" class mixin and utilities for enabling reports for nipype interfaces """
from __future__ import absolute_import, division, print_function, unicode_literals

import os
from sys import version_info
from abc import abstractmethod
from io import open

from nilearn.masking import apply_mask, unmask
from nilearn.image import threshold_img, load_img

from ..nipype.interfaces.base import (
    File, traits, BaseInterface, BaseInterfaceInputSpec, TraitedSpec)
from .. import NIWORKFLOWS_LOG
from ..viz.utils import cuts_from_bbox, compose_view

PY3 = version_info[0] > 2


class ReportCapableInputSpec(BaseInterfaceInputSpec):
    generate_report = traits.Bool(
        False, usedefault=True, desc="Set to true to enable report generation for node")
    out_report = File(
        'report.svg', usedefault=True, desc='filename for the visual report')
    compress_report = traits.Enum('auto', True, False, usedefault=True,
                                  desc="Compress the reportlet using SVGO or"
                                       "WEBP. 'auto' - compress if relevant "
                                       "software is installed, True = force,"
                                       "False - don't attempt to compress")


class ReportCapableOutputSpec(TraitedSpec):
    out_report = File(desc='filename for the visual report')


class ReportCapableInterface(BaseInterface):

    """ temporary mixin to enable reports for nipype interfaces """

    def __init__(self, **inputs):
        self._out_report = None
        super(ReportCapableInterface, self).__init__(**inputs)

    def _run_interface(self, runtime):
        """ delegates to base interface run method, then attempts to generate reports """
        try:
            runtime = super(
                ReportCapableInterface, self)._run_interface(runtime)
        except NotImplementedError:
            pass  # the interface is derived from BaseInterface

        # leave early if there's nothing to do
        if not self.inputs.generate_report:
            return runtime

        self._out_report = os.path.abspath(self.inputs.out_report)
        self._post_run_hook(runtime)

        # check exit code and act consequently
        NIWORKFLOWS_LOG.debug('Running report generation code')

        if hasattr(runtime, 'returncode') and runtime.returncode not in [0, None]:
            self._generate_error_report(
                errno=runtime.get('returncode', None))
        else:
            self._generate_report()
            NIWORKFLOWS_LOG.info('Successfully created report (%s)',
                                 self._out_report)

        return runtime

    def _list_outputs(self):
        try:
            outputs = super(ReportCapableInterface, self)._list_outputs()
        except NotImplementedError:
            outputs = {}
        if self._out_report is not None:
            outputs['out_report'] = self._out_report
        return outputs

    @abstractmethod
    def _post_run_hook(self, runtime):
        """ A placeholder to run stuff after the normal execution of the
        interface (i.e. assign proper inputs to reporting functions) """
        pass

    @abstractmethod
    def _generate_report(self):
        """
        Saves an svg object.
        """
        raise NotImplementedError

    def _generate_error_report(self, errno=None):
        """ Saves an svg snippet """
        # as of now we think this will be the same for every interface
        NIWORKFLOWS_LOG.warn('Report was not generated')

        errorstr = '<div><span class="error">Failed to generate report!</span>.\n'
        if errno:
            errorstr += (' <span class="error">Interface returned exit '
                         'code %d</span>.\n') % errno
        errorstr += '</div>\n'
        with open(self._out_report, 'w' if PY3 else 'wb') as outfile:
            outfile.write(errorstr)


class RegistrationRC(ReportCapableInterface):

    """ An abstract mixin to registration nipype interfaces """

    def __init__(self, **inputs):
        self._fixed_image = None
        self._moving_image = None
        self._fixed_image_mask = None
        self._fixed_image_label = "fixed"
        self._moving_image_label = "moving"
        self._contour = None
        super(RegistrationRC, self).__init__(**inputs)

    def _generate_report(self):
        """ Generates the visual report """
        from niworkflows.viz.utils import plot_registration
        NIWORKFLOWS_LOG.info('Generating visual report')

        fixed_image_nii = load_img(self._fixed_image)
        moving_image_nii = load_img(self._moving_image)
        contour_nii = load_img(self._contour) if self._contour is not None else None

        if self._fixed_image_mask:
            fixed_image_nii = unmask(apply_mask(fixed_image_nii,
                                                self._fixed_image_mask),
                                     self._fixed_image_mask)
            # since the moving image is already in the fixed image space we
            # should apply the same mask
            moving_image_nii = unmask(apply_mask(moving_image_nii,
                                                 self._fixed_image_mask),
                                      self._fixed_image_mask)
            mask_nii = load_img(self._fixed_image_mask)
        else:
            mask_nii = threshold_img(fixed_image_nii, 1e-3)

        n_cuts = 7
        if not self._fixed_image_mask and contour_nii:
            cuts = cuts_from_bbox(contour_nii, cuts=n_cuts)
        else:
            cuts = cuts_from_bbox(mask_nii, cuts=n_cuts)

        # Call composer
        compose_view(
            plot_registration(fixed_image_nii, 'fixed-image',
                              estimate_brightness=True,
                              cuts=cuts,
                              label=self._fixed_image_label,
                              contour=contour_nii,
                              compress=self.inputs.compress_report),
            plot_registration(moving_image_nii, 'moving-image',
                              estimate_brightness=True,
                              cuts=cuts,
                              label=self._moving_image_label,
                              contour=contour_nii,
                              compress=self.inputs.compress_report),
            out_file=self._out_report
        )


class SegmentationRC(ReportCapableInterface):

    """ An abstract mixin to segmentation nipype interfaces """

    def _generate_report(self):
        from niworkflows.viz.utils import plot_segs
        compose_view(
            plot_segs(
                image_nii=self._anat_file,
                seg_niis=self._seg_files,
                bbox_nii=self._mask_file,
                out_file=self.inputs.out_report,
                masked=self._masked,
                compress=self.inputs.compress_report
            ),
            fg_svgs=None,
            out_file=self._out_report
        )


class SurfaceSegmentationRC(ReportCapableInterface):

    """ An abstract mixin to registration nipype interfaces """

    def __init__(self, **inputs):
        self._anat_file = None
        self._mask_file = None
        self._contour = None
        super(SurfaceSegmentationRC, self).__init__(**inputs)

    def _generate_report(self):
        """ Generates the visual report """
        from niworkflows.viz.utils import plot_registration
        NIWORKFLOWS_LOG.info('Generating visual report')

        anat = load_img(self._anat_file)
        contour_nii = load_img(self._contour) if self._contour is not None else None

        if self._mask_file:
            anat = unmask(apply_mask(anat, self._mask_file), self._mask_file)
            mask_nii = load_img(self._mask_file)
        else:
            mask_nii = threshold_img(anat, 1e-3)

        n_cuts = 7
        if not self._mask_file and contour_nii:
            cuts = cuts_from_bbox(contour_nii, cuts=n_cuts)
        else:
            cuts = cuts_from_bbox(mask_nii, cuts=n_cuts)

        # Call composer
        compose_view(
            plot_registration(anat, 'fixed-image',
                              estimate_brightness=True,
                              cuts=cuts,
                              contour=contour_nii,
                              compress=self.inputs.compress_report),
            [],
            out_file=self._out_report
        )
