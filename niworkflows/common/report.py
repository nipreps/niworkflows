# -*- coding: utf-8 -*-
# @Author: shoshber
""" class mixin and utilities for enabling reports for nipype interfaces """
from __future__ import absolute_import, division, print_function

import jinja2
import os
import string
from abc import abstractmethod
from io import open
from pkg_resources import resource_filename as pkgrf
from sys import version_info

from nipype.interfaces.base import File, traits, BaseInterface, BaseInterfaceInputSpec, TraitedSpec
from niworkflows import NIWORKFLOWS_LOG
from niworkflows.viz import utils as viz

PY3 = version_info[0] > 2

class ReportCapableInputSpec(BaseInterfaceInputSpec):
    generate_report = traits.Bool(
        False, usedefault=True, desc="Set to true to enable report generation for node")
    out_report = File(
        'report.html', usedefault=True, desc='filename for the visual report')

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
            runtime = super(ReportCapableInterface, self)._run_interface(runtime)
        except NotImplementedError:
            pass  # the interface is derived from BaseInterface

        # leave early if there's nothing to do
        if not self.inputs.generate_report:
            return runtime

        self._out_report = os.path.abspath(self.inputs.out_report)
        self._post_run_hook(runtime)

        # check exit code and act consequently
        NIWORKFLOWS_LOG.debug('Running report generation code')

        _report_ok = False
        if hasattr(runtime, 'returncode') and runtime.returncode == 0:
            self._generate_report()
            _report_ok = True
            NIWORKFLOWS_LOG.info('Successfully created report (%s)',
                                 self._out_report)

        if not _report_ok:
            self._generate_error_report(
                errno=runtime.get('returncode', None))

        return runtime

    def _list_outputs(self):
        outputs = super(ReportCapableInterface, self)._list_outputs()
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
        Saves an html object.
        """
        raise NotImplementedError

    def _generate_error_report(self, errno=None):
        """ Saves an html snippet """
        # as of now we think this will be the same for every interface
        errorstr = '<div><span class="error">Failed to generate report!</span>.\n'
        if errno:
            errorstr += (' <span class="error">Interface returned exit '
                         'code %d</span>\n') % errno
        errorstr += '</div>\n'
        with open(self._out_report, 'w' if PY3 else 'wb') as outfile:
            outfile.write(errorstr)

class RegistrationRCInputSpec(ReportCapableInputSpec):
    out_report = File(
        'report.svg', usedefault=True, desc='filename for the visual report')

class RegistrationRC(ReportCapableInterface):
    """ An abstract mixin to registration nipype interfaces """

    def __init__(self, **inputs):
        self._fixed_image = None
        self._moving_image = None
        super(RegistrationRC, self).__init__(**inputs)

    DEFAULT_MNI_CUTS = {
        'x': [-25, -20, -10, 0, 10, 20, 25],
        'y': [-25, -20, -10, 0, 10, 20, 25],
        'z': [-15, -10, -5, 0, 5, 10, 15]
    }

    def _generate_report(self):
        """ Generates the visual report """
        from niworkflows.viz.utils import compose_view, plot_registration
        NIWORKFLOWS_LOG.info('Generating visual report')

        # Call composer
        compose_view(
            plot_registration(self._fixed_image, 'fixed-image',
                              estimate_brightness=True,
                              cuts=self.DEFAULT_MNI_CUTS),
            plot_registration(self._moving_image, 'moving-image',
                              estimate_brightness=True,
                              cuts=self.DEFAULT_MNI_CUTS),
            out_file=self._out_report)

class SegmentationRC(ReportCapableInterface):
    """ An abstract mixin to segmentation nipype interfaces """
    def _generate_report(self):
        viz.plot_segs(
            image_nii=self._anat_file,
            seg_niis=self._seg_files,
            mask_nii=self._mask_file,
            out_file=self._out_report,
            masked=self._masked,
            title=self._report_title
        )


