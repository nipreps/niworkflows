# -*- coding: utf-8 -*-
# @Author: shoshber
""" class mixin and utilities for enabling reports for nipype interfaces """

from __future__ import absolute_import, division, print_function

import os
from abc import abstractmethod
from nipype.interfaces.base import File, traits, BaseInterfaceInputSpec, TraitedSpec
from niworkflows import NIWORKFLOWS_LOG

class ReportCapableInputSpec(BaseInterfaceInputSpec):
    generate_report = traits.Bool(
        False, usedefault=True, desc="Set to true to enable report generation for node")
    out_report = File(
        'report.html', usedefault=True, desc='filename for the visual report')

class ReportCapableOutputSpec(TraitedSpec):
    out_report = File(desc='filename for the visual report')

class ReportCapableInterface(object):
    """ temporary mixin to enable reports for nipype interfaces """

    def __init__(self, **inputs):
        self._out_report = None
        super(ReportCapableInterface, self).__init__(**inputs)

    def _run_interface(self, runtime):
        """ delegates to base interface run method, then attempts to generate reports """
        # make this _run_interface seamless (avoid wrap it into try..except)
        try:
            runtime = super(ReportCapableInterface, self)._run_interface(runtime)
        except NotImplementedError:
            pass  # the interface is derived from BaseInterface

        # leave early if there's nothing to do
        if not self.inputs.generate_report:
            return runtime

        # check exit code and act consequently
        NIWORKFLOWS_LOG.debug('Running report generation code')
        self._out_report = os.path.abspath(self.inputs.out_report)

        _report_ok = False
        if hasattr(runtime, 'returncode') and runtime.returncode == 0:
            try:
                self._generate_report(self._out_report)
                _report_ok = True
                NIWORKFLOWS_LOG.info('Successfully created report (%s)',
                                     self._out_report)
            except Exception as excp:
                NIWORKFLOWS_LOG.warn('Report generation failed, reason: %s',
                                     excp)

        if not _report_ok:
            self._out_report = self._generate_error_report(
                errno=runtime.get('returncode', None))

        return runtime

    def _list_outputs(self):
        outputs = super(ReportCapableInterface, self)._list_outputs()
        if self._out_report is not None:
            outputs['out_report'] = self._out_report
        return outputs

    @abstractmethod
    def _generate_report(self):
        """
        Saves an html object.
        """

    @abstractmethod
    def _generate_error_report(self, errno=None):
        """ Saves an html snippet """
        # as of now we think this will be the same for every interface

