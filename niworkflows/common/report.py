# -*- coding: utf-8 -*-
# @Author: shoshber
""" class mixin and utilities for enabling reports for nipype interfaces """

from __future__ import absolute_import, division, print_function

import os
from abc import abstractmethod
from nipype.interfaces.base import File, traits
from niworkflows import NIWORKFLOWS_LOG

class ReportCapableInputSpec(object):
    generate_report = traits.Bool(
        False, usedefault=True, desc="Set to true to enable report generation for node")

    out_report = File(
        'report.html', usedefault=True, desc='filename for the visual report')

class ReportCapableInterface(object):
    """ temporary mixin to enable reports for nipype interfaces """

    def __init__(self, **inputs):
        self._out_report = None
        super(ReportCapableInterface, self).__init__(**inputs)

    def _run_interface(self, runtime):
        """ delegates to base interface run method, then attempts to generate reports """
        # make this _run_interface seamless (avoid wrap it into try..except)
        runtime = super(ReportCapableInterface, self)._run_interface(runtime)

        # leave early if there's nothing to do
        if not self.inputs.generate_report:
            return runtime

        # check exit code and act consequently
        NIWORKFLOWS_LOG.debug('Running report generation code')
        if hasattr(runtime, 'returncode') and runtime.returncode == 0:
            self._out_report = self._generate_report()

        if self._out_report is not None:
            NIWORKFLOWS_LOG.info('Successfully created report (%s)', self._out_report)
        else:
            NIWORKFLOWS_LOG.warn('Interface did not exit gracefully (exit_code=%s)',
                                 runtime.get('returncode', 'None'))
            self._out_report = self._generate_error_report(
                errno=runtime.get('returncode', None))

        return runtime

    def _list_outputs(self):
        outputs = super(ReportCapableInterface, self)._list_outputs()
        if self._out_report is not None:
            outputs['html_report'] = self._out_report
        return outputs

    @abstractmethod
    def _generate_report(self):
        """
        Saves an html object - returns the path to the generated
        snippet or None if there was an error.
        """

    @abstractmethod
    def _generate_error_report(self, errno=None):
        """ Saves an html snippet """
        # as of now we think this will be the same for every interface

