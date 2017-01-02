# -*- coding: utf-8 -*-
""" test validators """
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import unittest
import mock

from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, traits
from nipype.utils.tmpdirs import InTemporaryDirectory
import nipype.pipeline.engine as pe

from niworkflows.common.report import ReportFile
from niworkflows.viz.validators import HTMLValidator, CSSValidator

DUMMY_FILE = '.coveragerc' # arbitrary existing file

class TestValidator(unittest.TestCase):
    """ Tests HTMLValidator and CSSValidator """

    unique_string = 'lala'
    mock_css = 'This is CSS.'

    def test_css_validator(self):
        def _use_validator(validator, css):
            validator.validate(css)

        def _curly_bracify(csses):
            return [css.replace('(', '{').replace(')', '}') for css in csses]
        valid_csses = self._unique_stringify(self.unique_string,
                                             ['', '@keyframe {}sdf;', '@keyframe sdf{};',
                                              '@keyframe {}{};', '@keyframe a{}a;', '* ( )',
                                              '#{} ( )', '#{} blah ( )', '#{} blah wut ( )',
                                              '    ', 'earp#{} ( )',
                                              '* ( counter-increment: {}blah )',
                                              '#{} ( counter-reset: blah{} )',
                                              '* ( color: black; counter-reset: blah{}; left: 0 )',
                                              '* ( counter-increment: {}a 1 a{} 2 a{}a 3 )',
                                              '* ( position: absolute )'])
        invalid_csses = self._unique_stringify(self.unique_string,
                                               ['#{} (position: fixed)', '* ( position: fixed )'])
        valid_csses = _curly_bracify(valid_csses)
        invalid_csses = _curly_bracify(invalid_csses)

        self._tester(
            valid_strings=valid_csses,
            invalid_strings=invalid_csses,
            validator=CSSValidator(),
            func=_use_validator)

    def test_html_validator(self):
        def _use_validator(validator, html):
            validator.simple_validate(html)

        valid_htmls = self._unique_stringify(self.unique_string,
                                             ['<super></super>', 'id=la{}', 'id={}{}', 'id = a{}',
                                              '', '<div class=heya id =yall{}', '{}', 'ID=what{}',
                                              'id=id{}', 'Id=a{}', 'iD=a{}', '<mainlandchina>'])
        invalid_htmls = self._unique_stringify(self.unique_string,
                                               ['<body>', '<head id={}x>', '<header>', '<footer>',
                                                '<p id = {}>', '<p id=wassup>',
                                                '<p id=s {}>', '<p id={} s>', '<p  ID =x {}x>',
                                                '<p id=a{}></p><p id=a{}>', '<style>', '<!DOCTYPE>',
                                                '<?xml>'])
        invalid_htmls = invalid_htmls + ['<p id=' + self.unique_string[1:] + '>']

        self._tester(
            valid_strings=valid_htmls,
            invalid_strings=invalid_htmls,
            validator=HTMLValidator(unique_string=self.unique_string),
            func=_use_validator)

    def test_html_validator_unique_id(self):
        ''' sometimes you can't know the unique id ahead of time,
        but you can divine it from the html'''
        html = '<div id={}>Stuff</div>'.format(self.unique_string)
        validator = HTMLValidator()
        validator.simple_validate(html)

        self.assertEqual(validator.unique_string, self.unique_string)

    def test_html_validator_no_unique_id(self):
        html = '<div><span id={}></span></div>'.format(self.unique_string)
        validator = HTMLValidator()

        with self.assertRaisesRegex(ValueError, 'could not be discovered'):
            validator.simple_validate(html)

    @mock.patch('niworkflows.viz.validators.CSSValidator')
    def test_html_css_interaction(self, mock_css_validator):
        ''' HTML validator should invoke CSS validator for the content of <style> tags '''
        explicitly_css = '<style type="text/css" scoped>{}</style>'.format(self.mock_css)
        implicitly_css = '<style scoped>{}</style>'.format(self.mock_css)

        validator = HTMLValidator(unique_string=self.unique_string,
                                  css_validator=mock_css_validator)
        for html in [explicitly_css, implicitly_css]:
            validator.feed(html)
            validator.close()
            validator.reset()
            mock_css_validator.validate.assert_called_once_with(self.mock_css)
            mock_css_validator.reset_mock()

        validator.feed('<not a style tag>asdfsdf</not>')
        validator.close()
        self.assertFalse(mock_css_validator.validate.called)

    def _tester(self, validator, valid_strings, invalid_strings, func):
        """ Test that the validator passes for all valid_strings;
        test that the validator throws an error for all invalid strings"""
        for string in valid_strings:
            func(validator, string)

        for string in invalid_strings:
            with self.assertRaises(ValueError):
                func(validator, string)

    def _unique_stringify(self, unique_string, strings):
        """ Utility function for inserting unique_string """
        new_strings = []
        for string in strings:
            try:
                new_string = string.format(unique_string)
            except IndexError:
                pass
            except:
                raise
            new_strings.append(new_string)

        if new_strings != strings:
            self._unique_stringify(unique_string, new_strings)

        return new_strings


class TestReportFile(unittest.TestCase):
    """ tests the custom Trait class ReportFile, defined in niworkflows/common/report.py """

    def setUp(self):
        with open(DUMMY_FILE) as file_handler:
            self.contents = file_handler.read()

    @mock.patch('niworkflows.viz.validators.HTMLValidator.simple_validate')
    def test_report_file_valid(self, mock_validator):
        """ Make sure HTMLValidator is called on the contents of the file """
        ReportFile(exists=True).validate(None, 'out_report', DUMMY_FILE)
        mock_validator.assert_called_once_with(self.contents)

    @mock.patch('niworkflows.viz.validators.HTMLValidator.simple_validate')
    def test_report_file_invalid(self, mock_validator):
        """ If the contents of the file don't pass the HTMLValidator, error """
        mock_validator.side_effect = ValueError('message')

        with self.assertRaisesRegex(traits.TraitError, 'valid'):
            ReportFile(exists=True).validate(None, 'out_report', DUMMY_FILE)

    def test_no_file(self):
        """ The file does not exist yet--behavior of ReportFile should be the same as for File """
        with self.assertRaisesRegex(traits.TraitError, 'must be a file name'):
            interface = StubInterface()
            interface.inputs.out_report = 'nonexistentfile.html'

    @mock.patch('niworkflows.viz.validators.HTMLValidator.simple_validate')
    def test_in_action(self, mock_validator):
        """ Make sure the ReportFile.validate() is called """
        with InTemporaryDirectory():
            interface = StubInterface()
            interface.run()
            self.assertTrue(mock_validator.caled)

    def test_init_exists(self):
        """ The ReportFile trait only makes sense if exists == True. """
        report_files = [ReportFile(), ReportFile(exists=True), ReportFile(exists=False)]

        for report_file in report_files:
            self.assertTrue(report_file.exists)

# Stub Interface/Input/Output to facilitate testing
class StubInputSpec(BaseInterfaceInputSpec):
    out_report = traits.File('report.html', exists=False, usedefault=True)

class StubOutputSpec(TraitedSpec):
    out_report = ReportFile(exists=True)
    out_file = traits.File('file.txt', exists=True)

class StubInterface(BaseInterface):
    input_spec = StubInputSpec
    output_spec = StubOutputSpec

    def _run_interface(self, runtime):
        open(self.inputs.out_report, 'w').close()
        return runtime

    def _list_outputs(self):
        return { 'out_report': os.path.abspath(self.inputs.out_report) }
