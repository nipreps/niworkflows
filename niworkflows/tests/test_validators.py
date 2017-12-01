# -*- coding: utf-8 -*-
""" test validators """

from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import mock
from niworkflows.viz.validators import HTMLValidator, CSSValidator

class TestValidator(unittest.TestCase):

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
            validator.reset()
            validator.feed(html)
            validator.close()

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
            validator = HTMLValidator(unique_string=self.unique_string),
            func=_use_validator)

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
        for string in valid_strings:
            func(validator, string)

        for string in invalid_strings:
            with self.assertRaises(ValueError):
                func(validator, string)

    def _unique_stringify(self, unique_string, strings):
        new_strings = []
        for string in strings:
            try:
                new_string = string.format(unique_string)
            except IndexError:
                pass
            except:
                print(string)
                raise
            new_strings.append(new_string)

        if new_strings != strings:
            self._unique_stringify(unique_string, new_strings)

        return new_strings
