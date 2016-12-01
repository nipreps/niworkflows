# -*- coding: utf-8 -*-
""" test validators """

from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import mock

from niworkflows.viz.utils import save_html
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


class TestHTMLValidator(unittest.TestCase):

    def test_html_validator(self):
        unique_string = 'lala'
        valid_htmls = ['<super></super>', 'id=la' + unique_string,
                       'id=' + unique_string + unique_string, 'id = a' + unique_string, '',
                       '<div class=heya id =yall' + unique_string, unique_string,
                       'ID=what' + unique_string, 'id=id' + unique_string, 'Id=a' + unique_string,
                       'iD=a' + unique_string]
        invalid_htmls = ['<body>', '<head id=' + unique_string + 'x', '<header', '<footer',
                         '<mainlandchina', 'id = ' + unique_string, 'id=wassup',
                         'id=s ' + unique_string, 'id=' + unique_string + ' s',
                         'id=' + unique_string[1:], '  ID =x ' + unique_string + 'x',
                         '<p id=a' + unique_string + '></p><p id=a' + unique_string + '>']

        validator = HTMLValidator(unique_string=unique_string)

        for html in valid_htmls:
            validator.feed(html)
            validator.close()

        for html in invalid_htmls:
            validator.feed(html)
            with self.assertRaises(ValueError):
                validator.close()

    @mock.patch('jinja2.Environment')
    @mock.patch('niworkflows.common.report.open', mock.mock_open(), create=True, name=open_mock)
    def test_save_html(self, jinja_mock):
        template_mock= mock.MagicMock()
        jinja_mock.return_value.get_template.return_value = template_mock

        unique_string = 'unique string'
        html = 'some html'
        report_file_name = 'report file name'

        save_html(template='overlay_3d_report.tpl',
                         report_file_name=report_file_name,
                         unique_string=unique_string,
                         another_keyword=html)

        template_mock.render.assert_called_once_with({'unique_string': unique_string,
                                                      'another_keyword': html})

