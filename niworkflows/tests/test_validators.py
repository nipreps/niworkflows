import unittest
import mock
from niworkflows.common import report

class TestValidator(unittest.TestCase):

    unique_string = 'lala'
    mock_css = 'This is CSS.'

    def test_css_validator(self):
        def _use_validator(validator, css):
            validator.validate(css)

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
                                               ['* ( position: fixed )'])

        for csses in [valid_csses, invalid_csses]:
            self._unique_stringify(self.unique_string, csses)
            csses = [css.replace('(', '{').replace(')', '}') for css in csses]

        self._tester(
            valid_strings=[ valid_csses],
            invalid_strings=[invalid_csses],
            validator=report.CSSValidator(),
            func=_use_validator)

    def test_html_validator(self):
        def _use_validator(validator, html):
            validator.reset()
            validator.feed(html)
            validator.close()

        valid_htmls = self._unique_stringify(self.unique_string,
                                             ['<super></super>', 'id=la{}', 'id={}{}', 'id = a{}',
                                              '', '<div class=heya id =yall{}', '{}', 'ID=what{}',
                                              'id=id{}', 'Id=a{}', 'iD=a{}'])
        invalid_htmls = self._unique_stringify(self.unique_string,
                                               ['<body>', '<head id={}x', '<header', '<footer',
                                                '<mainlandchina', 'id = {}', 'id=wassup',
                                                'id=s {}', 'id={}s', '  ID =x {}x',
                                                '<p id=a{}></p><p id=a{}>', '<style>', '<!blah>',
                                                '<?xml>'])
        invalid_htmls = invalid_htmls + ['id=' + self.unique_string[1:]]

        self._tester(
            valid_strings=valid_htmls,
            invalid_strings=invalid_htmls,
            validator = report.HTMLValidator(unique_string=self.unique_string),
            func=_use_validator)

    @mock.patch('niworkflows.common.report.CSSValidator')
    def test_html_css_interaction(self, mock_css_validator):
        ''' HTML validator should invoke CSS validator for the content of <style> tags '''
        explicitly_css = '<style type="text/css" scoped>{}</style>'.format(self.mock_css)
        implicitly_css = '<style scoped>{}</style>'.format(self.mock_css)

        validator = report.HTMLValidator(unique_string=self.unique_string,
                                         css_validator=mock_css_validator)
        for html in [explicitly_css, implicitly_css]:
            mock_css_validator.reset_mock()
            validator.feed(html)
            validator.close()
            mock_css_validator.validate.assert_called_once_with(self.mock_css)

    def _tester(self, validator, valid_strings, invalid_strings, func):
        '''
        for string in valid_strings:
            func(validator, string)
        '''
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
