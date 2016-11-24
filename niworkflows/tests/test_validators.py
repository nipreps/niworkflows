import unittest
from niworkflows.common import report

class TestValidator(unittest.TestCase):

    unique_string = 'lala'

    def test_css_validator(self):
        pass

    def test_html_validator(self):
        def _use_validator(validator, html):
            validator.feed(html)
            validator.close()

        valid_htmls = self._string_formatter(self.unique_string,
                                             ['<super></super>', 'id=la{}', 'id={}{}', 'id = a{}',
                                              '', '<div class=heya id =yall{}', '{}', 'ID=what{}',
                                              'id=id{}', 'Id=a{}', 'iD=a{}'])
        invalid_htmls = self._string_formatter(self.unique_string,
                                               ['<body>', '<head id={}x', '<header', '<footer',
                                                '<mainlandchina', 'id = {}', 'id=wassup',
                                                'id=s {}', 'id={}s', '  ID =x {}x',
                                                '<p id=a{}></p><p id=a{}>'])
        invalid_htmls = invalid_htmls + ['id=' + self.unique_string[1:]]

        self._tester(
            valid_strings=valid_htmls,
            invalid_strings=invalid_htmls,
            validator = report.HTMLValidator(unique_string=self.unique_string),
            func=_use_validator)

    def _tester(self, validator, valid_strings, invalid_strings, func):
        for string in valid_strings:
            func(validator, string)

        for string in invalid_strings:
            with self.assertRaises(ValueError):
                func(validator, string)

    def _string_formatter(self, unique_string, strings):
        new_strings = []
        for string in strings:
            try:
                new_strings.append(string.format(unique_string))
            except IndexError:
                pass

        if new_strings != strings:
            self._string_formatter(new_strings, unique_string)

        return new_strings
