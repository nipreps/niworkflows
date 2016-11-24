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

        valid_htmls = self._unique_stringify(self.unique_string,
                                             ['<super></super>', 'id=la{}', 'id={}{}', 'id = a{}',
                                              '', '<div class=heya id =yall{}', '{}', 'ID=what{}',
                                              'id=id{}', 'Id=a{}', 'iD=a{}'])
        invalid_htmls = self._unique_stringify(self.unique_string,
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
