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

        self._tester(
            valid_strings=['<super></super>', 'id=la' + self.unique_string,
                           'id=' + self.unique_string + self.unique_string,
                           'id = a' + self.unique_string, '',
                           '<div class=heya id =yall' + self.unique_string, self.unique_string,
                           'ID=what' + self.unique_string, 'id=id' + self.unique_string,
                           'Id=a' + self.unique_string, 'iD=a' + self.unique_string],
            invalid_strings=['<body>', '<head id=' + self.unique_string + 'x', '<header',
                             '<footer', '<mainlandchina', 'id = ' + self.unique_string,
                             'id=wassup', 'id=s ' + self.unique_string,
                             'id=' + self.unique_string + ' s', 'id=' + self.unique_string[1:],
                             '  ID =x ' + self.unique_string + 'x',
                             '<p id=a' + self.unique_string + '></p><p id=a' +
                             self.unique_string + '>'],
            validator = report.HTMLValidator(unique_string=self.unique_string),
            func=_use_validator)

    def _tester(self, validator, valid_strings, invalid_strings, func):
        for string in valid_strings:
            func(validator, string)

        for string in invalid_strings:
            with self.assertRaises(ValueError):
                func(validator, string)
