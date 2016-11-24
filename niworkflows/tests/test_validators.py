import unittest
from niworkflows.common import report

class TestHTMLValidator(unittest.TestCase):

    def test_css_validator(self):
        pass

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

        validator = report.HTMLValidator(unique_string=unique_string)
            
        for html in valid_htmls:
            validator.feed(html)
            validator.close()
        
        for html in invalid_htmls:
            validator.feed(html)
            with self.assertRaises(ValueError):
                validator.close()
