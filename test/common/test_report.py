import unittest
import mock

from nipype.interfaces.fsl.utils import SmoothOutputSpec

from niworkflows.common import report
from niworkflows.viz.utils import save_html
from niworkflows.viz.validators import HTMLValidator

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
