
import unittest
import sys
import pytest
import mock

from niworkflows.viz.utils import save_html

class TestUtils(unittest.TestCase):

    @mock.patch('jinja2.Environment')
    @mock.patch('niworkflows.common.report.open', mock.mock_open(), create=True)
    @pytest.mark.skipif(reason='this test always fails, mock not working OK')
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
