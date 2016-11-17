''' class mixin and utilities for enabling reports for nipype interfaces '''

from __future__ import absolute_import, division, print_function
from io import open
import uuid
import os
from abc import abstractmethod
import jinja2
from pkg_resources import resource_filename as pkgrf

class ReportCapableInterface(object):
    ''' temporary mixin to enable reports for nipype interfaces '''

    # constants
    ERROR_REPORT = 'error'
    SUCCESS_REPORT = 'success'

    def _run_interface(self, runtime):
        ''' delegates to base interface run method, then attempts to generate reports '''
        self.html_report = os.path.join(os.getcwd(), 'report.html')
        try:
            runtime = super(ReportCapableInterface, self)._run_interface(runtime)
            #  command line interfaces might not raise an exception, check return_code
            if runtime.returncode and runtime.returncode != 0:
                self._conditionally_generate_report(self.ERROR_REPORT)
            else:
                self._conditionally_generate_report(self.SUCCESS_REPORT)
            return runtime
        except:
            self._conditionally_generate_report(self.ERROR_REPORT)
            raise

    def _list_outputs(self):
        outputs = super(ReportCapableInterface, self)._list_outputs()
        if self.inputs.generate_report:
            outputs['html_report'] = self.html_report
        return outputs

    def _conditionally_generate_report(self, flag):
        ''' Do nothing if generate_report is not True.
        Otherwise delegate to a report generating method  '''

        # don't do anything unless the generate_report boolean is set to True
        if not self.inputs.generate_report:
            return

        if flag == self.SUCCESS_REPORT:
            self._generate_report()
        elif flag == self.ERROR_REPORT:
            self._generate_error_report()
        else:
            raise ValueError("Cannot generate report with flag {}. "
                             "Use constants SUCCESS_REPORT and ERROR_REPORT."
                             .format(flag))

    @abstractmethod
    def _generate_report(self):
        ''' Saves an html snippet '''

    @abstractmethod
    def _generate_error_report(self):
        ''' Saves an html snippet '''
        # as of now we think this will be the same for every interface


def save_html(template, interface_prefix, **kwargs):
    ''' save an actual html file with name report_file_name. interface_prefix must be lowercase alphabetical.
    kwargs should all contain valid html that  will be sent to the jinja2 renderer '''

    if not interface_prefix.isalpha():
        raise ValueError("interface_prefix must use alphabetical characters and have length at least 1")

    validate_html(' '.join(kwargs.values()))

    searchpath = pkgrf('niworkflows', '/')
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(searchpath=searchpath),
        trim_blocks=True, lstrip_blocks=True
    )
    report_tpl = env.get_template('viz/' + template)
    kwargs['unique_string'] = interface_prefix + uuid.uuid4()
    report_render = report_tpl.render(kwargs)

    with open(report_file_name, 'w') as handle:
        handle.write(report_render)

def validate_html(html)
    ''' There are limitations on the html passed to save_html because
    save_html's result will be concatenated with other html strings

    html may not contain 'id=', '<head', '<body', '<header', '<footer', '<main',
    because those elements are supposed to be unique. it also should not contain
    '<style' because selectors/@keyframes, etc. in embedded CSS may conflict in
    unpredictable ways. '''

    bad_strings=['id=', '<head', '<body', '<header', '<footer', '<main', '<style']
    found_strings = []
    for bad_string in bad_strings:
        if html.find(bad_string) != -1: # the bad_string is found in html
            found_strings.append(bad_string)

    if len(found_strings) > 0:
        raise ValueError('Found the following illegal strings in html ({}...{}): {}'
                         .format(html[:5], html[-5:], found_strings))


def as_svg(image):
    ''' takes an image as created by nilearn.plotting and returns a blob svg.
    A bit hacky. '''
    filename = 'temp.svg'

    image.savefig(filename)

    with open(filename, 'r') as file_obj:
        image_svg = file_obj.readlines()
    image_svg = image_svg[4:] # strip out extra DOCTYPE, etc headers
    image_svg = ''.join(image_svg) # straight up giant string

    return image_svg
