''' class mixin and utilities for enabling reports for nipype interfaces '''

from __future__ import absolute_import, division, print_function
from io import open
import uuid
import os
import string
from abc import abstractmethod
from html.parser import HTMLParser

import jinja2
from pkg_resources import resource_filename as pkgrf

class ReportCapableInterface(object):
    ''' temporary mixin to enable reports for nipype interfaces '''

    # constants
    ERROR_REPORT = 'error'
    SUCCESS_REPORT = 'success'

    def _run_interface(self, runtime):
        ''' delegates to base interface run method, then attempts to generate reports;
        may need to be changed completely and added instead to Node.write_report() '''
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


def save_html(template, report_file_name, unique_string, **kwargs):
    ''' save an actual html file with name report_file_name. unique_string's
    first character must be alphabetical; every call to save_html must have a
    unique unique_string. kwargs should all contain valid html that will be sent
    to the jinja2 renderer '''

    if not unique_string[0].isalpha():
        raise ValueError('unique_string must be a valid id value in html; '
                         'the first character must be alphabetical. Received unique_string={}'
                         .format(unique_string))

    # validate html
    validator = HTMLValidator(unique_string=unique_string)
    for key, html in enumerate(kwargs.keys()):
        validator.feed(html)
        validator.close()

    searchpath = pkgrf('niworkflows', '/')
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(searchpath=searchpath),
        trim_blocks=True, lstrip_blocks=True
    )
    report_tpl = env.get_template('viz/' + template)
    kwargs['unique_string'] = unique_string
    report_render = report_tpl.render(kwargs)

    with open(report_file_name, 'w') as handle:
        handle.write(report_render)

class HTMLValidator(HTMLParser):
    ''' There are limitations on the html passed to save_html because
    save_html's result will be concatenated with other html strings

    html may not contain the tags 'head', 'body', 'header', 'footer', 'main',
    because those elements are supposed to be unique.

    html should also not contain '<style' because selectors/@keyframes, etc. in
    embedded CSS may conflict in unpredictable ways. However, due to lack of
    control over svg creation, this is not checked for.

    If the html contains a tag with the id attribute, the value of id must
    contain unique_string and not be equal to unique_string. In addition, all
    id's within any one save_html call are also unique from each other.

    html is assumed to be complete, valid html
    '''

    def __init__(self, unique_string):
        super(HTMLValidator, self).__init__()
        self.unique_string = unique_string
        self.bad_tags = []
        self.bad_ids = []
        self.taken_ids = [unique_string] # in template

    def handle_starttag(self, tag, attrs):
        if tag in ['head', 'body', 'header', 'footer', 'main']:
            self.bad_tags.append(tag)
        for attr, value in attrs:
            if attr=='id':
                # if unique_string is not found in the id name
                if value.find(self.unique_string) == -1:
                    self.bad_ids.append(value)
                elif value in self.taken_ids: # the value is already being used as an id
                    self.bad_ids.append(value)

    def close(self):
        super(HTMLValidator, self).close()
        error_string = ''
        if len(self.bad_tags) > 0:
            error_string = 'Found the following illegal tags: {}.\n'.format(self.bad_tags)
        if len(self.bad_ids) > 0:
            error_string = error_string + 'Found the following illegal ids: {}.\n ids must '
            'contain unique_string ({}) and be unique from each other.\n'.format(
                self.bad_ids, self.unique_string)
        if len(error_string) > 0:
            raise ValueError(error_string)

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
