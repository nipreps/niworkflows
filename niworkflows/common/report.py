# -*- coding: utf-8 -*-
# @Author: shoshber
""" class mixin and utilities for enabling reports for nipype interfaces """
from __future__ import absolute_import, division, print_function

from io import open
import os
import string
from abc import abstractmethod
from html.parser import HTMLParser
import tinycss
import jinja2
from pkg_resources import resource_filename as pkgrf

from sys import version_info

from nipype.interfaces.base import File, traits, BaseInterface, BaseInterfaceInputSpec, TraitedSpec
from niworkflows import NIWORKFLOWS_LOG

PY3 = version_info[0] > 2

class ReportCapableInputSpec(BaseInterfaceInputSpec):
    generate_report = traits.Bool(
        False, usedefault=True, desc="Set to true to enable report generation for node")
    out_report = File(
        'report.html', usedefault=True, desc='filename for the visual report')

class ReportCapableOutputSpec(TraitedSpec):
    out_report = File(desc='filename for the visual report')

class ReportCapableInterface(BaseInterface):
    """ temporary mixin to enable reports for nipype interfaces """

    def __init__(self, **inputs):
        self._out_report = None
        super(ReportCapableInterface, self).__init__(**inputs)

    def _run_interface(self, runtime):
        ''' delegates to base interface run method, then attempts to generate reports;
        may need to be changed completely and added instead to Node.write_report() or, ideally, 
        .run()'''
        self.html_report = os.path.join(runtime.cwd, 'report.html')

        # make this _run_interface seamless (avoid wrap it into try..except)
        try:
            runtime = super(ReportCapableInterface, self)._run_interface(runtime)
        except NotImplementedError:
            pass  # the interface is derived from BaseInterface

        # leave early if there's nothing to do
        if not self.inputs.generate_report:
            return runtime

        self._post_run_hook(runtime)

        # check exit code and act consequently
        NIWORKFLOWS_LOG.debug('Running report generation code')
        self._out_report = os.path.abspath(self.inputs.out_report)

        _report_ok = False
        if hasattr(runtime, 'returncode') and runtime.returncode == 0:
            self._generate_report()
            _report_ok = True
            NIWORKFLOWS_LOG.info('Successfully created report (%s)',
                                 self._out_report)

        if not _report_ok:
            self._generate_error_report(
                errno=runtime.get('returncode', None))

        return runtime

    def _list_outputs(self):
        outputs = super(ReportCapableInterface, self)._list_outputs()
        if self._out_report is not None:
            outputs['out_report'] = self._out_report
        return outputs

    @abstractmethod
    def _post_run_hook(self, runtime):
        """ A placeholder to run stuff after the normal execution of the
        interface (i.e. assign proper inputs to reporting functions) """
        pass

    @abstractmethod
    def _generate_report(self):
        """
        Saves an html object.
        """
        raise NotImplementedError

    def _generate_error_report(self, errno=None):
        """ Saves an html snippet """
        # as of now we think this will be the same for every interface
        errorstr = '<div><span class="error">Failed to generate report!</span>.\n'
        if errno:
            errorstr += (' <span class="error">Interface returned exit '
                         'code %d</span>\n') % errno
        errorstr += '</div>\n'
        with open(self._out_report, 'w' if PY3 else 'wb') as outfile:
            outfile.write(errorstr)


class RegistrationRCInputSpec(ReportCapableInputSpec):
    out_report = File(
        'report.svg', usedefault=True, desc='filename for the visual report')

class RegistrationRC(ReportCapableInterface):
    """ An abstract mixin to registration nipype interfaces """

    def __init__(self, **inputs):
        self._fixed_image = None
        self._moving_image = None
        super(RegistrationRC, self).__init__(**inputs)

    DEFAULT_MNI_CUTS = {
        'x': [-25, -20, -10, 0, 10, 20, 25],
        'y': [-25, -20, -10, 0, 10, 20, 25],
        'z': [-15, -10, -5, 0, 5, 10, 15]
    }

    def _generate_report(self):
        """ Generates the visual report """
        from niworkflows.viz.utils import compose_view, plot_xyz
        NIWORKFLOWS_LOG.info('Generating visual report')

        # Call composer
        compose_view(
            plot_xyz(self._fixed_image, 'fixed-image',
                     estimate_brightness=True,
                     cuts=self.DEFAULT_MNI_CUTS),
            plot_xyz(self._moving_image, 'moving-image',
                     estimate_brightness=True,
                     cuts=self.DEFAULT_MNI_CUTS),
            out_file=self._out_report)


class SegmentationRC(ReportCapableInterface):
    """ An abstract mixin to registration nipype interfaces """
    pass

class HTMLValidator(HTMLParser):
    ''' There are limitations on the html passed to save_html because
    save_html's result will be concatenated with other html strings

    html may not contain the tags 'head', 'body', 'header', 'footer', 'main',
    because those elements are supposed to be unique.

    If the html contains a tag with the id attribute, the value of id must
    contain unique_string and not be equal to unique_string. In addition, all
    id's within any one save_html call are also unique from each other.

    If appropriate, invokes CSSValidator.

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

class CSSValidator():
    ''' CSS embedded in HTML snippets must follow the following guidelines:
    * selectors must be '*', or begin with '#unique_string' or '@keyframe'
    * keyframe names and counter names must contain, but not be equal to, the unique string
    * position may not be fixed

    Similar to  HTMLValidator, valid CSS is assumed and not checked.'''

    def __init__(self, unique_string):
        self.unique_string = unique_string
        self.parser = tinycss.make_parser()

    def validate(self, css):
        stylesheet = self.parser.parse_stylesheet(css)
        for rule in stylesheet.rules:
            self.validate_selector(rule)
            self.validate_block(rule)
        if len(parser.errors) > 0:
            warnings.warn('CSS Validator encountered the following parser errors. '
                          'CSS may not be syntactically correct, and CSS Validator may not have '
                          'been able to do its job. \n{}'.format(parser.errors))

    def validate_selectors(self, rule):
        ''' Checks for at-rules that are not keyframes, invalid keyframe names, and selectors
        that do not start with the id unique_string'''
        if rule.at_keyword is not None:
            if rule.at_keyword != 'keyframe':
                raise ValueError('Found invalid @-rule {} in CSS.'.format(rule))
            validate_unique_name(rule.head[0], '@keyframe')
        else: # not an at-rule
            biggest_selector = rule.selector[0]
            if biggest_selector[-len(self.unique_string):] != '#' + self.unique_string: # first selector must specify id unique_string
                raise ValueError('Found an invalid rule set in CSS. First selector in {} is not anything#{}'.format(rule, self.unique_string))
            elif biggest_selector != '*':
                raise ValueError('Found an invalid rule set in CSS. Selector {} is not '
                                 '@keyframe, "*", or start with id unique_string {}'.
                                 format(rule.selector, self.unique_string))

    def validate_blocks(self, rule):
        ''' checks counter names and position values '''
        if rule.at_keyword is not None:
            declarations = parser.parse_declaration_list(rule.body)
        else: # not an at-rule
            declarations = rule.declarations

        for declaration in declarations:
            if declaration.name == 'position':
                if 'fixed' in declaration.value:
                    raise ValueError('Found illegal position `fixed` in CSS.')
            elif 'counter' in declaration.name:
                for value in declaration.value:
                    if value.type == tinycss.IDENT:
                        validate_unique_name(value, 'counter')

    def validate_unique_name(self, name, title):
        if not (self.unique_string in name and self.unique_string != name):
            raise ValueError('Found illegal {} name {} in CSS. {} names must contain the unique id {}'.format(title, name, title, unique_string))
