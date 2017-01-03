# -*- coding: utf-8 -*-
""" css/html validation """
from __future__ import absolute_import, division, print_function, unicode_literals
import warnings
from builtins import object, open

from html.parser import HTMLParser
import tinycss

from nipype.interfaces.base import traits

class ReportFile(traits.File):
    """ A trait that validates the HTML of reportlets for concatenatability """

    def __init__(self, *args, **kwargs):
        """ The contents of the file must pass validation, therefore the file must exist. """
        super(ReportFile, self).__init__(*args, **kwargs)
        self.exists = True

    def validate(self, object, name, value):
        """ Validates that a specified value is valid for this trait. """
        validated_value = super(ReportFile, self).validate(object, name, value)

        try:
            with open(value) as file_handler:
                HTMLValidator().simple_validate(file_handler.read())
        except ValueError:
            self.error(object, 'out_report', value)

        return validated_value

    def info(self):
        """ returns a string that will be used in the error message """
        return super(ReportFile, self).info() + ' and pass HTML validation'

class CSSValidator(object):
    ''' no attribute in CSS may be position: fixed
    Like  HTMLValidator, valid CSS is assumed and not checked.'''

    def __init__(self):
        self.parser = tinycss.make_parser()

    def validate(self, css):
        stylesheet = self.parser.parse_stylesheet(css)
        for rule in stylesheet.rules:
            self.validate_no_fixed_position(rule)
        if not stylesheet.errors is None and len(stylesheet.errors) > 0:
            warnings.warn('CSS Validator encountered the following parser errors while parsing '
                          '(CSS starting `{}`). CSS may not be syntactically correct, and CSS '
                          'Validator may not have been able to do its job. \n{}'.format(
                              css[:5], stylesheet.errors))

    def validate_no_fixed_position(self, rule):
        ''' checks counter names and position values '''
        if rule.at_keyword is not None:
            declarations = self.parser.parse_declaration_list(rule.body)
        else:  # not an at-rule
            declarations = rule.declarations

        for declaration in declarations:
            if (declaration.name == 'position' and
                    'fixed' in [value.as_css() for value in declaration.value]):
                raise ValueError('Found illegal position `fixed` in CSS.')


class HTMLValidator(HTMLParser, object):
    ''' There are limitations on the html passed to save_html because
    save_html's result will be concatenated with other html strings

    html may not contain the tags 'head', 'body', 'header', 'footer', 'main',
    because those elements are supposed to be unique.

    If the html contains a tag with the id attribute, the value of id must
    contain unique_string and not be equal to unique_string. In addition, all
    id's within any one save_html call are also unique from each other. All
    <style> tags must be scoped.

    If appropriate, invokes CSSValidator.

    If unique_string is not given to `__init__`, HTMLValidator will attempt to discover it.
    See `discover_unique_string` for details.

    html is assumed to be complete, valid html
    '''

    def __init__(self, unique_string=None, css_validator=CSSValidator()):
        self.unique_string = unique_string
        self.css_validator = css_validator

        self._reset()

        super(HTMLValidator, self).__init__()

    def simple_validate(self, html):
        """ utility to make simple cases easy """
        self.reset()
        self.feed(html)
        self.close()

    def discover_unique_string(self, attrs):
        """ the first start tag must have an id attribute,
        with that attribute's value being the unique_string """
        for attr, value in attrs:
            if attr == 'id':
                self.unique_string = value

        # if self.unique_string is not found, error immediately
        if self.unique_string is None:
            raise ValueError('unique_string was not specified and could not be discovered. '
                             'Expected the first start tag to have "id=unique_string" '
                             'but its attributes were {}'.format(attrs))

    def handle_starttag(self, tag, attrs):
        if self.unique_string is None: # unique_string hasn't been found yet
            self.discover_unique_string(attrs)

        if tag in ['head', 'body', 'header', 'footer', 'main']:
            self.bad_tags.append(tag)
        elif tag == 'style':
            self.in_style = True
            if not 'scoped' in [attribute for attribute, value in attrs]:
                self.bad_tags.append(tag)
        for attr, value in attrs:
            if attr == 'id':
                # if unique_string is not found in the id name
                if value.find(self.unique_string) == -1:
                    self.bad_ids.append(value)
                # the value is already being used as an id
                elif value in self.taken_ids:
                    self.same_ids.append(value)
                else:
                    self.taken_ids.append(value)

    def handle_endtag(self, tag):
        self.in_style = False

    def handle_data(self, data):
        if self.in_style:
            self.css_validator.validate(data)

    def handle_decl(self, decl):
        self.bad_tags.append(decl)

    def handle_pi(self, pi):
        self.bad_tags.append(pi)

    def reset(self):
        super(HTMLValidator, self).reset()
        self._reset()

    def _reset(self):
        """ internal reset helper """
        self.bad_tags = []
        self.bad_ids = []
        self.taken_ids = [self.unique_string]  # in template
        self.same_ids = []
        self.in_style = False

    def close(self):
        super(HTMLValidator, self).close()
        error_string = ''
        if len(self.bad_tags) > 0:
            error_string = (
                'Found the following illegal tags. All <style> '
                'tags must be scoped: {}.\n').format(self.bad_tags)
        if len(self.bad_ids) > 0:
            error_string += (
                'Found the following illegal ids: {}.\n ids must '
                'contain unique_string ({}) and be unique from each other.\n').format(
                    self.bad_ids, self.unique_string)
        if len(self.same_ids) > 0:
            error_string += (
                'ids must be unique, but the following same ids were found: '
                '{}\n').format(self.same_ids)
        if len(error_string) > 0:
            raise ValueError(error_string)
