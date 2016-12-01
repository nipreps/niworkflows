# -*- coding: utf-8 -*-
""" css/html validation """
from __future__ import absolute_import, division, print_function, unicode_literals

from html.parser import HTMLParser
import tinycss
import warnings
from builtins import object

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

    html is assumed to be complete, valid html
    '''

    def __init__(self, unique_string, css_validator=CSSValidator()):
        self.unique_string = unique_string
        self.css_validator = css_validator

        # Class' members should be initialized here
        self.bad_tags = []
        self.bad_ids = []
        self.taken_ids = [self.unique_string]  # in template
        self.in_style = False

        super(HTMLValidator, self).__init__()

    def handle_starttag(self, tag, attrs):
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
                    self.bad_ids.append(value)

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
        self.bad_tags = []
        self.bad_ids = []
        self.taken_ids = [self.unique_string]  # in template
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

        if len(error_string) > 0:
            raise ValueError(error_string)
