#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from nipype.interfaces.base import BaseInterface


class SimpleInterface(BaseInterface):
    """ An interface pattern that allows outputs to be set in a dictionary """
    def __init__(self, **inputs):
        super(SimpleInterface, self).__init__(**inputs)
        self._results = {}

    def _list_outputs(self):
        return self._results
