#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Data fetchers module """
from __future__ import absolute_import, division, print_function, unicode_literals

from .getters import (
    get_brainweb_1mm_normal,
    get_ds003_downsampled,
    get_mni_template,
    get_mni_template_ras,
    get_ants_oasis_template,
    get_ants_oasis_template_ras,
    get_mni_epi,
    get_mni152_nlin_sym_las,
    get_mni152_nlin_sym_ras,
    get_mni_icbm152_linear,
    get_mni_icbm152_nlin_asym_09c,
    get_oasis_dkt31_mni152,
    get_hcp32k_files,
    get_conte69_mesh,
    get_dataset,
    get_template,
    get_bids_examples,
    TEMPLATE_MAP,
)
