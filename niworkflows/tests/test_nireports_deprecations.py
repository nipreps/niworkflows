# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Hard deprecation coverage for NiReports-migrated APIs."""

import importlib
import re

import pytest

from niworkflows._deprecated import DeprecationError


@pytest.mark.parametrize(
    ('module_name', 'replacement_hint'),
    [
        ('niworkflows.viz', 'nireports.reportlets'),
        ('niworkflows.interfaces.plotting', 'nireports.interfaces'),
        ('niworkflows.reports', 'nireports.assembler'),
        ('niworkflows.interfaces.reportlets.masks', 'nireports.interfaces.reporting.masks'),
        (
            'niworkflows.interfaces.reportlets.segmentation',
            'nireports.interfaces.reporting.segmentation',
        ),
    ],
)
def test_deprecated_modules_raise(module_name, replacement_hint):
    with pytest.raises(DeprecationError, match=replacement_hint):
        importlib.import_module(module_name)


def test_registration_migrated_classes_raise():
    registration = importlib.import_module('niworkflows.interfaces.reportlets.registration')

    with pytest.raises(
        DeprecationError,
        match=re.escape('nireports.interfaces.reporting.registration'),
    ):
        getattr(registration, 'BBRegisterRPT')


@pytest.mark.parametrize(
    'name',
    ['SpatialNormalizationRPT', 'ANTSRegistrationRPT', 'ANTSApplyTransformsRPT'],
)
def test_registration_unported_classes_still_available(name):
    registration = importlib.import_module('niworkflows.interfaces.reportlets.registration')
    assert getattr(registration, name).__name__ == name
