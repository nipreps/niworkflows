# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" py.test configuration file """
import os
from tempfile import mkdtemp
import pytest

from niworkflows.data.getters import (
    get_mni_template_ras, get_ds003_downsampled, get_ants_oasis_template_ras
)


def pytest_runtest_setup(item):
    """Change to temporal directory"""
    os.chdir(mkdtemp())


@pytest.fixture
def mni_dir():
    return get_mni_template_ras()


@pytest.fixture
def reference():
    return os.path.join(get_mni_template_ras(), 'MNI152_T1_2mm.nii.gz')


@pytest.fixture
def reference_mask():
    return os.path.join(get_mni_template_ras(), 'MNI152_T1_2mm_brain_mask.nii.gz')


@pytest.fixture
def moving():
    return os.path.join(get_ds003_downsampled(), 'sub-01/anat/sub-01_T1w.nii.gz')


@pytest.fixture
def nthreads():
    from multiprocessing import cpu_count
    # Tests are linear, so don't worry about leaving space for a control thread
    return min(os.getenv('CIRCLE_NPROCS', 8), cpu_count())


@pytest.fixture
def oasis_dir():
    return get_ants_oasis_template_ras()
