# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""py.test configuration file"""

import datetime as dt
import os
from pathlib import Path

import pytest
from templateflow.api import get as get_template

from niworkflows.testing import data_env_canary, test_data_env
from niworkflows.tests.data import load_test_data

datadir = load_test_data()


def _run_interface_mock(objekt, runtime):
    runtime.returncode = 0
    runtime.endTime = dt.datetime.isoformat(dt.datetime.now(dt.timezone.utc))

    objekt._out_report = os.path.abspath(objekt.inputs.out_report)
    objekt._post_run_hook(runtime)
    objekt._generate_report()
    return runtime


@pytest.fixture
def reference():
    return str(get_template('MNI152Lin', resolution=2, desc=None, suffix='T1w'))


@pytest.fixture
def reference_mask():
    return str(get_template('MNI152Lin', resolution=2, desc='brain', suffix='mask'))


@pytest.fixture
def moving():
    data_env_canary()
    return str(Path(test_data_env) / 'ds000003/sub-01/anat/sub-01_T1w.nii.gz')


@pytest.fixture
def nthreads():
    from multiprocessing import cpu_count

    # Tests are linear, so don't worry about leaving space for a control thread
    return min(int(os.getenv('CIRCLE_NPROCS', '8')), cpu_count())
