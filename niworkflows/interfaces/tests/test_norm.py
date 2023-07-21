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
from ..norm import SpatialNormalization


def test_get_settings():
    norm = SpatialNormalization(moving='T1w', flavor='fast')
    settings = norm._get_settings()
    assert len(settings) == 1
    assert settings[0].split('/')[-1] == 't1w-mni_registration_fast_000.json'

    norm = SpatialNormalization(moving='T1w', flavor='testing')
    settings = norm._get_settings()
    assert len(settings) == 3
