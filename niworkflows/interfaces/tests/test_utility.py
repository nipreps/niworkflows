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
"""KeySelect tests."""

from pathlib import Path

import pytest

from ..utility import KeySelect, _tsv2json


def test_KeySelect():
    """Test KeySelect."""
    with pytest.raises(ValueError, match=r'duplicated entries'):
        KeySelect(fields='field1', keys=['a', 'b', 'c', 'a'])

    with pytest.raises(ValueError, match=r'list or .* must be provided'):
        KeySelect(fields=[])


def test_tsv2json(tmp_path):
    Path.write_bytes(tmp_path / 'empty.tsv', b'')
    res = _tsv2json(tmp_path / 'empty.tsv', None, 'any_column')
    assert res == {}
    res = _tsv2json(tmp_path / 'empty.tsv', None, 'any_column', additional_metadata={'a': 'b'})
    assert res == {}
