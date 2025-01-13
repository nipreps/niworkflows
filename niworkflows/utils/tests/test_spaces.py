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
"""Test spaces."""

import pytest

from ..spaces import OutputReferencesAction, Reference, SpatialReferences


@pytest.fixture
def parser():
    """Create a parser."""
    import argparse

    pars = argparse.ArgumentParser()
    pars.add_argument(
        '--spaces',
        nargs='*',
        default=SpatialReferences(),
        action=OutputReferencesAction,
        help='user defined spaces',
    )
    return pars


@pytest.mark.parametrize(
    ('spaces', 'expected'),
    [
        (('MNI152NLin6Asym',), ('MNI152NLin6Asym:res-native',)),
        (
            ('fsaverage:den-10k', 'MNI152NLin6Asym'),
            ('fsaverage:den-10k', 'MNI152NLin6Asym:res-native'),
        ),
        (
            ('fsaverage:den-10k:den-30k', 'MNI152NLin6Asym:res-1:res-2'),
            (
                'fsaverage:den-10k',
                'fsaverage:den-30k',
                'MNI152NLin6Asym:res-1',
                'MNI152NLin6Asym:res-2',
            ),
        ),
        (
            ('fsaverage:den-10k:den-30k', 'MNI152NLin6Asym:res-1:res-2', 'fsaverage5'),
            (
                'fsaverage:den-10k',
                'fsaverage:den-30k',
                'MNI152NLin6Asym:res-1',
                'MNI152NLin6Asym:res-2',
            ),
        ),
        (
            (
                'fsaverage:den-10k:den-30k',
                'MNI152NLin6Asym:res-1:res-2',
                'fsaverage:den-10k:den-30k',
                'MNI152NLin6Asym:res-1:res-2',
            ),
            (
                'fsaverage:den-10k',
                'fsaverage:den-30k',
                'MNI152NLin6Asym:res-1',
                'MNI152NLin6Asym:res-2',
            ),
        ),
        (('MNI152NLin6Asym', 'func'), ('MNI152NLin6Asym:res-native', 'func')),
    ],
)
def test_space_action(parser, spaces, expected):
    """Test action."""
    pargs = parser.parse_known_args(args=('--spaces',) + spaces)[0]
    parsed_spaces = pargs.spaces
    assert isinstance(parsed_spaces, SpatialReferences)
    assert all(isinstance(sp, Reference) for sp in parsed_spaces.references), (
        'Every element must be a `Reference`'
    )
    assert len(parsed_spaces.references) == len(expected)
    for ref, expected_ref in zip(parsed_spaces.references, expected):
        assert str(ref) == expected_ref


@pytest.mark.parametrize(('flag', 'expected'), [(('--spaces',), True), (None, False)])
def test_space_action_edgecases(parser, flag, expected):
    pargs = parser.parse_known_args(flag)[0]
    spaces = pargs.spaces
    assert spaces.is_cached() is expected
