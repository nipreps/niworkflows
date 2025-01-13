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
"""Utilities tests"""

import os
from shutil import which

import pytest
from nipype.pipeline import engine as pe
from templateflow.api import get as get_template

from niworkflows.interfaces.reportlets.masks import SimpleShowMaskRPT


@pytest.mark.skipif(
    which('svgo') is None or which('cwebp') is None, reason='svgo or cwebp missing'
)
def test_compression(tmp_path):
    """the BET report capable test"""

    uncompressed = (
        pe.Node(
            SimpleShowMaskRPT(
                generate_report=True,
                background_file=str(
                    get_template('OASIS30ANTs', resolution=1, desc=None, suffix='T1w')
                ),
                mask_file=str(
                    get_template(
                        'OASIS30ANTs',
                        resolution=1,
                        desc='BrainCerebellumRegistration',
                        suffix='mask',
                    )
                ),
                compress_report=False,
            ),
            name='uncompressed',
            base_dir=str(tmp_path),
        )
        .run()
        .outputs.out_report
    )

    compressed = (
        pe.Node(
            SimpleShowMaskRPT(
                generate_report=True,
                background_file=str(
                    get_template('OASIS30ANTs', resolution=1, desc=None, suffix='T1w')
                ),
                mask_file=str(
                    get_template(
                        'OASIS30ANTs',
                        resolution=1,
                        desc='BrainCerebellumRegistration',
                        suffix='mask',
                    )
                ),
                compress_report=True,
            ),
            name='compressed',
            base_dir=str(tmp_path),
        )
        .run()
        .outputs.out_report
    )

    size = int(os.stat(uncompressed).st_size)
    size_compress = int(os.stat(compressed).st_size)
    assert size >= size_compress, (
        f'The uncompressed report is smaller ({size})than the compressed report ({size_compress})'
    )
