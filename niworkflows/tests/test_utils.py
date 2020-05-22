# -*- coding: utf-8 -*-
""" Utilities tests """

from __future__ import absolute_import, division, print_function, unicode_literals

import os
from templateflow.api import get as get_template
from niworkflows.interfaces.masks import SimpleShowMaskRPT
from nipype.pipeline import engine as pe

import pytest
from shutil import which


@pytest.mark.skipif(
    which("svgo") is None or which("cwebp") is None, reason="svgo or cwebp missing"
)
def test_compression(tmp_path):
    """ the BET report capable test """

    uncompressed = (
        pe.Node(
            SimpleShowMaskRPT(
                generate_report=True,
                background_file=str(
                    get_template("OASIS30ANTs", resolution=1, desc=None, suffix="T1w")
                ),
                mask_file=str(
                    get_template(
                        "OASIS30ANTs",
                        resolution=1,
                        desc="BrainCerebellumRegistration",
                        suffix="mask",
                    )
                ),
                compress_report=False,
            ),
            name="uncompressed",
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
                    get_template("OASIS30ANTs", resolution=1, desc=None, suffix="T1w")
                ),
                mask_file=str(
                    get_template(
                        "OASIS30ANTs",
                        resolution=1,
                        desc="BrainCerebellumRegistration",
                        suffix="mask",
                    )
                ),
                compress_report=True,
            ),
            name="compressed",
            base_dir=str(tmp_path),
        )
        .run()
        .outputs.out_report
    )

    size = int(os.stat(uncompressed).st_size)
    size_compress = int(os.stat(compressed).st_size)
    assert size >= size_compress, (
        "The uncompressed report is smaller (%d)"
        "than the compressed report (%d)" % (size, size_compress)
    )
