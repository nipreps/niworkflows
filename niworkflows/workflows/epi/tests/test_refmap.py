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
"""Check the refmap module."""

import os
import unittest

from ....testing import has_afni
from ..refmap import init_epi_reference_wf


@unittest.skipUnless(has_afni, 'Needs AFNI')
def test_reference(tmpdir, ds000030_dir, workdir, outdir):
    """Exercise the EPI reference workflow."""
    tmpdir.chdir()

    wf = init_epi_reference_wf(omp_nthreads=os.cpu_count(), auto_bold_nss=True)
    if workdir:
        wf.base_dir = str(workdir)

    wf.inputs.inputnode.in_files = [
        str(f) for f in (ds000030_dir / 'sub-10228' / 'func').glob('*_bold.nii.gz')
    ]

    # if outdir:
    #     out_path = outdir / "masks" / folder.split("/")[-1]
    #     out_path.mkdir(exist_ok=True, parents=True)
    #     report = pe.Node(SimpleShowMaskRPT(), name="report")
    #     report.interface._always_run = True

    #     def _report_name(fname, out_path):
    #         from pathlib import Path

    #         return str(
    #             out_path
    #             / Path(fname)
    #             .name.replace(".nii", "_mask.svg")
    #             .replace("_magnitude", "_desc-magnitude")
    #             .replace(".gz", "")
    #         )

    #     # fmt: off
    #     wf.connect([
    #         (inputnode, report, [(("in_file", _report_name, out_path), "out_report")]),
    #         (brainmask_wf, report, [("outputnode.out_mask", "mask_file"),
    #                                 ("outputnode.out_file", "background_file")]),
    #     ])
    #     # fmt: on

    wf.run()
