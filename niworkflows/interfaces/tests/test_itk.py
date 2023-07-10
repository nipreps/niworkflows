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
import pytest
from ..itk import _applytfms
from ... import data
from nipype.interfaces.ants.base import Info


@pytest.mark.skipif(Info.version() is None, reason="Missing ANTs")
@pytest.mark.parametrize("ext", (".nii", ".nii.gz"))
@pytest.mark.parametrize("copy_dtype", (True, False))
@pytest.mark.parametrize("in_dtype", ("i2", "f4"))
def test_applytfms(tmpdir, ext, copy_dtype, in_dtype):
    import numpy as np
    import nibabel as nb

    in_file = str(tmpdir / ("src" + ext))
    nii = nb.Nifti1Image(np.zeros((5, 5, 5), dtype=np.float32), np.eye(4))
    nii.set_data_dtype(in_dtype)
    nii.to_filename(in_file)

    in_xform = data.load("itkIdentityTransform.txt")

    ifargs = {"copy_dtype": copy_dtype, "reference_image": in_file}
    args = (in_file, in_xform, ifargs, 0, str(tmpdir))
    out_file, cmdline = _applytfms(args)

    assert out_file == str(tmpdir / ("src_xform-%05d%s" % (0, ext)))

    out_nii = nb.load(out_file)
    assert np.allclose(nii.affine, out_nii.affine)
    assert np.allclose(nii.get_fdata(), out_nii.get_fdata())
    if copy_dtype:
        assert nii.get_data_dtype() == out_nii.get_data_dtype()
