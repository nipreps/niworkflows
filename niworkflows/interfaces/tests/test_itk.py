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
import re
from pathlib import Path

import numpy as np
import pytest
from nipype.interfaces.ants.base import Info
from nipype.pipeline import engine as pe

from ... import data
from ..itk import MCFLIRT2ITK, _applytfms
from .data import load_test_data


@pytest.mark.skipif(Info.version() is None, reason='Missing ANTs')
@pytest.mark.parametrize('ext', ['.nii', '.nii.gz'])
@pytest.mark.parametrize('copy_dtype', [True, False])
@pytest.mark.parametrize('in_dtype', ['i2', 'f4'])
def test_applytfms(tmpdir, ext, copy_dtype, in_dtype):
    import nibabel as nb
    import numpy as np

    in_file = str(tmpdir / ('src' + ext))
    nii = nb.Nifti1Image(np.zeros((5, 5, 5), dtype=np.float32), np.eye(4))
    nii.set_data_dtype(in_dtype)
    nii.to_filename(in_file)

    in_xform = data.load('itkIdentityTransform.txt')

    ifargs = {'copy_dtype': copy_dtype, 'reference_image': in_file}
    args = (in_file, in_xform, ifargs, 0, str(tmpdir))
    out_file, _cmdline = _applytfms(args)

    assert out_file == str(tmpdir / (f'src_xform-00000{ext}'))

    out_nii = nb.load(out_file)
    assert np.allclose(nii.affine, out_nii.affine)
    assert np.allclose(nii.get_fdata(), out_nii.get_fdata())
    if copy_dtype:
        assert nii.get_data_dtype() == out_nii.get_data_dtype()


def test_MCFLIRT2ITK(tmp_path):
    # Test that MCFLIRT2ITK produces output that is consistent with convert3d
    test_data = load_test_data()

    fsl2itk = pe.Node(
        MCFLIRT2ITK(
            in_files=[str(test_data / 'MAT_0098'), str(test_data / 'MAT_0099')],
            in_reference=str(test_data / 'boldref.nii'),
            in_source=str(test_data / 'boldref.nii'),
        ),
        name='fsl2itk',
        base_dir=str(tmp_path),
    )

    res = fsl2itk.run()
    out_file = Path(res.outputs.out_file)

    assert out_file.exists()
    lines = out_file.read_text().splitlines()

    assert lines[:2] == [
        '#Insight Transform File V1.0',
        '#Transform 0',
    ]
    assert re.match(
        r'Transform: (MatrixOffsetTransformBase|AffineTransform)_(float|double)_3_3',
        lines[2],
    )
    assert lines[3].startswith('Parameters: ')
    assert lines[4] == 'FixedParameters: 0 0 0'
    offset = 1 if lines[5] == '' else 0
    assert lines[5 + offset] == '#Transform 1'
    assert lines[6 + offset] == lines[2]
    assert lines[7 + offset].startswith('Parameters: ')

    params0 = np.array([float(p) for p in lines[3].split(' ')[1:]])
    params1 = np.array([float(p) for p in lines[7 + offset].split(' ')[1:]])
    # Empirically determined
    assert np.allclose(
        params0,
        np.array(
            [
                9.99998489e-01,
                -4.36657508e-04,
                -1.52316526e-03,
                4.36017740e-04,
                9.99999777e-01,
                -2.10558666e-04,
                1.52334852e-03,
                2.09440681e-04,
                9.99998624e-01,
                -1.28961869e-03,
                6.93155516e-02,
                -1.12375673e-02,
            ]
        ),
    )
    assert np.allclose(
        params1,
        np.array(
            [
                9.99999130e-01,
                -4.60021530e-04,
                -1.28828576e-03,
                4.58910652e-04,
                9.99999648e-01,
                -3.90485877e-04,
                1.28764980e-03,
                3.89513646e-04,
                9.99999178e-01,
                -9.19541650e-03,
                7.45419094e-02,
                -8.95843238e-03,
            ]
        ),
    )
