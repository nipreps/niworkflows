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
"""test nibabel interfaces."""

import json
import os
import uuid
from pathlib import Path

import nibabel as nb
import numpy as np
import pytest

from ..nibabel import (
    ApplyMask,
    Binarize,
    MapLabels,
    MergeROIs,
    MergeSeries,
    ReorientImage,
    SplitSeries,
)


@pytest.fixture
def create_roi(tmp_path):
    files = []

    def _create_roi(affine, img_data, roi_index):
        img_data[tuple(roi_index)] = 1
        nii = nb.Nifti1Image(img_data, affine)
        filename = tmp_path / f'{uuid.uuid4()}.nii.gz'
        files.append(filename)
        nii.to_filename(filename)
        return filename

    yield _create_roi
    # cleanup files
    for f in files:
        f.unlink()


def create_image(data, filename):
    nb.Nifti1Image(data, affine=np.eye(4)).to_filename(str(filename))
    return filename


# create a slightly off affine
bad_affine = np.eye(4)
bad_affine[0, -1] = -1


@pytest.mark.parametrize(
    ('affine', 'data', 'roi_index', 'error', 'err_message'),
    [
        (np.eye(4), np.zeros((2, 2, 2, 2), dtype=np.uint16), [1, 0], None, None),
        (
            np.eye(4),
            np.zeros((2, 2, 3, 2), dtype=np.uint16),
            [1, 0],
            True,
            'Mismatch in image shape',
        ),
        (
            bad_affine,
            np.zeros((2, 2, 2, 2), dtype=np.uint16),
            [1, 0],
            True,
            'Mismatch in affine',
        ),
        (
            np.eye(4),
            np.zeros((2, 2, 2, 2), dtype=np.uint16),
            [0, 0, 0],
            True,
            'Overlapping ROIs',
        ),
    ],
)
def test_merge_rois(tmpdir, create_roi, affine, data, roi_index, error, err_message):
    tmpdir.chdir()
    roi0 = create_roi(np.eye(4), np.zeros((2, 2, 2, 2), dtype=np.uint16), [0, 0])
    roi1 = create_roi(np.eye(4), np.zeros((2, 2, 2, 2), dtype=np.uint16), [0, 1])
    test_roi = create_roi(affine, data, roi_index)

    merge = MergeROIs(in_files=[roi0, roi1, test_roi])
    if error is None:
        merge.run()
        return
    # otherwise check expected exceptions
    with pytest.raises(ValueError, match=r'Mismatch|Overlapping') as err:
        merge.run()
    assert err_message in str(err.value)


def test_Binarize(tmp_path):
    """Test binarization interface."""
    os.chdir(str(tmp_path))

    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[5:15, 5:15, 5:15] = bool

    data = np.zeros_like(mask, dtype='float32')
    data[mask] = np.random.gamma(2, size=mask.sum())

    in_file = tmp_path / 'input.nii.gz'
    nb.Nifti1Image(data, np.eye(4), None).to_filename(str(in_file))

    binif = Binarize(thresh_low=0.0, in_file=str(in_file)).run()
    newmask = nb.load(binif.outputs.out_mask).get_fdata().astype(bool)
    assert np.all(mask == newmask)


def test_ApplyMask(tmp_path):
    """Test masking interface."""
    os.chdir(str(tmp_path))

    data = np.ones((20, 20, 20), dtype=float)
    mask = np.zeros_like(data)

    mask[7:12, 7:12, 7:12] = 0.5
    mask[8:11, 8:11, 8:11] = 1.0

    # Test the 3D
    in_file = tmp_path / 'input3D.nii.gz'
    nb.Nifti1Image(data, np.eye(4), None).to_filename(str(in_file))

    in_mask = tmp_path / 'mask.nii.gz'
    nb.Nifti1Image(mask, np.eye(4), None).to_filename(str(in_mask))

    masked1 = ApplyMask(in_file=str(in_file), in_mask=str(in_mask), threshold=0.4).run()
    assert nb.load(masked1.outputs.out_file).get_fdata().sum() == 5**3

    masked1 = ApplyMask(in_file=str(in_file), in_mask=str(in_mask), threshold=0.6).run()
    assert nb.load(masked1.outputs.out_file).get_fdata().sum() == 3**3

    data4d = np.stack((data, 2 * data, 3 * data), axis=-1)
    # Test the 4D case
    in_file4d = tmp_path / 'input4D.nii.gz'
    nb.Nifti1Image(data4d, np.eye(4), None).to_filename(str(in_file4d))

    masked1 = ApplyMask(in_file=str(in_file4d), in_mask=str(in_mask), threshold=0.4).run()
    assert nb.load(masked1.outputs.out_file).get_fdata().sum() == 5**3 * 6

    masked1 = ApplyMask(in_file=str(in_file4d), in_mask=str(in_mask), threshold=0.6).run()
    assert nb.load(masked1.outputs.out_file).get_fdata().sum() == 3**3 * 6

    # Test errors
    nb.Nifti1Image(mask, 2 * np.eye(4), None).to_filename(str(in_mask))
    with pytest.raises(ValueError, match=r'affines are not similar'):
        ApplyMask(in_file=str(in_file), in_mask=str(in_mask), threshold=0.4).run()

    nb.Nifti1Image(mask[:-1, ...], np.eye(4), None).to_filename(str(in_mask))
    with pytest.raises(ValueError, match=r'sizes do not match'):
        ApplyMask(in_file=str(in_file), in_mask=str(in_mask), threshold=0.4).run()
    with pytest.raises(ValueError, match=r'sizes do not match'):
        ApplyMask(in_file=str(in_file4d), in_mask=str(in_mask), threshold=0.4).run()


@pytest.mark.parametrize(
    ('shape', 'exp_n'),
    [
        ((20, 20, 20, 15), 15),
        ((20, 20, 20), 1),
        ((20, 20, 20, 1), 1),
        ((20, 20, 20, 1, 3), 3),
        ((20, 20, 20, 3, 1), 3),
        ((20, 20, 20, 1, 3, 3), -1),
        ((20, 1, 20, 15), 15),
        ((20, 1, 20), 1),
        ((20, 1, 20, 1), 1),
        ((20, 1, 20, 1, 3), 3),
        ((20, 1, 20, 3, 1), 3),
        ((20, 1, 20, 1, 3, 3), -1),
    ],
)
def test_SplitSeries(tmp_path, shape, exp_n):
    """Test 4-to-3 NIfTI split interface."""
    os.chdir(tmp_path)

    in_file = str(tmp_path / 'input.nii.gz')
    nb.Nifti1Image(np.ones(shape, dtype=float), np.eye(4), None).to_filename(in_file)

    _interface = SplitSeries(in_file=in_file)
    if exp_n > 0:
        split = _interface.run()
        n = int(isinstance(split.outputs.out_files, str)) or len(split.outputs.out_files)
        assert n == exp_n
    else:
        with pytest.raises(ValueError, match=r'Invalid shape'):
            _interface.run()


def test_MergeSeries(tmp_path):
    """Test 3-to-4 NIfTI concatenation interface."""
    os.chdir(str(tmp_path))

    in_file = tmp_path / 'input3D.nii.gz'
    nb.Nifti1Image(np.ones((20, 20, 20), dtype=float), np.eye(4), None).to_filename(str(in_file))

    merge = MergeSeries(in_files=[str(in_file)] * 5).run()
    assert nb.load(merge.outputs.out_file).dataobj.shape == (20, 20, 20, 5)

    in_4D = tmp_path / 'input4D.nii.gz'
    nb.Nifti1Image(np.ones((20, 20, 20, 4), dtype=float), np.eye(4), None).to_filename(str(in_4D))

    merge = MergeSeries(in_files=[str(in_file)] + [str(in_4D)]).run()
    assert nb.load(merge.outputs.out_file).dataobj.shape == (20, 20, 20, 5)

    with pytest.raises(ValueError, match=r'incorrect number of dimensions'):
        MergeSeries(in_files=[str(in_file)] + [str(in_4D)], allow_4D=False).run()


def test_MergeSeries_affines(tmp_path):
    os.chdir(str(tmp_path))

    files = ['img0.nii.gz', 'img1.nii.gz']
    data = np.ones((10, 10, 10), dtype=np.uint16)
    aff = np.eye(4)
    nb.Nifti1Image(data, aff, None).to_filename(files[0])
    # slightly alter affine
    aff[0][0] = 1.00005
    nb.Nifti1Image(data, aff, None).to_filename(files[1])

    # affine mismatch will cause this to fail
    with pytest.raises(ValueError, match=r'does not match affine'):
        MergeSeries(in_files=files).run()
    # but works if we set a tolerance
    MergeSeries(in_files=files, affine_tolerance=1e-04).run()


LABEL_MAPPINGS = {5: 1, 6: 1, 7: 2}
LABEL_INPUT = np.arange(8, dtype=np.uint16).reshape(2, 2, 2)
LABEL_OUTPUT = np.asarray([0, 1, 2, 3, 4, 1, 1, 2]).reshape(2, 2, 2)


@pytest.mark.parametrize(
    ('data', 'mapping', 'tojson', 'expected'),
    [
        (LABEL_INPUT, LABEL_MAPPINGS, False, LABEL_OUTPUT),
        (LABEL_INPUT, LABEL_MAPPINGS, True, LABEL_OUTPUT),
    ],
)
def test_map_labels(tmpdir, data, mapping, tojson, expected):
    tmpdir.chdir()
    in_file = create_image(data, Path('test.nii.gz'))
    maplbl = MapLabels(in_file=in_file)
    if tojson:
        map_file = Path('mapping.json')
        map_file.write_text(json.dumps(mapping))
        maplbl.inputs.mappings_file = map_file
    else:
        maplbl.inputs.mappings = mapping
    out_file = maplbl.run().outputs.out_file

    orig = nb.load(in_file).get_fdata()
    labels = nb.load(out_file).get_fdata()
    assert orig.shape == labels.shape
    assert np.all(labels == expected)

    Path(in_file).unlink()
    if tojson:
        Path(map_file).unlink()


def create_save_img(ornt: str):
    data = np.random.rand(2, 2, 2)
    img = nb.Nifti1Image(data, affine=np.eye(4))
    # img will always be in RAS at the start
    ras = nb.orientations.axcodes2ornt('RAS')
    if ornt != 'RAS':
        new = nb.orientations.axcodes2ornt(ornt)
        xfm = nb.orientations.ornt_transform(ras, new)
        img = img.as_reoriented(xfm)
    out_file = f'{uuid.uuid4()}.nii.gz'
    img.to_filename(out_file)
    return out_file


@pytest.mark.parametrize(
    ('in_ornt', 'out_ornt'),
    [
        ('RAS', 'RAS'),
        ('RAS', 'LAS'),
        ('LAS', 'RAS'),
        ('RAS', 'RPI'),
        ('LPI', 'RAS'),
    ],
)
def test_reorient_image(tmpdir, in_ornt, out_ornt):
    tmpdir.chdir()

    in_file = create_save_img(ornt=in_ornt)
    in_img = nb.load(in_file)
    assert ''.join(nb.aff2axcodes(in_img.affine)) == in_ornt

    # test string representation
    res = ReorientImage(in_file=in_file, target_orientation=out_ornt).run()
    out_file = res.outputs.out_file
    out_img = nb.load(out_file)
    assert ''.join(nb.aff2axcodes(out_img.affine)) == out_ornt
    Path(out_file).unlink()

    # test with target file
    target_file = create_save_img(ornt=out_ornt)
    target_img = nb.load(target_file)
    assert ''.join(nb.aff2axcodes(target_img.affine)) == out_ornt
    res = ReorientImage(in_file=in_file, target_file=target_file).run()
    out_file = res.outputs.out_file
    out_img = nb.load(out_file)
    assert ''.join(nb.aff2axcodes(out_img.affine)) == out_ornt

    # cleanup
    for f in (in_file, target_file, out_file):
        Path(f).unlink()
