import nibabel as nb
import numpy as np

import pytest
from ..images import update_header_fields, overwrite_header


def random_image():
    return nb.Nifti1Image(np.random.random((5, 5, 5, 5)), np.eye(4))


@pytest.mark.parametrize("fields", [
    {},
    {'intent_code': 0},
    {'intent_code': 0, 'sform_code': 4},
    {'sform_code': 3},
    # Changes to these fields have no effect
    {'scl_slope': 3., 'scl_inter': 3.},
    {'vox_offset': 20.},
    ])
@pytest.mark.parametrize("slope, inter", [
    (None, None),
    (1., 0.),
    (2., 2.)
    ])
def test_update_header_fields(tmp_path, fields, slope, inter):
    fname = str(tmp_path / 'test_file.nii')

    # Generate file
    init_img = random_image()
    init_img.header.set_slope_inter(slope, inter)
    init_img.to_filename(fname)

    # Reference load
    pre_img = nb.load(fname, mmap=False)
    pre_data = pre_img.get_fdata()

    update_header_fields(fname, **fields)

    # Post-rewrite load
    post_img = nb.load(fname)

    # Data should be identical
    assert np.array_equal(pre_data, post_img.get_fdata())


@pytest.mark.parametrize("fields", [
    {'datatype': 2},
    ])
@pytest.mark.parametrize("slope, inter", [
    (None, None),
    (2., 2.)
    ])
def test_update_header_fields_exceptions(tmp_path, fields, slope, inter):
    fname = str(tmp_path / 'test_file.nii')

    # Generate file
    img = random_image()
    img.header.set_slope_inter(slope, inter)
    img.to_filename(fname)

    with pytest.raises(ValueError):
        update_header_fields(fname, **fields)


def test_overwrite_header_reject_mmap(tmp_path):
    fname = str(tmp_path / 'test_file.nii')

    random_image().to_filename(fname)

    img = nb.load(fname, mmap=True)
    with pytest.raises(ValueError):
        overwrite_header(img, fname)
