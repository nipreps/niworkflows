"""test nibabel interfaces."""
import os
import numpy as np
import nibabel as nb
import pytest

from ..nibabel import Binarize, ApplyMask


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
    with pytest.raises(ValueError):
        ApplyMask(in_file=str(in_file), in_mask=str(in_mask), threshold=0.4).run()

    nb.Nifti1Image(mask[:-1, ...], np.eye(4), None).to_filename(str(in_mask))
    with pytest.raises(ValueError):
        ApplyMask(in_file=str(in_file), in_mask=str(in_mask), threshold=0.4).run()
    with pytest.raises(ValueError):
        ApplyMask(in_file=str(in_file4d), in_mask=str(in_mask), threshold=0.4).run()
