"""test nibabel interfaces."""
import os
import numpy as np
import nibabel as nb

from ..nibabel import Binarize


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
