from pathlib import Path
import numpy as np
import nibabel as nb
from ..freesurfer import FSInjectBrainExtracted


def test_inject_skullstrip(tmp_path):
    t1_mgz = tmp_path / "sub-01" / "mri" / "T1.mgz"
    t1_mgz.parent.mkdir(parents=True)
    # T1.mgz images are uint8
    nb.MGHImage(np.ones((5, 5, 5), dtype=np.uint8), np.eye(4)).to_filename(str(t1_mgz))

    mask_nii = tmp_path / "mask.nii.gz"
    # Masks may be in a different space (and need resampling), but should be boolean,
    # or uint8 in NIfTI
    nb.Nifti1Image(np.ones((6, 6, 6), dtype=np.uint8), np.eye(4)).to_filename(
        str(mask_nii)
    )

    FSInjectBrainExtracted(
        subjects_dir=str(tmp_path), subject_id="sub-01", in_brain=str(mask_nii)
    ).run()

    assert Path.exists(tmp_path / "sub-01" / "mri" / "brainmask.auto.mgz")
    assert Path.exists(tmp_path / "sub-01" / "mri" / "brainmask.mgz")

    # Run a second time to hit "already exists" condition
    FSInjectBrainExtracted(
        subjects_dir=str(tmp_path), subject_id="sub-01", in_brain=str(mask_nii)
    ).run()
