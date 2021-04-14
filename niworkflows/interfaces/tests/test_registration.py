import pytest
import numpy as np
import nibabel as nb
from nipype.pipeline import engine as pe
from nipype.interfaces import afni
from niworkflows.interfaces.registration import EstimateReferenceImage


@pytest.mark.parametrize("n_vols", (1, 2, 50, 51))
@pytest.mark.skipif(afni.Info.version() is None, reason="Realignment requires 3dvolreg")
def test_EstimateReferenceImage_truncation(tmp_path, n_vols):
    # Smoke test to ensure that logic for limiting loaded volumes is followed
    data = np.zeros((12, 12, 12, n_vols), dtype="f4")
    data[3:9, 3:9, 3:9, :] = 1

    in_file = str(tmp_path / "orig.nii")

    nb.Nifti1Image(data, np.eye(4)).to_filename(in_file)

    # One input
    genref = pe.Node(
        EstimateReferenceImage(in_file=in_file), name="genref", base_dir=tmp_path
    )
    # Single volume implies sbref
    if n_vols == 1:
        genref.inputs.sbref_file = in_file

    genref.run()

    # Two inputs... only sbref permits multiple files
    genref2 = pe.Node(
        EstimateReferenceImage(in_file=in_file), name="genref", base_dir=tmp_path
    )
    genref2.inputs.sbref_file = [in_file, in_file]

    genref2.run()
