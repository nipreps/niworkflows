import pytest
from ..itk import _applytfms
from nipype.interfaces.ants.base import Info


@pytest.mark.skipif(Info.version() is None, reason="Missing ANTs")
@pytest.mark.parametrize("ext", (".nii", ".nii.gz"))
@pytest.mark.parametrize("copy_dtype", (True, False))
@pytest.mark.parametrize("in_dtype", ("i2", "f4"))
def test_applytfms(tmpdir, ext, copy_dtype, in_dtype):
    import numpy as np
    import nibabel as nb
    from pkg_resources import resource_filename as pkgr_fn

    in_file = str(tmpdir / ("src" + ext))
    nii = nb.Nifti1Image(np.zeros((5, 5, 5), dtype=np.float32), np.eye(4))
    nii.set_data_dtype(in_dtype)
    nii.to_filename(in_file)

    in_xform = pkgr_fn("niworkflows", "data/itkIdentityTransform.txt")

    ifargs = {"copy_dtype": copy_dtype, "reference_image": in_file}
    args = (in_file, in_xform, ifargs, 0, str(tmpdir))
    out_file, cmdline = _applytfms(args)

    assert out_file == str(tmpdir / ("src_xform-%05d%s" % (0, ext)))

    out_nii = nb.load(out_file)
    assert np.allclose(nii.affine, out_nii.affine)
    assert np.allclose(nii.get_fdata(), out_nii.get_fdata())
    if copy_dtype:
        assert nii.get_data_dtype() == out_nii.get_data_dtype()
