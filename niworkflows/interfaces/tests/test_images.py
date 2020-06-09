import time
import numpy as np
import nibabel as nb
from nipype.pipeline import engine as pe
from nipype.interfaces import nilearn as nl
from .. import images as im
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "qform_add, sform_add, expectation",
    [
        (0, 0, "no_warn"),
        (0, 1e-14, "no_warn"),
        (0, 1e-09, "no_warn"),
        (1e-6, 0, "warn"),
        (0, 1e-6, "warn"),
        (1e-5, 0, "warn"),
        (0, 1e-5, "warn"),
        (1e-3, 1e-3, "no_warn"),
    ],
)
# just a diagonal of ones in qform and sform and see that this doesn't warn
# only look at the 2 areas of images.py that I added and get code coverage of those
def test_qformsform_warning(tmp_path, qform_add, sform_add, expectation):
    fname = str(tmp_path / "test.nii")

    # make a random image
    random_data = np.random.random(size=(5, 5, 5) + (5,))
    img = nb.Nifti1Image(random_data, np.eye(4) + sform_add)
    # set the qform of the image before calling it
    img.set_qform(np.eye(4) + qform_add)
    img.to_filename(fname)

    validate = pe.Node(im.ValidateImage(), name="validate", base_dir=str(tmp_path))
    validate.inputs.in_file = fname
    res = validate.run()
    out_report = Path(res.outputs.out_report).read_text()
    if expectation == "warn":
        assert "Note on" in out_report
    elif expectation == "no_warn":
        assert len(out_report) == 0


@pytest.mark.parametrize(
    "qform_code, warning_text",
    [(0, "Note on orientation"), (1, "WARNING - Invalid qform")],
)
def test_bad_qform(tmp_path, qform_code, warning_text):
    fname = str(tmp_path / "test.nii")

    # make a random image
    random_data = np.random.random(size=(5, 5, 5) + (5,))
    img = nb.Nifti1Image(random_data, np.eye(4))

    # Some magic terms from a bad qform in the wild
    img.header["qform_code"] = qform_code
    img.header["quatern_b"] = 0
    img.header["quatern_c"] = 0.998322
    img.header["quatern_d"] = -0.0579125
    img.to_filename(fname)

    validate = pe.Node(im.ValidateImage(), name="validate", base_dir=str(tmp_path))
    validate.inputs.in_file = fname
    res = validate.run()
    assert warning_text in Path(res.outputs.out_report).read_text()


def test_no_good_affines(tmp_path):
    fname = str(tmp_path / "test.nii")

    # make a random image
    random_data = np.random.random(size=(5, 5, 5) + (5,))
    img = nb.Nifti1Image(random_data, None)
    img.header["qform_code"] = 0
    img.header["sform_code"] = 0
    img.to_filename(fname)

    validate = pe.Node(im.ValidateImage(), name="validate", base_dir=str(tmp_path))
    validate.inputs.in_file = fname
    res = validate.run()
    assert (
        "WARNING - Missing orientation information"
        in Path(res.outputs.out_report).read_text()
    )


@pytest.mark.parametrize(
    "nvols, nmasks, ext, factor",
    [
        (500, 10, ".nii", 2),
        (500, 10, ".nii.gz", 5),
        (200, 3, ".nii", 1.1),
        (200, 3, ".nii.gz", 2),
        (200, 10, ".nii", 1.1),
        (200, 10, ".nii.gz", 2),
    ],
)
def test_signal_extraction_equivalence(tmp_path, nvols, nmasks, ext, factor):
    nlsignals = str(tmp_path / "nlsignals.tsv")
    imsignals = str(tmp_path / "imsignals.tsv")

    vol_shape = (64, 64, 40)

    img_fname = str(tmp_path / ("img" + ext))
    masks_fname = str(tmp_path / ("masks" + ext))

    random_data = np.random.random(size=vol_shape + (nvols,)) * 2000
    random_mask_data = np.random.random(size=vol_shape + (nmasks,)) < 0.2

    nb.Nifti1Image(random_data, np.eye(4)).to_filename(img_fname)
    nb.Nifti1Image(random_mask_data.astype(np.uint8), np.eye(4)).to_filename(
        masks_fname
    )

    se1 = nl.SignalExtraction(
        in_file=img_fname,
        label_files=masks_fname,
        class_labels=["a%d" % i for i in range(nmasks)],
        out_file=nlsignals,
    )
    se2 = im.SignalExtraction(
        in_file=img_fname,
        label_files=masks_fname,
        class_labels=["a%d" % i for i in range(nmasks)],
        out_file=imsignals,
    )

    tic = time.time()
    se1.run()
    toc = time.time()
    se2.run()
    toc2 = time.time()

    tab1 = np.loadtxt(nlsignals, skiprows=1)
    tab2 = np.loadtxt(imsignals, skiprows=1)

    assert np.allclose(tab1, tab2)

    t1 = toc - tic
    t2 = toc2 - toc

    assert t2 < t1 / factor


@pytest.mark.parametrize(
    "shape, mshape",
    [
        ((10, 10, 10), (10, 10, 10)),
        ((10, 10, 10, 1), (10, 10, 10)),
        ((10, 10, 10, 1, 1), (10, 10, 10)),
        ((10, 10, 10, 2), (10, 10, 10, 2)),
        ((10, 10, 10, 2, 1), (10, 10, 10, 2)),
        ((10, 10, 10, 2, 2), None),
    ],
)
def test_IntraModalMerge(tmpdir, shape, mshape):
    """Exercise the various types of inputs."""
    tmpdir.chdir()

    data = np.random.normal(size=shape).astype("float32")
    fname = str(tmpdir.join("file1.nii.gz"))
    nb.Nifti1Image(data, np.eye(4), None).to_filename(fname)

    if mshape is None:
        with pytest.raises(RuntimeError):
            im.IntraModalMerge(in_files=fname).run()
        return

    merged = str(im.IntraModalMerge(in_files=fname).run().outputs.out_file)
    merged_data = nb.load(merged).get_fdata(dtype="float32")
    assert merged_data.shape == mshape
    assert np.allclose(np.squeeze(data), merged_data)

    merged = str(
        im.IntraModalMerge(in_files=[fname, fname], hmc=False).run().outputs.out_file
    )
    merged_data = nb.load(merged).get_fdata(dtype="float32")
    new_mshape = (*mshape[:3], 2 if len(mshape) == 3 else mshape[3] * 2)
    assert merged_data.shape == new_mshape


def test_conform_resize(tmpdir):
    fname = str(tmpdir / "test.nii")

    random_data = np.random.random(size=(5, 5, 5))
    img = nb.Nifti1Image(random_data, np.eye(4))
    img.to_filename(fname)
    conform = pe.Node(im.Conform(), name="conform", base_dir=str(tmpdir))
    conform.inputs.in_file = fname
    conform.inputs.target_zooms = (1, 1, 1.5)
    conform.inputs.target_shape = (5, 5, 5)
    res = conform.run()

    out_img = nb.load(res.outputs.out_file)
    assert out_img.header.get_zooms() == conform.inputs.target_zooms


def test_conform_set_zooms(tmpdir):
    fname = str(tmpdir / "test.nii")

    random_data = np.random.random(size=(5, 5, 5))
    img = nb.Nifti1Image(random_data, np.eye(4))
    img.to_filename(fname)
    conform = pe.Node(im.Conform(), name="conform", base_dir=str(tmpdir))
    conform.inputs.in_file = fname
    conform.inputs.target_zooms = (1, 1, 1.002)
    conform.inputs.target_shape = (5, 5, 5)
    res = conform.run()

    out_img = nb.load(res.outputs.out_file)
    assert np.allclose(out_img.header.get_zooms(), conform.inputs.target_zooms)
