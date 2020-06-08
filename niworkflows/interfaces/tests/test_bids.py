"""Tests on BIDS compliance."""
import os
from pathlib import Path

import numpy as np
import nibabel as nb
import pytest
from nipype.interfaces.base import Undefined

from .. import bids as bintfs


XFORM_CODES = {
    "MNI152Lin": 4,
    "T1w": 2,
    "boldref": 2,
    None: 1,
}

T1W_PATH = "ds054/sub-100185/anat/sub-100185_T1w.nii.gz"
BOLD_PATH = "ds054/sub-100185/func/sub-100185_task-machinegame_run-01_bold.nii.gz"


@pytest.mark.parametrize("out_path_base", [None, "fmriprep"])
@pytest.mark.parametrize(
    "source,input_files,entities,expectation",
    [
        (
            T1W_PATH,
            ["anat.nii.gz"],
            {"desc": "preproc"},
            "sub-100185/anat/sub-100185_desc-preproc_T1w.nii.gz",
        ),
        (
            T1W_PATH,
            ["anat.nii.gz"],
            {"desc": "preproc", "space": "MNI"},
            "sub-100185/anat/sub-100185_space-MNI_desc-preproc_T1w.nii.gz",
        ),
        (
            T1W_PATH,
            ["anat.nii.gz"],
            {"desc": "preproc", "space": "MNI", "resolution": "native"},
            "sub-100185/anat/sub-100185_space-MNI_desc-preproc_T1w.nii.gz",
        ),
        (
            T1W_PATH,
            ["anat.nii.gz"],
            {"desc": "preproc", "space": "MNI", "resolution": "high"},
            "sub-100185/anat/sub-100185_space-MNI_res-high_desc-preproc_T1w.nii.gz",
        ),
        (
            T1W_PATH,
            ["tfm.txt"],
            {"from": "fsnative", "to": "T1w", "suffix": "xfm"},
            "sub-100185/anat/sub-100185_from-fsnative_to-T1w_mode-image_xfm.txt",
        ),
        (
            T1W_PATH,
            ["tfm.h5"],
            {"from": "MNI152NLin2009cAsym", "to": "T1w", "suffix": "xfm"},
            "sub-100185/anat/sub-100185_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5",
        ),
        (
            T1W_PATH,
            ["anat.nii.gz"],
            {"desc": "brain", "suffix": "mask"},
            "sub-100185/anat/sub-100185_desc-brain_mask.nii.gz",
        ),
        (
            T1W_PATH,
            ["anat.nii.gz"],
            {"desc": "brain", "suffix": "mask", "space": "MNI"},
            "sub-100185/anat/sub-100185_space-MNI_desc-brain_mask.nii.gz",
        ),
        (
            T1W_PATH,
            ["anat.surf.gii"],
            {"suffix": "pial", "hemi": "L"},
            "sub-100185/anat/sub-100185_hemi-L_pial.surf.gii",
        ),
        (
            T1W_PATH,
            ["aseg.nii", "aparc.nii"],
            {"desc": ["aseg", "aparcaseg"], "suffix": "dseg"},
            [
                f"sub-100185/anat/sub-100185_desc-{s}_dseg.nii"
                for s in ("aseg", "aparcaseg")
            ],
        ),
        (
            T1W_PATH,
            ["anat.nii", "anat.json"],
            {"desc": "preproc"},
            [
                f"sub-100185/anat/sub-100185_desc-preproc_T1w.{ext}"
                for ext in ("nii", "json")
            ],
        ),
        (
            T1W_PATH,
            ["anat.nii.gz"] * 3,
            {"label": ["GM", "WM", "CSF"], "suffix": "probseg"},
            [
                f"sub-100185/anat/sub-100185_label-{lab}_probseg.nii.gz"
                for lab in ("GM", "WM", "CSF")
            ],
        ),
        # BOLD data
        (
            BOLD_PATH,
            ["aroma.csv"],
            {"suffix": "AROMAnoiseICs"},
            "sub-100185/func/sub-100185_task-machinegame_run-1_AROMAnoiseICs.csv",
        ),
        (
            BOLD_PATH,
            ["confounds.tsv"],
            {"suffix": "regressors", "desc": "confounds"},
            "sub-100185/func/sub-100185_task-machinegame_run-1_desc-confounds_regressors.tsv",
        ),
        (
            BOLD_PATH,
            ["mixing.tsv"],
            {"suffix": "mixing", "desc": "MELODIC"},
            "sub-100185/func/sub-100185_task-machinegame_run-1_desc-MELODIC_mixing.tsv",
        ),
        (
            BOLD_PATH,
            ["lh.func.gii"],
            {"space": "fsaverage", "density": "10k", "hemi": "L"},
            "sub-100185/func/sub-100185_task-machinegame_run-1_"
            "space-fsaverage_den-10k_hemi-L_bold.func.gii",
        ),
        (
            BOLD_PATH,
            ["hcp.dtseries.nii"],
            {"space": "fsLR", "density": "91k"},
            "sub-100185/func/sub-100185_task-machinegame_run-1_"
            "space-fsLR_den-91k_bold.dtseries.nii",
        ),
        (
            BOLD_PATH,
            ["ref.nii"],
            {"space": "MNI", "suffix": "boldref"},
            "sub-100185/func/sub-100185_task-machinegame_run-1_space-MNI_boldref.nii",
        ),
        (
            BOLD_PATH,
            ["dseg.nii"],
            {"space": "MNI", "suffix": "dseg", "desc": "aseg"},
            "sub-100185/func/sub-100185_task-machinegame_run-1_space-MNI_desc-aseg_dseg.nii",
        ),
        (
            BOLD_PATH,
            ["mask.nii"],
            {"space": "MNI", "suffix": "mask", "desc": "brain"},
            "sub-100185/func/sub-100185_task-machinegame_run-1_space-MNI_desc-brain_mask.nii",
        ),
        (
            BOLD_PATH,
            ["bold.nii"],
            {"space": "MNI", "desc": "preproc"},
            "sub-100185/func/sub-100185_task-machinegame_run-1_space-MNI_desc-preproc_bold.nii",
        ),
        # Nondeterministic order - do we really need this to work, or we can stay safe with
        # MapNodes?
        # (T1W_PATH, [f"{s}-{l}.nii.gz" for s in ("MNIa", "MNIb") for l in ("GM", "WM", "CSF")],
        #     {"space": ["MNIa", "MNIb"], "label": ["GM", "WM", "CSF"], "suffix": "probseg"},
        #     [f"sub-100185/anat/sub-100185_space-{s}_label-{l}_probseg.nii.gz"
        #      for s in ("MNIa", "MNIb") for l in ("GM", "WM", "CSF")]),
        (
            T1W_PATH,
            ["anat.html"],
            {"desc": "conform", "datatype": "figures"},
            "sub-100185/figures/sub-100185_desc-conform_T1w.html",
        ),
        (
            BOLD_PATH,
            ["aroma.csv"],
            {"suffix": "AROMAnoiseICs", "extension": "h5"},
            ValueError,
        ),
        (
            T1W_PATH,
            ["anat.nii.gz"] * 3,
            {"desc": "preproc", "space": "MNI"},
            ValueError,
        ),
        (
            "sub-07/ses-preop/anat/sub-07_ses-preop_T1w.nii.gz",
            ["tfm.h5"],
            {"from": "orig", "to": "target", "suffix": "xfm"},
            "sub-07/ses-preop/anat/sub-07_ses-preop_from-orig_to-target_mode-image_xfm.h5",
        ),
        (
            "sub-07/ses-preop/anat/sub-07_ses-preop_run-1_T1w.nii.gz",
            ["tfm.txt"],
            {"from": "orig", "to": "T1w", "suffix": "xfm"},
            "sub-07/ses-preop/anat/sub-07_ses-preop_run-1_from-orig_to-T1w_mode-image_xfm.txt",
        ),
    ],
)
@pytest.mark.parametrize("dismiss_entities", [None, ("run", "session")])
def test_DerivativesDataSink_build_path(
    tmp_path,
    out_path_base,
    source,
    input_files,
    entities,
    expectation,
    dismiss_entities,
):
    """Check a few common derivatives generated by NiPreps."""
    ds_inputs = []
    for input_file in input_files:
        fname = tmp_path / input_file
        if fname.name.rstrip(".gz").endswith(".nii"):
            hdr = nb.Nifti1Header()
            hdr.set_qform(np.eye(4), code=2)
            hdr.set_sform(np.eye(4), code=2)
            units = ("mm", "sec") if "bold" in input_file else ("mm",)
            size = (10, 10, 10, 10) if "bold" in input_file else (10, 10, 10)
            hdr.set_xyzt_units(*units)
            nb.Nifti1Image(np.zeros(size), np.eye(4), hdr).to_filename(fname)
        else:
            (tmp_path / input_file).write_text("")

        ds_inputs.append(str(fname))

    dds = bintfs.DerivativesDataSink(
        in_file=ds_inputs,
        base_directory=str(tmp_path),
        source_file=source,
        out_path_base=out_path_base,
        dismiss_entities=dismiss_entities,
        **entities,
    )

    if type(expectation) == type(Exception):
        with pytest.raises(expectation):
            dds.run()
        return

    output = dds.run().outputs.out_file
    if isinstance(expectation, str):
        expectation = [expectation]
        output = [output]

    if dismiss_entities:
        if "run" in dismiss_entities:
            expectation = [e.replace("_run-1", "") for e in expectation]

        if "session" in dismiss_entities:
            expectation = [
                e.replace("_ses-preop", "").replace("ses-preop/", "")
                for e in expectation
            ]

    base = out_path_base or "niworkflows"
    for out, exp in zip(output, expectation):
        assert Path(out).relative_to(tmp_path) == Path(base) / exp

    os.chdir(str(tmp_path))  # Exercise without setting base_directory
    dds = bintfs.DerivativesDataSink(
        in_file=ds_inputs,
        dismiss_entities=dismiss_entities,
        source_file=source,
        out_path_base=out_path_base,
        **entities,
    )

    output = dds.run().outputs.out_file
    if isinstance(output, str):
        output = [output]

    for out, exp in zip(output, expectation):
        assert Path(out).relative_to(tmp_path) == Path(base) / exp


@pytest.mark.parametrize(
    "space, size, units, xcodes, zipped, fixed, data_dtype",
    [
        ("T1w", (30, 30, 30, 10), ("mm", "sec"), (2, 2), True, [False], None),
        ("T1w", (30, 30, 30, 10), ("mm", "sec"), (0, 2), True, [True], "float64"),
        ("T1w", (30, 30, 30, 10), ("mm", "sec"), (0, 0), True, [True], "<i4"),
        ("T1w", (30, 30, 30, 10), ("mm", None), (2, 2), True, [True], "<f4"),
        ("T1w", (30, 30, 30, 10), (None, None), (0, 2), True, [True], None),
        ("T1w", (30, 30, 30, 10), (None, "sec"), (0, 0), True, [True], None),
        ("MNI152Lin", (30, 30, 30, 10), ("mm", "sec"), (4, 4), True, [False], None),
        ("MNI152Lin", (30, 30, 30, 10), ("mm", "sec"), (0, 2), True, [True], None),
        ("MNI152Lin", (30, 30, 30, 10), ("mm", "sec"), (0, 0), True, [True], None),
        ("MNI152Lin", (30, 30, 30, 10), ("mm", None), (4, 4), True, [True], None),
        ("MNI152Lin", (30, 30, 30, 10), (None, None), (0, 2), True, [True], None),
        ("MNI152Lin", (30, 30, 30, 10), (None, "sec"), (0, 0), True, [True], None),
        (None, (30, 30, 30, 10), ("mm", "sec"), (1, 1), True, [False], None),
        (None, (30, 30, 30, 10), ("mm", "sec"), (0, 0), True, [True], None),
        (None, (30, 30, 30, 10), ("mm", "sec"), (0, 2), True, [True], None),
        (None, (30, 30, 30, 10), ("mm", None), (1, 1), True, [True], None),
        (None, (30, 30, 30, 10), (None, None), (0, 2), True, [True], None),
        (None, (30, 30, 30, 10), (None, "sec"), (0, 0), True, [True], None),
        (None, (30, 30, 30, 10), (None, "sec"), (0, 0), False, [True], None),
    ],
)
def test_DerivativesDataSink_bold(
    tmp_path, space, size, units, xcodes, zipped, fixed, data_dtype
):
    fname = str(tmp_path / "source.nii") + (".gz" if zipped else "")

    hdr = nb.Nifti1Header()
    hdr.set_qform(np.eye(4), code=xcodes[0])
    hdr.set_sform(np.eye(4), code=xcodes[1])
    hdr.set_xyzt_units(*units)
    nb.Nifti1Image(np.zeros(size), np.eye(4), hdr).to_filename(fname)

    # BOLD derivative in T1w space
    dds = bintfs.DerivativesDataSink(
        base_directory=str(tmp_path),
        keep_dtype=True,
        data_dtype=data_dtype or Undefined,
        desc="preproc",
        source_file=BOLD_PATH,
        space=space or Undefined,
        in_file=fname,
    ).run()

    nii = nb.load(dds.outputs.out_file)
    assert dds.outputs.fixed_hdr == fixed
    if data_dtype:
        assert nii.get_data_dtype() == np.dtype(data_dtype)
    assert int(nii.header["qform_code"]) == XFORM_CODES[space]
    assert int(nii.header["sform_code"]) == XFORM_CODES[space]
    assert nii.header.get_xyzt_units() == ("mm", "sec")


@pytest.mark.parametrize(
    "space, size, units, xcodes, fixed",
    [
        ("MNI152Lin", (30, 30, 30), ("mm", None), (4, 4), [False]),
        ("MNI152Lin", (30, 30, 30), ("mm", "sec"), (4, 4), [True]),
        ("MNI152Lin", (30, 30, 30), ("mm", "sec"), (0, 2), [True]),
        ("MNI152Lin", (30, 30, 30), ("mm", "sec"), (0, 0), [True]),
        ("MNI152Lin", (30, 30, 30), (None, None), (0, 2), [True]),
        ("MNI152Lin", (30, 30, 30), (None, "sec"), (0, 0), [True]),
        ("boldref", (30, 30, 30), ("mm", None), (2, 2), [False]),
        ("boldref", (30, 30, 30), ("mm", "sec"), (2, 2), [True]),
        ("boldref", (30, 30, 30), ("mm", "sec"), (0, 2), [True]),
        ("boldref", (30, 30, 30), ("mm", "sec"), (0, 0), [True]),
        ("boldref", (30, 30, 30), (None, None), (0, 2), [True]),
        ("boldref", (30, 30, 30), (None, "sec"), (0, 0), [True]),
        (None, (30, 30, 30), ("mm", None), (1, 1), [False]),
        (None, (30, 30, 30), ("mm", "sec"), (1, 1), [True]),
        (None, (30, 30, 30), ("mm", "sec"), (0, 2), [True]),
        (None, (30, 30, 30), ("mm", "sec"), (0, 0), [True]),
        (None, (30, 30, 30), (None, None), (0, 2), [True]),
        (None, (30, 30, 30), (None, "sec"), (0, 0), [True]),
    ],
)
def test_DerivativesDataSink_t1w(tmp_path, space, size, units, xcodes, fixed):
    fname = str(tmp_path / "source.nii.gz")

    hdr = nb.Nifti1Header()
    hdr.set_qform(np.eye(4), code=xcodes[0])
    hdr.set_sform(np.eye(4), code=xcodes[1])
    hdr.set_xyzt_units(*units)
    nb.Nifti1Image(np.zeros(size), np.eye(4), hdr).to_filename(fname)

    # BOLD derivative in T1w space
    dds = bintfs.DerivativesDataSink(
        base_directory=str(tmp_path),
        keep_dtype=True,
        desc="preproc",
        source_file=T1W_PATH,
        space=space or Undefined,
        in_file=fname,
    ).run()

    nii = nb.load(dds.outputs.out_file)
    assert dds.outputs.fixed_hdr == fixed
    assert int(nii.header["qform_code"]) == XFORM_CODES[space]
    assert int(nii.header["sform_code"]) == XFORM_CODES[space]
    assert nii.header.get_xyzt_units() == ("mm", "unknown")


@pytest.mark.parametrize("field", ["RepetitionTime", "UndefinedField"])
def test_ReadSidecarJSON_connection(testdata_dir, field):
    """
    This test prevents regressions of #333
    """
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from niworkflows.interfaces.bids import ReadSidecarJSON

    reg_fields = ["RepetitionTime"]
    n = pe.Node(ReadSidecarJSON(fields=reg_fields), name="node")
    n.inputs.in_file = str(
        testdata_dir / "ds054" / "sub-100185" / "fmap" / "sub-100185_phasediff.nii.gz"
    )
    o = pe.Node(niu.IdentityInterface(fields=["out_port"]), name="o")
    wf = pe.Workflow(name="json")

    if field in reg_fields:  # This should work
        wf.connect([(n, o, [(field, "out_port")])])
    else:
        with pytest.raises(Exception, match=r".*Some connections were not found.*"):
            wf.connect([(n, o, [(field, "out_port")])])


@pytest.mark.skipif(not os.getenv("FREESURFER_HOME"), reason="No FreeSurfer")
@pytest.mark.parametrize(
    "derivatives, subjects_dir",
    [
        (os.getenv("FREESURFER_HOME"), "subjects"),
        ("/tmp", "%s/%s" % (os.getenv("FREESURFER_HOME"), "subjects")),
    ],
)
def test_fsdir_noaction(derivatives, subjects_dir):
    """ Using $FREESURFER_HOME/subjects should exit early, however constructed """
    fshome = os.environ["FREESURFER_HOME"]
    res = bintfs.BIDSFreeSurferDir(
        derivatives=derivatives, subjects_dir=subjects_dir, freesurfer_home=fshome
    ).run()
    assert res.outputs.subjects_dir == "%s/subjects" % fshome


@pytest.mark.skipif(not os.getenv("FREESURFER_HOME"), reason="No FreeSurfer")
@pytest.mark.parametrize(
    "spaces", [[], ["fsaverage"], ["fsnative"], ["fsaverage5", "fsnative"]]
)
def test_fsdir(tmp_path, spaces):
    fshome = os.environ["FREESURFER_HOME"]
    subjects_dir = tmp_path / "freesurfer"

    # Verify we're starting clean
    for space in spaces:
        if space.startswith("fsaverage"):
            assert not Path.exists(subjects_dir / space)

    # Run three times to check idempotence
    # Third time force an overwrite
    for overwrite_fsaverage in (False, False, True):
        res = bintfs.BIDSFreeSurferDir(
            derivatives=str(tmp_path),
            spaces=spaces,
            freesurfer_home=fshome,
            overwrite_fsaverage=overwrite_fsaverage,
        ).run()
        assert res.outputs.subjects_dir == str(subjects_dir)

        for space in spaces:
            if space.startswith("fsaverage"):
                assert Path.exists(subjects_dir / space)


@pytest.mark.skipif(not os.getenv("FREESURFER_HOME"), reason="No FreeSurfer")
def test_fsdir_missing_space(tmp_path):
    fshome = os.environ["FREESURFER_HOME"]

    # fsaverage2 doesn't exist in source or destination, so can't copy
    with pytest.raises(FileNotFoundError):
        bintfs.BIDSFreeSurferDir(
            derivatives=str(tmp_path), spaces=["fsaverage2"], freesurfer_home=fshome
        ).run()

    subjects_dir = tmp_path / "freesurfer"

    # If fsaverage2 exists in the destination directory, no error is thrown
    Path.mkdir(subjects_dir / "fsaverage2")
    bintfs.BIDSFreeSurferDir(
        derivatives=str(tmp_path), spaces=["fsaverage2"], freesurfer_home=fshome
    ).run()
