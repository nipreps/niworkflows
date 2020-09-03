"""Testing module for fmriprep.workflows.bold.util."""
import pytest
import os
from pathlib import Path

import numpy as np
from nipype.pipeline import engine as pe
from nipype.utils.filemanip import fname_presuffix, copyfile
from nilearn.image import load_img

from ...utils.connections import listify
from niworkflows.interfaces.masks import ROIsPlot

from ..util import init_bold_reference_wf

# Multi-echo datasets
bold_datasets = ["""\
ds000210/sub-06_task-rest_run-01_echo-1_bold.nii.gz
ds000210/sub-06_task-rest_run-01_echo-2_bold.nii.gz
ds000210/sub-06_task-rest_run-01_echo-3_bold.nii.gz\
""".splitlines(), """\
ds000216/sub-03_task-rest_echo-1_bold.nii.gz
ds000216/sub-03_task-rest_echo-2_bold.nii.gz
ds000216/sub-03_task-rest_echo-3_bold.nii.gz
ds000216/sub-03_task-rest_echo-4_bold.nii.gz""".splitlines()]

# Single-echo datasets
bold_datasets += """\
ds000116/sub-12_task-visualoddballwithbuttonresponsetotargetstimuli_run-02_bold.nii.gz
ds000133/sub-06_ses-post_task-rest_run-01_bold.nii.gz
ds000140/sub-32_task-heatpainwithregulationandratings_run-02_bold.nii.gz
ds000157/sub-23_task-passiveimageviewing_bold.nii.gz
ds000237/sub-03_task-MemorySpan_acq-multiband_run-01_bold.nii.gz
ds000237/sub-06_task-MemorySpan_acq-multiband_run-01_bold.nii.gz
ds001240/sub-26_task-localizerimagination_bold.nii.gz
ds001240/sub-26_task-localizerviewing_bold.nii.gz
ds001240/sub-26_task-molencoding_run-01_bold.nii.gz
ds001240/sub-26_task-molencoding_run-02_bold.nii.gz
ds001240/sub-26_task-molretrieval_run-01_bold.nii.gz
ds001240/sub-26_task-molretrieval_run-02_bold.nii.gz
ds001240/sub-26_task-rest_bold.nii.gz
ds001362/sub-01_task-taskname_run-01_bold.nii.gz""".splitlines()

bold_datasets = [listify(d) for d in bold_datasets]


def symmetric_overlap(img1, img2):
    mask1 = load_img(img1).get_fdata() > 0
    mask2 = load_img(img2).get_fdata() > 0

    total1 = np.sum(mask1)
    total2 = np.sum(mask2)
    overlap = np.sum(mask1 & mask2)
    return overlap / np.sqrt(total1 * total2)


@pytest.mark.skipif(
    not os.getenv("FMRIPREP_REGRESSION_SOURCE")
    or not os.getenv("FMRIPREP_REGRESSION_TARGETS"),
    reason="FMRIPREP_REGRESSION_{SOURCE,TARGETS} env vars not set",
)
@pytest.mark.parametrize(
    "input_fname,expected_fname",
    [
        (
            [os.path.join(os.getenv("FMRIPREP_REGRESSION_SOURCE", ""), bf)
             for bf in base_fname],
            fname_presuffix(
                base_fname[0].replace("_echo-1", ""),
                suffix="_mask",
                use_ext=True,
                newpath=os.path.join(
                    os.getenv("FMRIPREP_REGRESSION_TARGETS", ""),
                    os.path.dirname(base_fname[0]),
                ),
            ),
        )
        for base_fname in bold_datasets
    ],
)
def test_masking(input_fname, expected_fname):
    basename = Path(input_fname[0]).name
    dsname = Path(expected_fname).parent.name

    # Reconstruct base_fname from above
    reports_dir = Path(os.getenv("FMRIPREP_REGRESSION_REPORTS", ""))
    newpath = reports_dir / dsname

    name = basename.rstrip("_bold.nii.gz").replace("-", "_")
    bold_reference_wf = init_bold_reference_wf(omp_nthreads=1, name=name,
                                               multiecho=len(input_fname) > 1)
    bold_reference_wf.inputs.inputnode.bold_file = input_fname[0] if len(input_fname) == 1 \
        else input_fname
    base_dir = os.getenv("CACHED_WORK_DIRECTORY")
    if base_dir:
        base_dir = Path(base_dir) / dsname
        base_dir.mkdir(parents=True, exist_ok=True)
        bold_reference_wf.base_dir = str(base_dir)

    out_fname = fname_presuffix(
        Path(expected_fname).name, suffix=".svg", use_ext=False, newpath=str(newpath)
    )
    newpath.mkdir(parents=True, exist_ok=True)

    mask_diff_plot = pe.Node(
        ROIsPlot(colors=["limegreen"], levels=[0.5]), name="mask_diff_plot"
    )
    mask_diff_plot.always_run = True
    mask_diff_plot.inputs.in_mask = expected_fname
    mask_diff_plot.inputs.out_report = out_fname

    outputnode = bold_reference_wf.get_node("outputnode")
    bold_reference_wf.connect(
        [
            (
                outputnode,
                mask_diff_plot,
                [("ref_image", "in_file"), ("bold_mask", "in_rois")],
            )
        ]
    )
    res = bold_reference_wf.run(plugin="MultiProc")

    combine_masks = [node for node in res.nodes if node.name.endswith("combine_masks")][
        0
    ]
    overlap = symmetric_overlap(expected_fname, combine_masks.result.outputs.out_file)

    mask_dir = reports_dir / "fmriprep_bold_mask" / dsname
    mask_dir.mkdir(parents=True, exist_ok=True)
    copyfile(
        combine_masks.result.outputs.out_file,
        str(mask_dir / Path(expected_fname).name),
        copy=True,
    )

    assert overlap > 0.95, input_fname
