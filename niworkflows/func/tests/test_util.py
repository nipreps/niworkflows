"""Testing module for fmriprep.workflows.bold.util."""
import pytest
import os
from pathlib import Path
from shutil import which

import numpy as np
from nipype.pipeline import engine as pe
from nipype.utils.filemanip import fname_presuffix, copyfile
from nilearn.image import load_img

from ...interfaces.reportlets.masks import ROIsPlot
from ...workflows.epi.refmap import init_epi_reference_wf

from ..util import init_enhance_and_skullstrip_bold_wf

datapath = os.getenv("FMRIPREP_REGRESSION_SOURCE")
parameters = []

if datapath:
    datapath = Path(datapath)
    bold_datasets = []

    for ds in datapath.glob("ds*/"):
        paths = [
            str(p.relative_to(datapath)) for p in ds.glob("*_bold.nii.gz") if p.exists()
        ]
        bold_datasets += sorted([[p] for p in paths if "echo-" not in p])
        meecho = sorted([p for p in paths if "echo-" in p])
        if meecho:
            bold_datasets.append(meecho)

    exp_masks = []
    for path in bold_datasets:
        path = path[0]
        exp_masks.append(
            str(
                (
                    datapath
                    / "derivatives"
                    / path.replace("_echo-1", "").replace("_bold.nii", "_bold_mask.nii")
                ).absolute()
            )
        )

    bold_datasets = [
        [str((datapath / p).absolute()) for p in ds] for ds in bold_datasets
    ]

    parameters = zip(bold_datasets, exp_masks)

    if not bold_datasets:
        raise RuntimeError(
            f"Data folder <{datapath}> was provided, but no images were found. "
            "Folder contents:\n{}".format(
                "\n".join([str(p) for p in datapath.glob("ds*/*.nii.gz")])
            )
        )


def symmetric_overlap(img1, img2):
    mask1 = load_img(img1).get_fdata() > 0
    mask2 = load_img(img2).get_fdata() > 0

    total1 = np.sum(mask1)
    total2 = np.sum(mask2)
    overlap = np.sum(mask1 & mask2)
    return overlap / np.sqrt(total1 * total2)


@pytest.mark.skipif(
    not datapath,
    reason="FMRIPREP_REGRESSION_SOURCE env var not set, or no data is available",
)
@pytest.mark.skipif(not which("antsAI"), reason="antsAI executable not found")
@pytest.mark.parametrize("input_fname,expected_fname", parameters)
def test_masking(input_fname, expected_fname):
    """Check for regressions in masking."""
    from nipype import config as ncfg

    basename = Path(input_fname[0]).name
    dsname = Path(expected_fname).parent.name

    # Reconstruct base_fname from above
    reports_dir = Path(os.getenv("FMRIPREP_REGRESSION_REPORTS", ""))
    newpath = reports_dir / dsname
    newpath.mkdir(parents=True, exist_ok=True)

    # Nipype config (logs and execution)
    ncfg.update_config(
        {
            "execution": {
                "crashdump_dir": str(newpath),
            }
        }
    )

    wf = pe.Workflow(name=basename.replace("_bold.nii.gz", "").replace("-", "_"))
    base_dir = os.getenv("CACHED_WORK_DIRECTORY")
    if base_dir:
        base_dir = Path(base_dir) / dsname
        base_dir.mkdir(parents=True, exist_ok=True)
        wf.base_dir = str(base_dir)

    epi_reference_wf = init_epi_reference_wf(omp_nthreads=os.cpu_count())
    epi_reference_wf.inputs.inputnode.in_files = (
        input_fname[0] if len(input_fname) == 1 else input_fname
    )
    enhance_and_skullstrip_bold_wf = init_enhance_and_skullstrip_bold_wf()

    out_fname = fname_presuffix(
        Path(expected_fname).name, suffix=".svg", use_ext=False, newpath=str(newpath)
    )

    mask_diff_plot = pe.Node(
        ROIsPlot(colors=["limegreen"], levels=[0.5]), name="mask_diff_plot"
    )
    mask_diff_plot.always_run = True
    mask_diff_plot.inputs.in_mask = expected_fname
    mask_diff_plot.inputs.out_report = out_fname

    # fmt:off
    wf.connect([
        (epi_reference_wf, enhance_and_skullstrip_bold_wf, [
            ("outputnode.epiref", "inputnode.in_file")
        ]),
        (enhance_and_skullstrip_bold_wf, mask_diff_plot, [
            ("outputnode.bias_corrected_file", "in_file"),
            ("outputnode.mask_file", "in_rois"),
        ]),
    ])

    res = wf.run(plugin="MultiProc")

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
