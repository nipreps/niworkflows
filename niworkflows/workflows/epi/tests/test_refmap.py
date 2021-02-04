"""Check the refmap module."""
import os
from ..refmap import init_epi_reference_wf


def test_reference(tmpdir, ds000030_dir, workdir, outdir):
    """Exercise the EPI reference workflow."""
    tmpdir.chdir()

    wf = init_epi_reference_wf(omp_nthreads=os.cpu_count())
    if workdir:
        wf.base_dir = str(workdir)

    wf.inputs.inputnode.in_files = [
        str(f) for f in (ds000030_dir / "sub-10228" / "func").glob("*_bold.nii.gz")
    ]

    # if outdir:
    #     out_path = outdir / "masks" / folder.split("/")[-1]
    #     out_path.mkdir(exist_ok=True, parents=True)
    #     report = pe.Node(SimpleShowMaskRPT(), name="report")
    #     report.interface._always_run = True

    #     def _report_name(fname, out_path):
    #         from pathlib import Path

    #         return str(
    #             out_path
    #             / Path(fname)
    #             .name.replace(".nii", "_mask.svg")
    #             .replace("_magnitude", "_desc-magnitude")
    #             .replace(".gz", "")
    #         )

    #     # fmt: off
    #     wf.connect([
    #         (inputnode, report, [(("in_file", _report_name, out_path), "out_report")]),
    #         (brainmask_wf, report, [("outputnode.out_mask", "mask_file"),
    #                                 ("outputnode.out_file", "background_file")]),
    #     ])
    #     # fmt: on

    wf.run()
