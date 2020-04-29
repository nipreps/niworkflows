# -*- coding: utf-8 -*-
"""Test viz module"""
import os
import numpy as np
import nibabel as nb
from .. import viz
from .conftest import datadir
from pathlib import Path


def test_carpetplot():
    """Write a carpetplot"""
    out_file_nifti = None
    out_file_cifti = None
    save_artifacts = os.getenv("SAVE_CIRCLE_ARTIFACTS", False)
    if save_artifacts:
        out_file_nifti = os.path.join(save_artifacts, "carpetplot_nifti.svg")
        out_file_cifti = os.path.join(save_artifacts, "carpetplot_cifti.svg")

    # volumetric NIfTI
    viz.plot_carpet(
        os.path.join(
            datadir, "sub-ds205s03_task-functionallocalizer_run-01_bold_volreg.nii.gz"
        ),
        atlaslabels=np.asanyarray(
            nb.load(
                os.path.join(
                    datadir,
                    "sub-ds205s03_task-functionallocalizer_run-01_bold_parc.nii.gz",
                )
            ).dataobj
        ),
        output_file=out_file_nifti,
        legend=True,
    )

    # CIFTI
    viz.plot_carpet(
        os.path.join(
            datadir,
            "sub-01_task-mixedgamblestask_run-02_space-fsLR_den-91k_bold.dtseries.nii",
        ),
        output_file=out_file_cifti,
    )


def test_plot_melodic_components(tmp_path):
    """Test plotting melodic components"""
    import numpy as np

    # save the artifacts
    out_dir = Path(os.getenv("SAVE_CIRCLE_ARTIFACTS", str(tmp_path)))
    all_noise = str(out_dir / "melodic_all_noise.svg")
    no_noise = str(out_dir / "melodic_no_noise.svg")
    no_classified = str(out_dir / "melodic_no_classified.svg")

    # melodic directory
    melodic_dir = tmp_path / "melodic"
    melodic_dir.mkdir(exist_ok=True)
    # melodic_mix
    mel_mix = np.random.randint(low=-5, high=5, size=[10, 2])
    mel_mix_file = str(melodic_dir / "melodic_mix")
    np.savetxt(mel_mix_file, mel_mix, fmt="%i")
    # melodic_FTmix
    mel_ftmix = np.random.rand(2, 5)
    mel_ftmix_file = str(melodic_dir / "melodic_FTmix")
    np.savetxt(mel_ftmix_file, mel_ftmix)
    # melodic_ICstats
    mel_icstats = np.random.rand(2, 2)
    mel_icstats_file = str(melodic_dir / "melodic_ICstats")
    np.savetxt(mel_icstats_file, mel_icstats)
    # melodic_IC
    mel_ic = np.random.rand(2, 2, 2, 2)
    mel_ic_file = str(melodic_dir / "melodic_IC.nii.gz")
    mel_ic_img = nb.Nifti2Image(mel_ic, np.eye(4))
    mel_ic_img.to_filename(mel_ic_file)
    # noise_components
    noise_comps = np.array([1, 2])
    noise_comps_file = str(tmp_path / "noise_ics.csv")
    np.savetxt(noise_comps_file, noise_comps, fmt="%i", delimiter=",")

    # create empty components file
    nocomps_file = str(tmp_path / "noise_none.csv")
    open(nocomps_file, "w").close()

    # in_file
    in_fname = str(tmp_path / "in_file.nii.gz")
    voxel_ts = np.random.rand(2, 2, 2, 10)
    in_file = nb.Nifti2Image(voxel_ts, np.eye(4))
    in_file.to_filename(in_fname)
    # report_mask
    report_fname = str(tmp_path / "report_mask.nii.gz")
    report_mask = nb.Nifti2Image(np.ones([2, 2, 2]), np.eye(4))
    report_mask.to_filename(report_fname)

    # run command with all noise components
    viz.utils.plot_melodic_components(
        str(melodic_dir),
        in_fname,
        tr=2.0,
        report_mask=report_fname,
        noise_components_file=noise_comps_file,
        out_file=all_noise,
    )
    # run command with no noise components
    viz.utils.plot_melodic_components(
        str(melodic_dir),
        in_fname,
        tr=2.0,
        report_mask=report_fname,
        noise_components_file=nocomps_file,
        out_file=no_noise,
    )

    # run command without noise components file
    viz.utils.plot_melodic_components(
        str(melodic_dir),
        in_fname,
        tr=2.0,
        report_mask=report_fname,
        out_file=no_classified,
    )


def test_compcor_variance_plot(tmp_path):
    """Test plotting CompCor variance"""
    out_dir = Path(os.getenv("SAVE_CIRCLE_ARTIFACTS", str(tmp_path)))
    out_file = str(out_dir / "variance_plot_short.svg")
    metadata_file = os.path.join(datadir, "confounds_metadata_short_test.tsv")
    viz.plots.compcor_variance_plot([metadata_file], output_file=out_file)
