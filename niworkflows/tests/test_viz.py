# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Test viz module"""

import os
from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd
import pytest

from niworkflows.interfaces.plotting import _get_tr
from niworkflows.utils.timeseries import _cifti_timeseries, _nifti_timeseries
from niworkflows.viz.plots import fMRIPlot

from .. import viz
from .conftest import datadir
from .generate_data import _create_dtseries_cifti


@pytest.mark.parametrize('tr', [None, 0.7])
@pytest.mark.parametrize('sorting', [None, 'ward', 'linkage'])
def test_carpetplot(tr, sorting):
    """Write a carpetplot"""
    save_artifacts = os.getenv('SAVE_CIRCLE_ARTIFACTS')

    rng = np.random.default_rng(2010)

    viz.plot_carpet(
        rng.normal(100, 20, size=(18000, 1900)),
        title='carpetplot with title',
        tr=tr,
        output_file=(
            os.path.join(
                save_artifacts,
                f'carpet_nosegs_{"index" if tr is None else "time"}_'
                f'{"nosort" if sorting is None else sorting}.svg',
            )
            if save_artifacts
            else None
        ),
        sort_rows=sorting,
        drop_trs=15,
    )

    labels = ('Ctx GM', 'Subctx GM', 'WM+CSF', 'Cereb.', 'Edge')
    sizes = (200, 100, 50, 100, 50)
    total_size = np.sum(sizes)
    data = np.zeros((total_size, 300))

    indexes = np.arange(total_size)
    rng.shuffle(indexes)
    segments = {}
    start = 0
    for group, size in zip(labels, sizes):
        segments[group] = indexes[start : start + size]
        data[indexes[start : start + size]] = rng.normal(
            rng.standard_normal(1) * 100, rng.normal(20, 5, size=1), size=(size, 300)
        )
        start += size

    viz.plot_carpet(
        data,
        segments,
        tr=tr,
        output_file=(
            os.path.join(
                save_artifacts,
                f'carpet_random_{"index" if tr is None else "seg"}_'
                f'{"nosort" if sorting is None else sorting}.svg',
            )
            if save_artifacts
            else None
        ),
        sort_rows=sorting,
    )

    data = np.zeros((total_size, 300))
    indexes = np.arange(total_size)
    rng.shuffle(indexes)
    segments = {}
    start = 0
    for i, (group, size) in enumerate(zip(labels, sizes)):
        segments[group] = indexes[start : start + size]
        data[indexes[start : start + size]] = i
        start += size

    viz.plot_carpet(
        data,
        segments,
        detrend=False,
        tr=tr,
        output_file=(
            os.path.join(
                save_artifacts,
                f'carpet_const_{"index" if tr is None else "time"}_'
                f'{"nosort" if sorting is None else sorting}.svg',
            )
            if save_artifacts
            else None
        ),
        sort_rows=sorting,
    )


@pytest.mark.parametrize(
    'input_files',
    [
        ('sub-ds205s03_task-functionallocalizer_run-01_bold_volreg.nii.gz', None),
        ('sub-01_task-mixedgamblestask_run-02_space-fsLR_den-91k_bold.dtseries.nii', None),
        (
            'sub-ds205s03_task-functionallocalizer_run-01_bold_volreg.nii.gz',
            'sub-ds205s03_task-functionallocalizer_run-01_bold_parc.nii.gz',
        ),
    ],
)
def test_fmriplot(input_files):
    """Exercise the fMRIPlot class."""
    save_artifacts = os.getenv('SAVE_CIRCLE_ARTIFACTS')
    rng = np.random.default_rng(2010)

    in_file = os.path.join(datadir, input_files[0])
    seg_file = os.path.join(datadir, input_files[1]) if input_files[1] is not None else None

    dtype = 'nifti' if input_files[0].endswith('volreg.nii.gz') else 'cifti'
    has_seg = '_parc' if seg_file else ''

    timeseries, segments = (
        _nifti_timeseries(in_file, seg_file) if dtype == 'nifti' else _cifti_timeseries(in_file)
    )

    fig = fMRIPlot(
        timeseries,
        segments,
        tr=_get_tr(nb.load(in_file)),
        confounds=pd.DataFrame(
            {
                'outliers': rng.normal(0.2, 0.2, timeseries.shape[-1] - 1),
                'DVARS': rng.normal(0.2, 0.2, timeseries.shape[-1] - 1),
                'FD': rng.normal(0.2, 0.2, timeseries.shape[-1] - 1),
            }
        ),
        units={'FD': 'mm'},
        paired_carpet=dtype == 'cifti',
    ).plot()
    if save_artifacts:
        fig.savefig(
            os.path.join(save_artifacts, f'fmriplot_{dtype}{has_seg}.svg'),
            bbox_inches='tight',
        )


def test_plot_melodic_components(tmp_path):
    """Test plotting melodic components"""
    import numpy as np

    # save the artifacts
    out_dir = Path(os.getenv('SAVE_CIRCLE_ARTIFACTS', str(tmp_path)))
    all_noise = str(out_dir / 'melodic_all_noise.svg')
    no_noise = str(out_dir / 'melodic_no_noise.svg')
    no_classified = str(out_dir / 'melodic_no_classified.svg')

    # melodic directory
    melodic_dir = tmp_path / 'melodic'
    melodic_dir.mkdir(exist_ok=True)
    # melodic_mix
    mel_mix = np.random.randint(low=-5, high=5, size=[10, 2])
    mel_mix_file = str(melodic_dir / 'melodic_mix')
    np.savetxt(mel_mix_file, mel_mix, fmt='%i')
    # melodic_FTmix
    mel_ftmix = np.random.rand(2, 5)
    mel_ftmix_file = str(melodic_dir / 'melodic_FTmix')
    np.savetxt(mel_ftmix_file, mel_ftmix)
    # melodic_ICstats
    mel_icstats = np.random.rand(2, 2)
    mel_icstats_file = str(melodic_dir / 'melodic_ICstats')
    np.savetxt(mel_icstats_file, mel_icstats)
    # melodic_IC
    mel_ic = np.random.rand(2, 2, 2, 2)
    mel_ic_file = str(melodic_dir / 'melodic_IC.nii.gz')
    mel_ic_img = nb.Nifti2Image(mel_ic, np.eye(4))
    mel_ic_img.to_filename(mel_ic_file)
    # noise_components
    noise_comps = np.array([1, 2])
    noise_comps_file = str(tmp_path / 'noise_ics.csv')
    np.savetxt(noise_comps_file, noise_comps, fmt='%i', delimiter=',')

    # create empty components file
    nocomps_file = str(tmp_path / 'noise_none.csv')
    open(nocomps_file, 'w').close()

    # in_file
    in_fname = str(tmp_path / 'in_file.nii.gz')
    voxel_ts = np.random.rand(2, 2, 2, 10)
    in_file = nb.Nifti2Image(voxel_ts, np.eye(4))
    in_file.to_filename(in_fname)
    # report_mask
    report_fname = str(tmp_path / 'report_mask.nii.gz')
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
    out_dir = Path(os.getenv('SAVE_CIRCLE_ARTIFACTS', str(tmp_path)))
    out_file = str(out_dir / 'variance_plot_short.svg')
    metadata_file = os.path.join(datadir, 'confounds_metadata_short_test.tsv')
    viz.plots.compcor_variance_plot([metadata_file], output_file=out_file)


@pytest.fixture
def create_surface_dtseries():
    """Create a dense timeseries CIFTI-2 file with only cortex structures"""
    out_file = _create_dtseries_cifti(
        timepoints=10,
        models=[
            ('CIFTI_STRUCTURE_CORTEX_LEFT', np.random.rand(29696, 10)),
            ('CIFTI_STRUCTURE_CORTEX_RIGHT', np.random.rand(29716, 10)),
        ],
    )
    yield str(out_file)
    out_file.unlink()


def test_cifti_surfaces_plot(tmp_path, create_surface_dtseries):
    """Test plotting CIFTI-2 surfaces"""
    os.chdir(tmp_path)
    out_dir = Path(os.getenv('SAVE_CIRCLE_ARTIFACTS', str(tmp_path)))
    out_file = str(out_dir / 'cifti_surfaces_plot.svg')
    viz.plots.cifti_surfaces_plot(create_surface_dtseries, output_file=out_file)
