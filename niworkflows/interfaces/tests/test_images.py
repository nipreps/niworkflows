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
"""Test images module."""

import time
from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
from nipype.interfaces import nilearn as nl
from nipype.pipeline import engine as pe

from niworkflows.testing import has_afni

from .. import images as im


@pytest.mark.parametrize(
    ('nvols', 'nmasks', 'ext', 'factor'),
    [
        (200, 3, '.nii', 1.1),
    ],
)
def test_signal_extraction_equivalence(tmp_path, nvols, nmasks, ext, factor):
    nlsignals = str(tmp_path / 'nlsignals.tsv')
    imsignals = str(tmp_path / 'imsignals.tsv')

    vol_shape = (64, 64, 40)

    img_fname = str(tmp_path / ('img' + ext))
    masks_fname = str(tmp_path / ('masks' + ext))

    random_data = np.random.random(size=vol_shape + (nvols,)) * 2000
    random_mask_data = np.random.random(size=vol_shape + (nmasks,)) < 0.2

    nb.Nifti1Image(random_data, np.eye(4)).to_filename(img_fname)
    nb.Nifti1Image(random_mask_data.astype(np.uint8), np.eye(4)).to_filename(masks_fname)

    se1 = nl.SignalExtraction(
        in_file=img_fname,
        label_files=masks_fname,
        class_labels=[f'a{i}' for i in range(nmasks)],
        out_file=nlsignals,
    )
    se2 = im.SignalExtraction(
        in_file=img_fname,
        label_files=masks_fname,
        class_labels=[f'a{i}' for i in range(nmasks)],
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
    ('shape', 'mshape'),
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

    data = np.random.normal(size=shape).astype('float32')
    fname = str(tmpdir.join('file1.nii.gz'))
    nb.Nifti1Image(data, np.eye(4), None).to_filename(fname)

    if mshape is None:
        with pytest.raises(RuntimeError):
            im.IntraModalMerge(in_files=fname).run()
        return

    merged = str(im.IntraModalMerge(in_files=fname).run().outputs.out_file)
    merged_data = nb.load(merged).get_fdata(dtype='float32')
    assert merged_data.shape == mshape
    assert np.allclose(np.squeeze(data), merged_data)

    merged = str(im.IntraModalMerge(in_files=[fname, fname], hmc=False).run().outputs.out_file)
    merged_data = nb.load(merged).get_fdata(dtype='float32')
    new_mshape = (*mshape[:3], 2 if len(mshape) == 3 else mshape[3] * 2)
    assert merged_data.shape == new_mshape


def test_conform_resize(tmpdir):
    fname = str(tmpdir / 'test.nii')

    random_data = np.random.random(size=(5, 5, 5))
    img = nb.Nifti1Image(random_data, np.eye(4))
    img.to_filename(fname)
    conform = pe.Node(im.Conform(), name='conform', base_dir=str(tmpdir))
    conform.inputs.in_file = fname
    conform.inputs.target_zooms = (1, 1, 1.5)
    conform.inputs.target_shape = (5, 5, 5)
    res = conform.run()

    out_img = nb.load(res.outputs.out_file)
    assert out_img.header.get_zooms() == conform.inputs.target_zooms


def test_conform_set_zooms(tmpdir):
    fname = str(tmpdir / 'test.nii')

    random_data = np.random.random(size=(5, 5, 5))
    img = nb.Nifti1Image(random_data, np.eye(4))
    img.to_filename(fname)
    conform = pe.Node(im.Conform(), name='conform', base_dir=str(tmpdir))
    conform.inputs.in_file = fname
    conform.inputs.target_zooms = (1, 1, 1.002)
    conform.inputs.target_shape = (5, 5, 5)
    res = conform.run()

    out_img = nb.load(res.outputs.out_file)
    assert np.allclose(out_img.header.get_zooms(), conform.inputs.target_zooms)


@pytest.mark.skipif(not has_afni, reason='Needs AFNI')
@pytest.mark.parametrize(
    'shape',
    [
        (10, 10, 10),
        (10, 10, 10, 1),
        (10, 10, 10, 10),
    ],
)
def test_RobustAverage(tmpdir, shape):
    """Exercise the various types of inputs."""
    tmpdir.chdir()

    data = np.ones(shape, dtype='float32')
    t_mask = [True]
    if len(shape) == 4 and shape[-1] > 1:
        data *= np.linspace(0.6, 1.0, num=10)[::-1]
        t_mask = np.zeros(shape[3], dtype=bool)
        t_mask[:3] = True

    fname = str(tmpdir.join('file1.nii.gz'))
    nb.Nifti1Image(data, np.eye(4), None).to_filename(fname)

    avg = im.RobustAverage(in_file=fname, t_mask=list(t_mask)).run()
    out_file = nb.load(avg.outputs.out_file)

    assert out_file.shape == (10, 10, 10)
    assert np.allclose(out_file.get_fdata(), 1.0)


def test_TemplateDimensions(tmp_path):
    """Exercise the various types of inputs."""
    shapes = [
        (10, 10, 10),
        (11, 11, 11),
    ]
    zooms = [
        (1, 1, 1),
        (0.9, 0.9, 0.9),
    ]

    for i, (shape, zoom) in enumerate(zip(shapes, zooms)):
        img = nb.Nifti1Image(np.ones(shape, dtype='float32'), np.eye(4))
        img.header.set_zooms(zoom)
        img.to_filename(tmp_path / f'test{i}.nii')

    anat_list = [str(tmp_path / f'test{i}.nii') for i in range(2)]
    td = im.TemplateDimensions(anat_list=anat_list)
    res = td.run()

    report = Path(res.outputs.out_report).read_text()
    assert 'Input T1w images: 2' in report
    assert 'Output dimensions: 11x11x11' in report
    assert 'Output voxel size: 0.9mm x 0.9mm x 0.9mm' in report
    assert 'Discarded images: 0' in report

    assert res.outputs.t1w_valid_list == anat_list
    assert res.outputs.anat_valid_list == anat_list
    assert np.allclose(res.outputs.target_zooms, (0.9, 0.9, 0.9))
    assert res.outputs.target_shape == (11, 11, 11)

    with pytest.warns(UserWarning, match='t1w_list .* is deprecated'):
        im.TemplateDimensions(t1w_list=anat_list)
