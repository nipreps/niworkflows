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
"""Tests on BIDS compliance."""

import json
import os
from hashlib import sha1
from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
from nipype.interfaces.base import Undefined
from packaging.version import Version

from niworkflows.testing import needs_data_dir

from .. import bids as bintfs

XFORM_CODES = {
    'MNI152Lin': 4,
    'T1w': 2,
    'boldref': 2,
    None: 1,
}

T1W_PATH = 'ds054/sub-100185/anat/sub-100185_T1w.nii.gz'
BOLD_PATH = 'ds054/sub-100185/func/sub-100185_task-machinegame_run-01_bold.nii.gz'


def make_prep_and_save(
    prep_interface,
    base_directory,
    out_path_base=None,
    **kwargs,
):
    if prep_interface is bintfs.DerivativesDataSink:
        kwargs.update(out_path_base=out_path_base, base_directory=base_directory)

    prep = save = prep_interface(**kwargs)

    if prep_interface is not bintfs.DerivativesDataSink:
        save = bintfs.SaveDerivative(base_directory=base_directory)

    return prep, save


def connect_and_run_save(prep_result, save):
    if prep_result.interface is bintfs.DerivativesDataSink:
        return prep_result

    save.inputs.in_file = prep_result.outputs.out_file
    save.inputs.relative_path = prep_result.outputs.out_path
    save.inputs.metadata = prep_result.outputs.out_meta

    return save.run()


@pytest.mark.parametrize('interface', [bintfs.DerivativesDataSink, bintfs.PrepareDerivative])
@pytest.mark.parametrize('out_path_base', [None, 'fmriprep'])
@pytest.mark.parametrize(
    ('source', 'input_files', 'entities', 'expectation', 'checksum'),
    [
        (
            T1W_PATH,
            ['anat.nii.gz'],
            {'desc': 'preproc'},
            'sub-100185/anat/sub-100185_desc-preproc_T1w.nii.gz',
            '7c047921def32da260df4a985019b9f5231659fa',
        ),
        (
            T1W_PATH,
            ['anat.nii.gz'],
            {'desc': 'preproc', 'space': 'MNI'},
            'sub-100185/anat/sub-100185_space-MNI_desc-preproc_T1w.nii.gz',
            'b22399f50ce454049d5d074457a92ab13e7fdf8c',
        ),
        (
            T1W_PATH,
            ['anat.nii.gz'],
            {'desc': 'preproc', 'space': 'MNI', 'resolution': 'native'},
            'sub-100185/anat/sub-100185_space-MNI_desc-preproc_T1w.nii.gz',
            'b22399f50ce454049d5d074457a92ab13e7fdf8c',
        ),
        (
            T1W_PATH,
            ['anat.nii.gz'],
            {'desc': 'preproc', 'space': 'MNI', 'resolution': 'high'},
            'sub-100185/anat/sub-100185_space-MNI_res-high_desc-preproc_T1w.nii.gz',
            'b22399f50ce454049d5d074457a92ab13e7fdf8c',
        ),
        (
            T1W_PATH,
            ['tfm.txt'],
            {'from': 'fsnative', 'to': 'T1w', 'suffix': 'xfm'},
            'sub-100185/anat/sub-100185_from-fsnative_to-T1w_mode-image_xfm.txt',
            'da39a3ee5e6b4b0d3255bfef95601890afd80709',
        ),
        (
            T1W_PATH,
            ['tfm.h5'],
            {'from': 'MNI152NLin2009cAsym', 'to': 'T1w', 'suffix': 'xfm'},
            'sub-100185/anat/sub-100185_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5',
            'da39a3ee5e6b4b0d3255bfef95601890afd80709',
        ),
        (
            T1W_PATH,
            ['anat.nii.gz'],
            {'desc': 'brain', 'suffix': 'mask'},
            'sub-100185/anat/sub-100185_desc-brain_mask.nii.gz',
            '7af86a1f6806a41078e4d2699d680dbe2b9f6ae2',
        ),
        (
            T1W_PATH,
            ['anat.nii.gz'],
            {'desc': 'brain', 'suffix': 'mask', 'space': 'MNI'},
            'sub-100185/anat/sub-100185_space-MNI_desc-brain_mask.nii.gz',
            '1591f90e0da2a624c972784dda6a01b5572add15',
        ),
        (
            T1W_PATH,
            ['anat.surf.gii'],
            {'suffix': 'pial', 'hemi': 'L'},
            'sub-100185/anat/sub-100185_hemi-L_pial.surf.gii',
            'da39a3ee5e6b4b0d3255bfef95601890afd80709',
        ),
        (
            T1W_PATH,
            ['aseg.nii', 'aparc.nii'],
            {'desc': ['aseg', 'aparcaseg'], 'suffix': 'dseg'},
            [f'sub-100185/anat/sub-100185_desc-{s}_dseg.nii' for s in ('aseg', 'aparcaseg')],
            [
                '5543a788bf3383d7a2fc41f5cff4e0bbb8f5f282',
                '5543a788bf3383d7a2fc41f5cff4e0bbb8f5f282',
            ],
        ),
        (
            T1W_PATH,
            ['anat.nii', 'anat.json'],
            {'desc': 'preproc'},
            [f'sub-100185/anat/sub-100185_desc-preproc_T1w.{ext}' for ext in ('nii', 'json')],
            [
                '25c107d4a3e6f98e48aa752c5bbd88ab8e8d069f',
                'da39a3ee5e6b4b0d3255bfef95601890afd80709',
            ],
        ),
        (
            T1W_PATH,
            ['anat.nii.gz'] * 3,
            {'label': ['GM', 'WM', 'CSF'], 'suffix': 'probseg'},
            [
                f'sub-100185/anat/sub-100185_label-{lab}_probseg.nii.gz'
                for lab in ('GM', 'WM', 'CSF')
            ],
            ['7c047921def32da260df4a985019b9f5231659fa'] * 3,
        ),
        # BOLD data
        (
            BOLD_PATH,
            ['aroma.csv'],
            {'suffix': 'AROMAnoiseICs'},
            'sub-100185/func/sub-100185_task-machinegame_run-01_AROMAnoiseICs.csv',
            'da39a3ee5e6b4b0d3255bfef95601890afd80709',
        ),
        (
            BOLD_PATH,
            ['confounds.tsv'],
            {'suffix': 'regressors', 'desc': 'confounds'},
            'sub-100185/func/sub-100185_task-machinegame_run-01_desc-confounds_regressors.tsv',
            'da39a3ee5e6b4b0d3255bfef95601890afd80709',
        ),
        (
            BOLD_PATH,
            ['mixing.tsv'],
            {'suffix': 'mixing', 'desc': 'MELODIC'},
            'sub-100185/func/sub-100185_task-machinegame_run-01_desc-MELODIC_mixing.tsv',
            'da39a3ee5e6b4b0d3255bfef95601890afd80709',
        ),
        (
            BOLD_PATH,
            ['lh.func.gii'],
            {'space': 'fsaverage', 'density': '10k', 'hemi': 'L'},
            'sub-100185/func/sub-100185_task-machinegame_run-01_'
            'hemi-L_space-fsaverage_den-10k_bold.func.gii',
            'da39a3ee5e6b4b0d3255bfef95601890afd80709',
        ),
        (
            BOLD_PATH,
            ['hcp.dtseries.nii'],
            {'space': 'fsLR', 'density': '91k'},
            'sub-100185/func/sub-100185_task-machinegame_run-01_'
            'space-fsLR_den-91k_bold.dtseries.nii',
            '335f1394ce90b58bbf27026b6eeec4d2124c11da',
        ),
        (
            BOLD_PATH,
            ['ref.nii'],
            {'space': 'MNI', 'suffix': 'boldref'},
            'sub-100185/func/sub-100185_task-machinegame_run-01_space-MNI_boldref.nii',
            '53d9b486d08fec5a952f68fcbcddb38a72818d4c',
        ),
        (
            BOLD_PATH,
            ['dseg.nii'],
            {'space': 'MNI', 'suffix': 'dseg', 'desc': 'aseg'},
            'sub-100185/func/sub-100185_task-machinegame_run-01_space-MNI_desc-aseg_dseg.nii',
            'ddadc9be8224eebe0177a65bf87300f275e17e96',
        ),
        (
            BOLD_PATH,
            ['mask.nii'],
            {'space': 'MNI', 'suffix': 'mask', 'desc': 'brain'},
            'sub-100185/func/sub-100185_task-machinegame_run-01_space-MNI_desc-brain_mask.nii',
            'f97a1877508139b42ea9fc476bdba367b001ab00',
        ),
        (
            BOLD_PATH,
            ['bold.nii'],
            {'space': 'MNI', 'desc': 'preproc'},
            'sub-100185/func/sub-100185_task-machinegame_run-01_space-MNI_desc-preproc_bold.nii',
            'aa1eed935e6a8dcca646b0c78ee57218e30e2974',
        ),
        # Nondeterministic order - do we really need this to work, or we can stay safe with
        # MapNodes?
        # (T1W_PATH, [f"{s}-{l}.nii.gz" for s in ("MNIa", "MNIb") for l in ("GM", "WM", "CSF")],
        #     {"space": ["MNIa", "MNIb"], "label": ["GM", "WM", "CSF"], "suffix": "probseg"},
        #     [f"sub-100185/anat/sub-100185_space-{s}_label-{l}_probseg.nii.gz"
        #      for s in ("MNIa", "MNIb") for l in ("GM", "WM", "CSF")]),
        (
            T1W_PATH,
            ['anat.html'],
            {'desc': 'conform', 'datatype': 'figures'},
            'sub-100185/figures/sub-100185_desc-conform_T1w.html',
            'da39a3ee5e6b4b0d3255bfef95601890afd80709',
        ),
        (
            BOLD_PATH,
            ['aroma.csv'],
            {'suffix': 'AROMAnoiseICs', 'extension': 'h5'},
            ValueError,
            None,
        ),
        (
            T1W_PATH,
            ['anat.nii.gz'] * 3,
            {'desc': 'preproc', 'space': 'MNI'},
            ValueError,
            None,
        ),
        (
            'sub-07/ses-preop/anat/sub-07_ses-preop_T1w.nii.gz',
            ['tfm.h5'],
            {'from': 'orig', 'to': 'target', 'suffix': 'xfm'},
            'sub-07/ses-preop/anat/sub-07_ses-preop_from-orig_to-target_mode-image_xfm.h5',
            'da39a3ee5e6b4b0d3255bfef95601890afd80709',
        ),
        (
            'sub-07/ses-preop/anat/sub-07_ses-preop_run-01_T1w.nii.gz',
            ['tfm.txt'],
            {'from': 'orig', 'to': 'T1w', 'suffix': 'xfm'},
            'sub-07/ses-preop/anat/sub-07_ses-preop_run-01_from-orig_to-T1w_mode-image_xfm.txt',
            'da39a3ee5e6b4b0d3255bfef95601890afd80709',
        ),
    ],
)
@pytest.mark.parametrize('dismiss_entities', [None, ('run', 'session')])
def test_DerivativesDataSink_build_path(
    tmp_path,
    interface,
    out_path_base,
    source,
    input_files,
    entities,
    expectation,
    checksum,
    dismiss_entities,
):
    """Check a few common derivatives generated by NiPreps."""
    if interface is bintfs.PrepareDerivative and out_path_base is not None:
        pytest.skip('PrepareDerivative does not support out_path_base')
    ds_inputs = []
    for input_file in input_files:
        fname = tmp_path / input_file
        if fname.name.endswith('.dtseries.nii'):
            axes = (
                nb.cifti2.SeriesAxis(start=0, step=2, size=20),
                nb.cifti2.BrainModelAxis.from_mask(np.ones((5, 5, 5))),
            )
            hdr = nb.cifti2.cifti2_axes.to_header(axes)
            cifti = nb.Cifti2Image(
                np.zeros(hdr.matrix.get_data_shape(), dtype=np.float32), header=hdr
            )
            cifti.nifti_header.set_intent('ConnDenseSeries')
            cifti.to_filename(fname)
        elif fname.name.rstrip('.gz').endswith('.nii'):
            hdr = nb.Nifti1Header()
            hdr.set_qform(np.eye(4), code=2)
            hdr.set_sform(np.eye(4), code=2)
            units = ('mm', 'sec') if 'bold' in input_file else ('mm',)
            size = (10, 10, 10, 10) if 'bold' in input_file else (10, 10, 10)
            hdr.set_xyzt_units(*units)
            nb.Nifti1Image(np.zeros(size), np.eye(4), hdr).to_filename(fname)
        else:
            fname.write_text('')

        ds_inputs.append(str(fname))

    base_directory = tmp_path / 'output'
    base_directory.mkdir()

    prep, save = make_prep_and_save(
        interface,
        base_directory=str(base_directory),
        out_path_base=out_path_base,
        in_file=ds_inputs,
        source_file=source,
        dismiss_entities=dismiss_entities,
        **entities,
    )
    if isinstance(expectation, type):
        with pytest.raises(expectation):
            prep.run()
        return

    prep_result = prep.run()
    save_result = connect_and_run_save(prep_result, save)

    output = save_result.outputs.out_file
    if isinstance(expectation, str):
        expectation = [expectation]
        output = [output]
        checksum = [checksum]

    if dismiss_entities:
        if 'run' in dismiss_entities:
            expectation = [e.replace('_run-01', '') for e in expectation]

        if 'session' in dismiss_entities:
            expectation = [
                e.replace('_ses-preop', '').replace('ses-preop/', '') for e in expectation
            ]

    base = (out_path_base or 'niworkflows') if interface == bintfs.DerivativesDataSink else ''
    for out, exp in zip(output, expectation):
        assert Path(out).relative_to(base_directory) == Path(base) / exp

    for out, exp in zip(output, expectation):
        assert Path(out).relative_to(base_directory) == Path(base) / exp
        # Regression - some images were given nan scale factors
        if out.endswith(('.nii', '.nii.gz')):
            img = nb.load(out)
            if isinstance(img, nb.Nifti1Image):
                with nb.openers.ImageOpener(out) as fobj:
                    hdr = img.header.from_fileobj(fobj)
                assert not np.isnan(hdr['scl_slope'])
                assert not np.isnan(hdr['scl_inter'])
    for out, chksum in zip(output, checksum):
        if chksum == '335f1394ce90b58bbf27026b6eeec4d2124c11da':
            if Version(nb.__version__) < Version('5.3'):
                # Nibabel 5.3 avoids unnecessary roundtrips for Cifti2Headers
                # Older versions transformed a `SeriesStep="2"` into `SeriesStep="2.0"`
                chksum = 'f7b8755c6ad0d8dcdb60676331b52a23ce288b61'
        assert sha1(Path(out).read_bytes()).hexdigest() == chksum  # noqa: S324


@pytest.mark.parametrize('interface', [bintfs.DerivativesDataSink, bintfs.PrepareDerivative])
def test_DerivativesDataSink_dtseries_json(tmp_path, interface):
    cifti_fname = str(tmp_path / 'test.dtseries.nii')

    axes = (
        nb.cifti2.SeriesAxis(start=0, step=2, size=20),
        nb.cifti2.BrainModelAxis.from_mask(np.ones((5, 5, 5))),
    )
    hdr = nb.cifti2.cifti2_axes.to_header(axes)
    cifti = nb.Cifti2Image(np.zeros(hdr.matrix.get_data_shape(), dtype=np.float32), header=hdr)
    cifti.nifti_header.set_intent('ConnDenseSeries')
    cifti.to_filename(cifti_fname)

    source_file = tmp_path / 'bids' / 'sub-01' / 'func' / 'sub-01_task-rest_bold.nii.gz'
    source_file.parent.mkdir(parents=True)
    source_file.touch()

    prep, save = make_prep_and_save(
        interface,
        base_directory=str(tmp_path),
        out_path_base='',
        in_file=cifti_fname,
        source_file=str(source_file),
        compress=False,
        space='fsLR',
        grayordinates='91k',
        RepetitionTime=2.0,
    )

    prep_result = prep.run()
    save_result = connect_and_run_save(prep_result, save)

    out_path = Path(save_result.outputs.out_file)

    assert out_path.name == 'sub-01_task-rest_space-fsLR_bold.dtseries.nii'
    old_sidecar = out_path.with_name('sub-01_task-rest_space-fsLR_bold.dtseries.json')
    new_sidecar = out_path.with_name('sub-01_task-rest_space-fsLR_bold.json')

    assert not old_sidecar.exists()
    assert new_sidecar.exists()
    assert 'RepetitionTime' in json.loads(new_sidecar.read_text())


@pytest.mark.parametrize('interface', [bintfs.DerivativesDataSink, bintfs.PrepareDerivative])
@pytest.mark.parametrize(
    ('space', 'size', 'units', 'xcodes', 'zipped', 'fixed', 'data_dtype'),
    [
        ('T1w', (30, 30, 30, 10), ('mm', 'sec'), (2, 2), True, [False], None),
        ('T1w', (30, 30, 30, 10), ('mm', 'sec'), (0, 2), True, [True], 'float64'),
        ('T1w', (30, 30, 30, 10), ('mm', 'sec'), (0, 0), True, [True], '<i4'),
        ('T1w', (30, 30, 30, 10), ('mm', None), (2, 2), True, [True], '<f4'),
        ('T1w', (30, 30, 30, 10), (None, None), (0, 2), True, [True], None),
        ('T1w', (30, 30, 30, 10), (None, 'sec'), (0, 0), True, [True], None),
        ('MNI152Lin', (30, 30, 30, 10), ('mm', 'sec'), (4, 4), True, [False], None),
        ('MNI152Lin', (30, 30, 30, 10), ('mm', 'sec'), (0, 2), True, [True], None),
        ('MNI152Lin', (30, 30, 30, 10), ('mm', 'sec'), (0, 0), True, [True], None),
        ('MNI152Lin', (30, 30, 30, 10), ('mm', None), (4, 4), True, [True], None),
        ('MNI152Lin', (30, 30, 30, 10), (None, None), (0, 2), True, [True], None),
        ('MNI152Lin', (30, 30, 30, 10), (None, 'sec'), (0, 0), True, [True], None),
        (None, (30, 30, 30, 10), ('mm', 'sec'), (1, 1), True, [False], None),
        (None, (30, 30, 30, 10), ('mm', 'sec'), (0, 0), True, [True], None),
        (None, (30, 30, 30, 10), ('mm', 'sec'), (0, 2), True, [True], None),
        (None, (30, 30, 30, 10), ('mm', None), (1, 1), True, [True], None),
        (None, (30, 30, 30, 10), (None, None), (0, 2), True, [True], None),
        (None, (30, 30, 30, 10), (None, 'sec'), (0, 0), True, [True], None),
        (None, (30, 30, 30, 10), (None, 'sec'), (0, 0), False, [True], None),
    ],
)
def test_DerivativesDataSink_bold(
    tmp_path, interface, space, size, units, xcodes, zipped, fixed, data_dtype
):
    fname = str(tmp_path / 'source.nii') + ('.gz' if zipped else '')

    hdr = nb.Nifti1Header()
    hdr.set_qform(np.eye(4), code=xcodes[0])
    hdr.set_sform(np.eye(4), code=xcodes[1])
    hdr.set_xyzt_units(*units)
    nb.Nifti1Image(np.zeros(size), np.eye(4), hdr).to_filename(fname)

    # BOLD derivative in T1w space
    prep, _ = make_prep_and_save(
        interface,
        base_directory=str(tmp_path),
        keep_dtype=True,
        data_dtype=data_dtype or Undefined,
        desc='preproc',
        source_file=BOLD_PATH,
        space=space or Undefined,
        in_file=fname,
    )

    prep_result = prep.run()

    nii = nb.load(prep_result.outputs.out_file)
    assert prep_result.outputs.fixed_hdr == fixed
    if data_dtype:
        assert nii.get_data_dtype() == np.dtype(data_dtype)
    assert int(nii.header['qform_code']) == XFORM_CODES[space]
    assert int(nii.header['sform_code']) == XFORM_CODES[space]
    assert nii.header.get_xyzt_units() == ('mm', 'sec')


@pytest.mark.parametrize('interface', [bintfs.DerivativesDataSink, bintfs.PrepareDerivative])
@pytest.mark.parametrize(
    ('space', 'size', 'units', 'xcodes', 'fixed'),
    [
        ('MNI152Lin', (30, 30, 30), ('mm', None), (4, 4), [False]),
        ('MNI152Lin', (30, 30, 30), ('mm', 'sec'), (4, 4), [True]),
        ('MNI152Lin', (30, 30, 30), ('mm', 'sec'), (0, 2), [True]),
        ('MNI152Lin', (30, 30, 30), ('mm', 'sec'), (0, 0), [True]),
        ('MNI152Lin', (30, 30, 30), (None, None), (0, 2), [True]),
        ('MNI152Lin', (30, 30, 30), (None, 'sec'), (0, 0), [True]),
        ('boldref', (30, 30, 30), ('mm', None), (2, 2), [False]),
        ('boldref', (30, 30, 30), ('mm', 'sec'), (2, 2), [True]),
        ('boldref', (30, 30, 30), ('mm', 'sec'), (0, 2), [True]),
        ('boldref', (30, 30, 30), ('mm', 'sec'), (0, 0), [True]),
        ('boldref', (30, 30, 30), (None, None), (0, 2), [True]),
        ('boldref', (30, 30, 30), (None, 'sec'), (0, 0), [True]),
        (None, (30, 30, 30), ('mm', None), (1, 1), [False]),
        (None, (30, 30, 30), ('mm', 'sec'), (1, 1), [True]),
        (None, (30, 30, 30), ('mm', 'sec'), (0, 2), [True]),
        (None, (30, 30, 30), ('mm', 'sec'), (0, 0), [True]),
        (None, (30, 30, 30), (None, None), (0, 2), [True]),
        (None, (30, 30, 30), (None, 'sec'), (0, 0), [True]),
    ],
)
def test_DerivativesDataSink_t1w(tmp_path, interface, space, size, units, xcodes, fixed):
    fname = str(tmp_path / 'source.nii.gz')

    hdr = nb.Nifti1Header()
    hdr.set_qform(np.eye(4), code=xcodes[0])
    hdr.set_sform(np.eye(4), code=xcodes[1])
    hdr.set_xyzt_units(*units)
    nb.Nifti1Image(np.zeros(size), np.eye(4), hdr).to_filename(fname)

    # BOLD derivative in T1w space
    prep, _ = make_prep_and_save(
        interface,
        base_directory=str(tmp_path),
        keep_dtype=True,
        desc='preproc',
        source_file=T1W_PATH,
        space=space or Undefined,
        in_file=fname,
    )

    prep_result = prep.run()

    nii = nb.load(prep_result.outputs.out_file)
    assert prep_result.outputs.fixed_hdr == fixed
    assert int(nii.header['qform_code']) == XFORM_CODES[space]
    assert int(nii.header['sform_code']) == XFORM_CODES[space]
    assert nii.header.get_xyzt_units() == ('mm', 'unknown')


@pytest.mark.parametrize('interface', [bintfs.DerivativesDataSink, bintfs.PrepareDerivative])
@pytest.mark.parametrize(
    'source_file',
    [
        BOLD_PATH,
        [BOLD_PATH],
        [BOLD_PATH, 'ds054/sub-100185/func/sub-100185_task-machinegame_run-02_bold.nii.gz'],
    ],
)
@pytest.mark.parametrize('source_dtype', ['<i4', '<f4'])
@pytest.mark.parametrize('in_dtype', ['<i4', '<f4'])
def test_DerivativesDataSink_data_dtype_source(
    tmp_path, interface, source_file, source_dtype, in_dtype
):
    def make_empty_nii_with_dtype(fname, dtype):
        Path(fname).parent.mkdir(exist_ok=True, parents=True)

        size = (2, 3, 4, 5)

        nb.Nifti1Image(np.zeros(size, dtype=dtype), np.eye(4)).to_filename(fname)

    in_file = str(tmp_path / 'in.nii')
    make_empty_nii_with_dtype(in_file, in_dtype)

    if isinstance(source_file, str):
        source_file = str(tmp_path / source_file)
        make_empty_nii_with_dtype(source_file, source_dtype)

    elif isinstance(source_file, list):
        source_file = [str(tmp_path / s) for s in source_file]
        for s in source_file:
            make_empty_nii_with_dtype(s, source_dtype)

    prep, _save = make_prep_and_save(
        interface,
        base_directory=str(tmp_path),
        data_dtype='source',
        desc='preproc',
        source_file=source_file,
        in_file=in_file,
    )

    prep_result = prep.run()

    nii = nb.load(prep_result.outputs.out_file)
    assert nii.get_data_dtype() == np.dtype(source_dtype)


@pytest.mark.parametrize('interface', [bintfs.DerivativesDataSink, bintfs.PrepareDerivative])
def test_DerivativesDataSink_fmapid(tmp_path, interface):
    """Ascertain #637 is not regressing."""
    source_file = [
        (tmp_path / s)
        for s in [
            'sub-36/fmap/sub-36_dir-1_run-01_epi.nii.gz',
            'sub-36/fmap/sub-36_dir-1_run-02_epi.nii.gz',
            'sub-36/fmap/sub-36_dir-2_run-01_epi.nii.gz',
            'sub-36/fmap/sub-36_dir-2_run-02_epi.nii.gz',
        ]
    ]
    for fname in source_file:
        fname.parent.mkdir(exist_ok=True, parents=True)
        fname.write_text('')

    in_file = tmp_path / 'report.svg'
    in_file.write_text('')

    prep, save = make_prep_and_save(
        interface,
        base_directory=str(tmp_path),
        datatype='figures',
        suffix='fieldmap',
        desc='pepolar',
        dismiss_entities=('fmap',),
        allowed_entities=('fmapid',),
        fmapid='auto00000',
        source_file=[str(s.absolute()) for s in source_file],
        in_file=str(in_file),
    )

    prep_result = prep.run()
    save_result = connect_and_run_save(prep_result, save)

    assert save_result.outputs.out_file.endswith(
        'sub-36_fmapid-auto00000_desc-pepolar_fieldmap.svg'
    )


@pytest.mark.parametrize('interface', [bintfs.DerivativesDataSink, bintfs.PrepareDerivative])
@pytest.mark.parametrize('dtype', ['i2', 'u2', 'f4'])
def test_DerivativesDataSink_values(tmp_path, interface, dtype):
    # We use static checksums above, which ensures we don't break things, but
    # pins the tests to specific values.
    # Here we use random values, check that the values are preserved, and then
    # the checksums are unchanged across two runs.
    fname = str(tmp_path / 'source.nii.gz')
    rng = np.random.default_rng()
    hdr = nb.Nifti1Header()
    hdr.set_qform(np.eye(4), code=1)
    hdr.set_sform(np.eye(4), code=1)
    nb.Nifti1Image(rng.uniform(500, 2000, (5, 5, 5)), np.eye(4), hdr).to_filename(fname)

    orig_data = np.asanyarray(nb.load(fname).dataobj)
    expected = np.rint(orig_data) if dtype[0] in 'iu' else orig_data

    prep, _ = make_prep_and_save(
        interface,
        base_directory=str(tmp_path),
        keep_dtype=True,
        data_dtype=dtype,
        desc='preproc',
        source_file=T1W_PATH,
        in_file=fname,
    )

    prep_result = prep.run()

    out_file = Path(prep_result.outputs.out_file)

    nii = nb.load(out_file)
    assert np.allclose(nii.dataobj, expected)

    checksum = sha1(out_file.read_bytes()).hexdigest()  # noqa: S324
    out_file.unlink()

    # Rerun to ensure determinism with non-zero data
    prep, _ = make_prep_and_save(
        interface,
        base_directory=str(tmp_path),
        keep_dtype=True,
        data_dtype=dtype,
        desc='preproc',
        source_file=T1W_PATH,
        in_file=fname,
    )

    prep_result = prep.run()

    assert sha1(out_file.read_bytes()).hexdigest() == checksum  # noqa: S324


@needs_data_dir
@pytest.mark.parametrize('field', ['RepetitionTime', 'UndefinedField'])
def test_ReadSidecarJSON_connection(testdata_dir, field):
    """
    This test prevents regressions of #333
    """
    from nipype.interfaces import utility as niu
    from nipype.pipeline import engine as pe

    from ..bids import ReadSidecarJSON

    reg_fields = ['RepetitionTime']
    n = pe.Node(ReadSidecarJSON(fields=reg_fields), name='node')
    n.inputs.in_file = str(
        testdata_dir / 'ds054' / 'sub-100185' / 'fmap' / 'sub-100185_phasediff.nii.gz'
    )
    o = pe.Node(niu.IdentityInterface(fields=['out_port']), name='o')
    wf = pe.Workflow(name='json')

    if field in reg_fields:  # This should work
        wf.connect([(n, o, [(field, 'out_port')])])
    else:
        with pytest.raises(Exception, match=r'.*Some connections were not found.*'):
            wf.connect([(n, o, [(field, 'out_port')])])


@pytest.mark.skipif(not os.getenv('FREESURFER_HOME'), reason='No FreeSurfer')
@pytest.mark.parametrize(
    ('derivatives', 'subjects_dir'),
    [
        (os.getenv('FREESURFER_HOME'), 'subjects'),
        ('/tmp', os.path.join(os.getenv('FREESURFER_HOME', ''), 'subjects')),  # noqa: S108
    ],
)
def test_fsdir_noaction(derivatives, subjects_dir):
    """Using $FREESURFER_HOME/subjects should exit early, however constructed"""
    fshome = os.environ['FREESURFER_HOME']
    res = bintfs.BIDSFreeSurferDir(
        derivatives=derivatives, subjects_dir=subjects_dir, freesurfer_home=fshome
    ).run()
    assert res.outputs.subjects_dir == f'{fshome}/subjects'


@pytest.mark.skipif(not os.getenv('FREESURFER_HOME'), reason='No FreeSurfer')
@pytest.mark.parametrize('spaces', [[], ['fsaverage'], ['fsnative'], ['fsaverage5', 'fsnative']])
def test_fsdir(tmp_path, spaces):
    fshome = os.environ['FREESURFER_HOME']
    subjects_dir = tmp_path / 'freesurfer'

    # Verify we're starting clean
    for space in spaces:
        if space.startswith('fsaverage'):
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
            if space.startswith('fsaverage'):
                assert Path.exists(subjects_dir / space)


@pytest.mark.skipif(not os.getenv('FREESURFER_HOME'), reason='No FreeSurfer')
def test_fsdir_missing_space(tmp_path):
    fshome = os.environ['FREESURFER_HOME']

    # fsaverage2 doesn't exist in source or destination, so can't copy
    with pytest.raises(FileNotFoundError):
        bintfs.BIDSFreeSurferDir(
            derivatives=str(tmp_path), spaces=['fsaverage2'], freesurfer_home=fshome
        ).run()

    subjects_dir = tmp_path / 'freesurfer'

    # If fsaverage2 exists in the destination directory, no error is thrown
    Path.mkdir(subjects_dir / 'fsaverage2')
    bintfs.BIDSFreeSurferDir(
        derivatives=str(tmp_path), spaces=['fsaverage2'], freesurfer_home=fshome
    ).run()


@pytest.mark.skipif(not os.getenv('FREESURFER_HOME'), reason='No FreeSurfer')
@pytest.mark.parametrize('min_version', [None, '7.0.0'])
def test_fsdir_min_version(tmp_path, min_version):
    fshome = os.environ['FREESURFER_HOME']
    subjects_dir = tmp_path / 'freesurfer'

    patched_subject_dir = subjects_dir / 'fsaverage' / 'older'
    patched_subject_dir.mkdir(parents=True)

    bfsd = bintfs.BIDSFreeSurferDir(
        subjects_dir=subjects_dir,
        derivatives=str(tmp_path),
        spaces=['fsaverage'],
        freesurfer_home=fshome,
    )

    if min_version:
        bfsd.inputs.minimum_fs_version = min_version

    bfsd.run()
    if min_version:
        # should have been overwritten with proper subjects dir
        assert not patched_subject_dir.exists()
    else:
        assert patched_subject_dir.exists()


def test_BIDSDataGrabber():
    x = bintfs.BIDSDataGrabber(anat_only=True)
    assert x._require_t1w is True
    assert x._require_funcs is False

    x = bintfs.BIDSDataGrabber(anat_only=False, require_t1w=False)
    assert x._require_t1w is False
    assert x._require_funcs is True

    x = bintfs.BIDSDataGrabber(anat_derivatives='derivatives')
    assert x._require_t1w is False
