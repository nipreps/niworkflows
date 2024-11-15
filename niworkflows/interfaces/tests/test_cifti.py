import json
from pathlib import Path
from unittest import mock

import nibabel as nb
import numpy as np
import pytest

from ..cifti import CIFTI_STRUCT_WITH_LABELS, GenerateCifti, _create_cifti_image


@pytest.fixture(scope='module')
def cifti_data():
    import tempfile

    with tempfile.TemporaryDirectory('cifti-data') as tmp:
        out = Path(tmp).absolute()
        volume_file = str(out / 'volume.nii.gz')
        left_gii = str(out / 'left.gii')
        right_gii = str(out / 'right.gii')
        surface_data = [nb.gifti.GiftiDataArray(np.ones(32492, dtype='i4')) for _ in range(4)]
        vol = nb.Nifti1Image(np.ones((91, 109, 91, 4)), np.eye(4))
        gii = nb.GiftiImage(darrays=surface_data)

        vol.to_filename(volume_file)
        gii.to_filename(left_gii)
        gii.to_filename(right_gii)
        yield volume_file, left_gii, right_gii


def test_GenerateCifti(tmpdir, cifti_data):
    tmpdir.chdir()

    bold_volume = cifti_data[0]
    bold_surfaces = list(cifti_data[1:])

    gen = GenerateCifti(
        bold_file=bold_volume,
        surface_bolds=bold_surfaces,
        grayordinates='91k',
        TR=1,
    )
    res = gen.run().outputs

    cifti = nb.load(res.out_file)
    assert cifti.shape == (4, 91282)
    matrix = cifti.header.matrix
    assert matrix.mapped_indices == [0, 1]
    series_map = matrix.get_index_map(0)
    bm_map = matrix.get_index_map(1)
    assert series_map.indices_map_to_data_type == 'CIFTI_INDEX_TYPE_SERIES'
    assert bm_map.indices_map_to_data_type == 'CIFTI_INDEX_TYPE_BRAIN_MODELS'
    assert len(list(bm_map.brain_models)) == len(CIFTI_STRUCT_WITH_LABELS)

    metadata = json.loads(Path(res.out_metadata).read_text())
    assert 'Density' in metadata
    assert 'SpatialReference' in metadata
    for key in ('VolumeReference', 'CIFTI_STRUCTURE_LEFT_CORTEX', 'CIFTI_STRUCTURE_RIGHT_CORTEX'):
        assert key in metadata['SpatialReference']


def test__create_cifti_image(tmp_path):
    bold_data = np.arange(8, dtype='f4').reshape((2, 2, 2, 1), order='F')
    LAS = [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    bold_img = nb.Nifti1Image(bold_data, LAS)
    label_img = nb.Nifti1Image(np.full((2, 2, 2), 16, 'u1'), LAS)

    bold_file = tmp_path / 'bold.nii'
    volume_label = tmp_path / 'label.nii'
    bold_img.to_filename(bold_file)
    label_img.to_filename(volume_label)

    # Only add one structure to the CIFTI file
    with mock.patch(
        'niworkflows.interfaces.cifti.CIFTI_STRUCT_WITH_LABELS',
        {'CIFTI_STRUCTURE_BRAIN_STEM': (16,)},
    ):
        dummy_fnames = ('', '')
        cifti_file = _create_cifti_image(bold_file, volume_label, dummy_fnames, dummy_fnames, 2.0)

    cimg = nb.load(cifti_file)
    series, bm = (cimg.header.get_axis(i) for i in (0, 1))
    assert len(series) == 1  # Time
    assert len(bm) == 8  # Voxel

    # Maintaining Fortran ordering, data comes out as it went in
    assert np.array_equal(cimg.get_fdata(), bold_data.reshape((1, 8), order='F'))

    # Brain model voxels are indexed in Fortran order (fastest first)
    assert np.array_equal(bm.voxel[:4], [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
