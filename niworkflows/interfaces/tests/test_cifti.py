import json
from pathlib import Path

import nibabel as nb
import numpy as np
import pytest

from ..cifti import GenerateCifti, CIFTI_STRUCT_WITH_LABELS


@pytest.fixture(scope="module")
def cifti_data():
    import tempfile

    with tempfile.TemporaryDirectory('cifti-data') as tmp:
        out = Path(tmp).absolute()
        volume_file = str(out / "volume.nii.gz")
        left_gii = str(out / "left.gii")
        right_gii = str(out / "right.gii")
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
        grayordinates="91k",
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
