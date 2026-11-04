import nibabel as nb
import nipype.pipeline.engine as pe
import nitransforms as nt
import numpy as np
import pandas as pd
import pytest
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from scipy.spatial.transform import Rotation

from ..confounds import (
    FramewiseDisplacement,
    FSLMotionParams,
    FSLRMSDeviation,
)
from ..itk import MCFLIRT2ITK


def test_FSLRMSDeviation(tmp_path, data_dir):
    base = 'sub-01_task-mixedgamblestask_run-01'
    xfms = data_dir / f'{base}_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt'
    boldref = data_dir / f'{base}_desc-hmc_boldref.nii.gz'
    timeseries = data_dir / f'{base}_desc-motion_timeseries.tsv'

    rmsd = pe.Node(
        FSLRMSDeviation(xfm_file=str(xfms), boldref_file=str(boldref)),
        name='rmsd',
        base_dir=str(tmp_path),
    )
    res = rmsd.run()

    orig = pd.read_csv(timeseries, sep='\t')['rmsd']
    derived = pd.read_csv(res.outputs.out_file, sep='\t')['rmsd']

    # RMSD is nominally in mm, so 0.1um is an acceptable deviation
    assert np.allclose(orig.values, derived.values, equal_nan=True, atol=1e-4)


def test_FSLMotionParams(tmp_path, data_dir):
    base = 'sub-01_task-mixedgamblestask_run-01'
    xfms = data_dir / f'{base}_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt'
    boldref = data_dir / f'{base}_desc-hmc_boldref.nii.gz'
    orig_timeseries = data_dir / f'{base}_desc-motion_timeseries.tsv'

    motion = pe.Node(
        FSLMotionParams(xfm_file=str(xfms), boldref_file=str(boldref)),
        name='fsl_motion',
        base_dir=str(tmp_path),
    )
    res = motion.run()

    derived_params = pd.read_csv(res.outputs.out_file, sep='\t')
    # orig_timeseries includes framewise_displacement
    orig_params = pd.read_csv(orig_timeseries, sep='\t')[derived_params.columns]
    # orig_timeseries has MCFLIRT's signs, on the LAS axes of this boldref
    orig_params[['trans_x', 'rot_x']] *= -1

    # Motion parameters are in mm and rad
    # These are empirically determined bounds, but they seem reasonable
    # for the units
    limits = pd.DataFrame(
        {
            'trans_x': [1e-4],
            'trans_y': [1e-4],
            'trans_z': [1e-4],
            'rot_x': [1e-6],
            'rot_y': [1e-6],
            'rot_z': [1e-6],
        }
    )
    max_diff = (orig_params - derived_params).abs().max()
    assert np.all(max_diff < limits)


def _add_offset(img):
    """The image with 200 added to every voxel, as float32."""
    data = np.float32(img.dataobj) + 200
    return nb.Nifti1Image(data, img.affine, img.header)


def _reorient(img, axcodes):
    """The image stored with voxel axes ``axcodes``."""
    return img.as_reoriented(ornt_transform(io_orientation(img.affine), axcodes2ornt(axcodes)))


def _as_ail(img):
    """The image stored as AIL: L/R on the last voxel axis, with a positive determinant."""
    return _reorient(img, 'AIL')


@pytest.mark.parametrize('axcodes', ['LAS', 'RAS', 'AIL', 'PLS'])
def test_FSLMotionParams_ras(tmp_path, data_dir, axcodes):
    """Columns are RAS+ translations and right-handed rotations about RAS+ axes,
    of the correction that maps each volume onto the reference, however the boldref is stored.
    """
    boldref_file = tmp_path / 'boldref.nii.gz'
    _reorient(
        nb.load(data_dir / 'sub-01_task-mixedgamblestask_run-01_desc-hmc_boldref.nii.gz'),
        axcodes,
    ).to_filename(boldref_file)

    translation = nb.affines.from_matvec(np.eye(3), [1, 2, 3])
    rotations = [
        nb.affines.from_matvec(Rotation.from_rotvec(0.05 * np.eye(3)[axis]).as_matrix(), [0, 0, 0])
        for axis in range(3)
    ]
    corrections = np.stack([translation, *rotations])

    # nitransforms' affines map reference coordinates to volume coordinates
    xfm_file = tmp_path / 'hmc.txt'
    nt.linear.LinearTransformsMapping(
        np.linalg.inv(corrections), reference=boldref_file
    ).to_filename(xfm_file, fmt='itk')

    motion = pe.Node(
        FSLMotionParams(xfm_file=str(xfm_file), boldref_file=str(boldref_file)),
        name='fsl_motion',
        base_dir=str(tmp_path),
    )
    derived = pd.read_csv(motion.run().outputs.out_file, sep='\t')
    trans = derived[['trans_x', 'trans_y', 'trans_z']].to_numpy()
    rot = derived[['rot_x', 'rot_y', 'rot_z']].to_numpy()

    assert np.allclose(trans[0], [1.0, 2.0, 3.0])
    assert np.allclose(rot, np.r_[[np.zeros(3)], 0.05 * np.eye(3)])


@pytest.mark.parametrize(
    ('mcflirt_dir', 'transform', 'axes', 'signs'),
    [
        pytest.param('mcflirt_large_motion', None, [0, 1, 2], [-1, 1, 1], id='las'),
        pytest.param(
            'mcflirt_large_motion', nb.as_closest_canonical, [0, 1, 2], [-1, 1, 1], id='ras'
        ),
        pytest.param(
            'mcflirt_large_motion_offset', _add_offset, [0, 1, 2], [-1, 1, 1], id='offset'
        ),
        # FSL reverses the first voxel axis of AIL, so its axes are P, I and L
        pytest.param('mcflirt_large_motion_ail', _as_ail, [2, 0, 1], [-1, -1, -1], id='ail'),
    ],
)
def test_FSLMotionParams_mcflirt(tmp_path, data_dir, mcflirt_dir, transform, axes, signs):
    """FSLMotionParams reproduces MCFLIRT's .par from MCFLIRT's matrices, on RAS+ axes.

    MCFLIRT's axes are FSL's, which follow the voxel grid. RAS+ axis ``i`` is
    ``signs[i]`` times MCFLIRT's axis ``axes[i]``, for rotations and translations alike.

    The fixtures are MCFLIRT outputs for a synthetic series with rotations
    up to 0.1 rad, registered to the test boldref (or a variant of it).
    See ``tools/generate_mcflirt_fixtures.py``.
    """
    boldref_file = data_dir / 'sub-01_task-mixedgamblestask_run-01_desc-hmc_boldref.nii.gz'
    if transform is not None:
        boldref = transform(nb.load(boldref_file))
        boldref_file = tmp_path / 'boldref.nii.gz'
        boldref.to_filename(boldref_file)

    mcflirt_out = data_dir / mcflirt_dir
    fsl2itk = pe.Node(
        MCFLIRT2ITK(
            in_files=[str(mat) for mat in sorted(mcflirt_out.glob('MAT_*'))],
            in_reference=str(boldref_file),
            in_source=str(boldref_file),
        ),
        name='fsl2itk',
        base_dir=str(tmp_path),
    )
    xfm_file = fsl2itk.run().outputs.out_file

    motion = pe.Node(
        FSLMotionParams(xfm_file=xfm_file, boldref_file=str(boldref_file)),
        name='fsl_motion',
        base_dir=str(tmp_path),
    )
    derived = pd.read_csv(motion.run().outputs.out_file, sep='\t')

    # MCFLIRT's .par columns: rot_x rot_y rot_z (rad), trans_x trans_y trans_z (mm)
    par = np.loadtxt(mcflirt_out / 'mcf.par')
    rot = derived[['rot_x', 'rot_y', 'rot_z']].to_numpy()
    trans = derived[['trans_x', 'trans_y', 'trans_z']].to_numpy()

    # MCFLIRT writes its matrices to 6 decimal places, which leaves about 1e-4 mm
    assert np.allclose(rot, signs * par[:, :3][:, axes], atol=1e-5)
    assert np.allclose(trans, signs * par[:, 3:][:, axes], atol=1e-3)


def test_FramewiseDisplacement(tmp_path, data_dir):
    timeseries = data_dir / 'sub-01_task-mixedgamblestask_run-01_desc-motion_timeseries.tsv'

    framewise_displacement = pe.Node(
        FramewiseDisplacement(in_file=str(timeseries)),
        name='framewise_displacement',
        base_dir=str(tmp_path),
    )
    res = framewise_displacement.run()

    orig = pd.read_csv(timeseries, sep='\t')['framewise_displacement']
    derived = pd.read_csv(res.outputs.out_file, sep='\t')['FramewiseDisplacement']

    assert np.allclose(orig.values, derived.values, equal_nan=True)
