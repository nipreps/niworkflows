import nipype.pipeline.engine as pe
import numpy as np
import pandas as pd

from ..confounds import (
    FramewiseDisplacement,
    FSLMotionParams,
    FSLRMSDeviation,
)


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
