import os
from pathlib import Path

import numpy as np
import nibabel as nb
import pytest
from nipype.interfaces.base import Undefined

from .. import bids as bintfs


XFORM_CODES = {
    'MNI152Lin': 4,
    'T1w': 2,
    'boldref': 2,
    None: 1,
}

BOLD_PATH = 'ds054/sub-100185/func/sub-100185_task-machinegame_run-01_bold.nii.gz'


@pytest.mark.parametrize('space, size, units, xcodes, zipped, fixed', [
    ('T1w', (30, 30, 30, 10), ('mm', 'sec'), (2, 2), True, [False]),
    ('T1w', (30, 30, 30, 10), ('mm', 'sec'), (0, 2), True, [True]),
    ('T1w', (30, 30, 30, 10), ('mm', 'sec'), (0, 0), True, [True]),
    ('T1w', (30, 30, 30, 10), ('mm', None), (2, 2), True, [True]),
    ('T1w', (30, 30, 30, 10), (None, None), (0, 2), True, [True]),
    ('T1w', (30, 30, 30, 10), (None, 'sec'), (0, 0), True, [True]),
    ('MNI152Lin', (30, 30, 30, 10), ('mm', 'sec'), (4, 4), True, [False]),
    ('MNI152Lin', (30, 30, 30, 10), ('mm', 'sec'), (0, 2), True, [True]),
    ('MNI152Lin', (30, 30, 30, 10), ('mm', 'sec'), (0, 0), True, [True]),
    ('MNI152Lin', (30, 30, 30, 10), ('mm', None), (4, 4), True, [True]),
    ('MNI152Lin', (30, 30, 30, 10), (None, None), (0, 2), True, [True]),
    ('MNI152Lin', (30, 30, 30, 10), (None, 'sec'), (0, 0), True, [True]),
    (None, (30, 30, 30, 10), ('mm', 'sec'), (1, 1), True, [False]),
    (None, (30, 30, 30, 10), ('mm', 'sec'), (0, 0), True, [True]),
    (None, (30, 30, 30, 10), ('mm', 'sec'), (0, 2), True, [True]),
    (None, (30, 30, 30, 10), ('mm', None), (1, 1), True, [True]),
    (None, (30, 30, 30, 10), (None, None), (0, 2), True, [True]),
    (None, (30, 30, 30, 10), (None, 'sec'), (0, 0), True, [True]),
    (None, (30, 30, 30, 10), (None, 'sec'), (0, 0), False, [True]),
])
def test_DerivativesDataSink_bold(tmp_path, space, size, units, xcodes, zipped, fixed):
    fname = str(tmp_path / 'source.nii') + ('.gz' if zipped else '')

    hdr = nb.Nifti1Header()
    hdr.set_qform(np.eye(4), code=xcodes[0])
    hdr.set_sform(np.eye(4), code=xcodes[1])
    hdr.set_xyzt_units(*units)
    nb.Nifti1Image(np.zeros(size), np.eye(4), hdr).to_filename(fname)

    # BOLD derivative in T1w space
    dds = bintfs.DerivativesDataSink(
        base_directory=str(tmp_path),
        keep_dtype=True,
        desc='preproc',
        source_file=BOLD_PATH,
        space=space or Undefined,
        in_file=fname,
    ).run()

    nii = nb.load(dds.outputs.out_file)
    assert dds.outputs.fixed_hdr == fixed
    assert int(nii.header['qform_code']) == XFORM_CODES[space]
    assert int(nii.header['sform_code']) == XFORM_CODES[space]
    assert nii.header.get_xyzt_units() == ('mm', 'sec')


T1W_PATH = 'ds054/sub-100185/anat/sub-100185_T1w.nii.gz'


@pytest.mark.parametrize('space, size, units, xcodes, fixed', [
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
])
def test_DerivativesDataSink_t1w(tmp_path, space, size, units, xcodes, fixed):
    fname = str(tmp_path / 'source.nii.gz')

    hdr = nb.Nifti1Header()
    hdr.set_qform(np.eye(4), code=xcodes[0])
    hdr.set_sform(np.eye(4), code=xcodes[1])
    hdr.set_xyzt_units(*units)
    nb.Nifti1Image(np.zeros(size), np.eye(4), hdr).to_filename(fname)

    # BOLD derivative in T1w space
    dds = bintfs.DerivativesDataSink(
        base_directory=str(tmp_path),
        keep_dtype=True,
        desc='preproc',
        source_file=T1W_PATH,
        space=space or Undefined,
        in_file=fname
    ).run()

    nii = nb.load(dds.outputs.out_file)
    assert dds.outputs.fixed_hdr == fixed
    assert int(nii.header['qform_code']) == XFORM_CODES[space]
    assert int(nii.header['sform_code']) == XFORM_CODES[space]
    assert nii.header.get_xyzt_units() == ('mm', 'unknown')


@pytest.mark.parametrize('field', [
    'RepetitionTime',
    'UndefinedField',
])
def test_ReadSidecarJSON_connection(testdata_dir, field):
    """
    This test prevents regressions of #333
    """
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from niworkflows.interfaces.bids import ReadSidecarJSON

    reg_fields = ['RepetitionTime']
    n = pe.Node(ReadSidecarJSON(fields=reg_fields), name='node')
    n.inputs.in_file = str(testdata_dir / 'ds054' / 'sub-100185' / 'fmap' /
                           'sub-100185_phasediff.nii.gz')
    o = pe.Node(niu.IdentityInterface(fields=['out_port']), name='o')
    wf = pe.Workflow(name='json')

    if field in reg_fields:  # This should work
        wf.connect([
            (n, o, [(field, 'out_port')]),
        ])
    else:
        with pytest.raises(Exception, match=r'.*Some connections were not found.*'):
            wf.connect([
                (n, o, [(field, 'out_port')]),
            ])


@pytest.mark.skipif(not os.getenv('FREESURFER_HOME'), reason="No FreeSurfer")
@pytest.mark.parametrize("derivatives, subjects_dir", [
    (os.getenv('FREESURFER_HOME'), 'subjects'),
    ('/tmp', "%s/%s" % (os.getenv('FREESURFER_HOME'), 'subjects'))
    ])
def test_fsdir_noaction(derivatives, subjects_dir):
    """ Using $FREESURFER_HOME/subjects should exit early, however constructed """
    fshome = os.environ['FREESURFER_HOME']
    res = bintfs.BIDSFreeSurferDir(derivatives=derivatives, subjects_dir=subjects_dir,
                                   freesurfer_home=fshome).run()
    assert res.outputs.subjects_dir == '%s/subjects' % fshome


@pytest.mark.skipif(not os.getenv('FREESURFER_HOME'), reason="No FreeSurfer")
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
        res = bintfs.BIDSFreeSurferDir(derivatives=str(tmp_path), spaces=spaces,
                                       freesurfer_home=fshome,
                                       overwrite_fsaverage=overwrite_fsaverage).run()
        assert res.outputs.subjects_dir == str(subjects_dir)

        for space in spaces:
            if space.startswith('fsaverage'):
                assert Path.exists(subjects_dir / space)


@pytest.mark.skipif(not os.getenv('FREESURFER_HOME'), reason="No FreeSurfer")
def test_fsdir_missing_space(tmp_path):
    fshome = os.environ['FREESURFER_HOME']

    # fsaverage2 doesn't exist in source or destination, so can't copy
    with pytest.raises(FileNotFoundError):
        bintfs.BIDSFreeSurferDir(derivatives=str(tmp_path), spaces=['fsaverage2'],
                                 freesurfer_home=fshome).run()

    subjects_dir = tmp_path / 'freesurfer'

    # If fsaverage2 exists in the destination directory, no error is thrown
    Path.mkdir(subjects_dir / 'fsaverage2')
    bintfs.BIDSFreeSurferDir(derivatives=str(tmp_path), spaces=['fsaverage2'],
                             freesurfer_home=fshome).run()
