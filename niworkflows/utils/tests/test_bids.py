"""Tests for :mod:`niworkflows.utils.bids`."""

from pathlib import Path

import pytest

from ..bids import collect_data
from ..testing import generate_bids_skeleton

SKELETON = {
    'dataset_description': {'Name': 'sample', 'BIDSVersion': '1.6.0'},
    '01': [{'anat': [{'suffix': 'T1w', 'metadata': {'EchoTime': 1}}]}],
}

SESSIONS_SKELETON = {
    '01': [
        {
            'session': session,
            'anat': [{'suffix': 'T1w'}],
            'func': [{'task': 'rest', 'suffix': 'bold', 'metadata': {'RepetitionTime': 2.0}}],
        }
        for session in ('anat', 'func1', 'func2')
    ],
}


def test_collect_data_ignores_filter_without_query(tmp_path):
    """A filter naming a suffix absent from ``queries`` is dropped, not an error."""
    root = tmp_path / 'bids'
    generate_bids_skeleton(root, SKELETON)

    subj_data, _ = collect_data(
        root,
        '01',
        bids_validate=False,
        queries={'t1w': {'datatype': 'anat', 'suffix': 'T1w'}},
        bids_filters={'pet': {'session': '15'}},
    )
    assert len(subj_data['t1w']) == 1


@pytest.mark.parametrize(
    ('session_id', 'bold_session', 't1w', 'bold'),
    [
        (None, 'func2', ['anat', 'func1', 'func2'], ['func2']),
        (['anat', 'func1'], 'func1', ['anat', 'func1'], ['func1']),
        ('func1', 'func1', ['func1'], ['func1']),
    ],
    ids=['no_session_id', 'narrow_sessions', 'clobber_session'],
)
def test_collect_data_session_filters(tmp_path, session_id, bold_session, t1w, bold):
    """BIDS filters narrow down the requested sessions for their query only."""
    root = tmp_path / 'bids'
    generate_bids_skeleton(root, SESSIONS_SKELETON)

    subj_data, _ = collect_data(
        root,
        '01',
        session_id=session_id,
        bids_validate=False,
        bids_filters={'bold': {'session': bold_session}},
    )

    def _sessions(files):
        return [Path(f).parts[-3].removeprefix('ses-') for f in files]

    assert _sessions(subj_data['t1w']) == t1w
    assert _sessions(subj_data['bold']) == bold


def test_collect_data_session_filter_conflict(tmp_path):
    """BIDS filters cannot select sessions outside of the requested ones."""
    root = tmp_path / 'bids'
    generate_bids_skeleton(root, SESSIONS_SKELETON)

    with pytest.raises(ValueError, match='Conflicting entities for "session"'):
        collect_data(
            root,
            '01',
            session_id=['anat', 'func1'],
            bids_validate=False,
            bids_filters={'bold': {'session': 'func2'}},
        )
