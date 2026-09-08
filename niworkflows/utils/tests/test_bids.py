"""Tests for :mod:`niworkflows.utils.bids`."""

from ..bids import collect_data
from ..testing import generate_bids_skeleton

SKELETON = {
    'dataset_description': {'Name': 'sample', 'BIDSVersion': '1.6.0'},
    '01': [{'anat': [{'suffix': 'T1w', 'metadata': {'EchoTime': 1}}]}],
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
