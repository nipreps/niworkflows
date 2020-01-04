''' Testing module for niworkflows.reports.core '''

import os
from pathlib import Path
from pkg_resources import resource_filename
import tempfile

import pytest

from ..core import Report


@pytest.fixture
def test_report():
    test_data_path = resource_filename(
        'niworkflows',
        os.path.join('data', 'tests', 'work', 'reportlets'))
    out_dir = tempfile.mkdtemp()

    return Report(Path(test_data_path), Path(out_dir), 'fakeiuud',
                  subject_id='01', packagename='fmriprep')


@pytest.mark.parametrize(
    "orderings,expected_entities,expected_value_combos",
    [
        (['session', 'task', 'run'],
         ['task', 'run'],
         [
            ('faketask', None),
            ('mixedgamblestask', 1),
            ('mixedgamblestask', 2),
            ('mixedgamblestask', 3),
        ]),
        (['run', 'task', 'session'],
         ['run', 'task'],
         [
            (None, 'faketask'),
            (1, 'mixedgamblestask'),
            (2, 'mixedgamblestask'),
            (3, 'mixedgamblestask'),
        ]),
        ([''],
         [],
         []),
        (['session'],
         [],
         []),
        ([],
         [],
         [],),
        (['madeupentity'],
         [],
         [],),
    ]
)
def test_process_orderings(test_report, orderings,
                           expected_entities, expected_value_combos):
    test_report.init_layout()
    entities, value_combos = test_report._process_orderings(orderings, test_report.layout)

    assert entities == expected_entities
    assert expected_value_combos == value_combos
