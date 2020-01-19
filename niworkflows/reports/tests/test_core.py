''' Testing module for niworkflows.reports.core '''

import os
from pathlib import Path
from pkg_resources import resource_filename as pkgrf
import tempfile
from itertools import product
from yaml import safe_load as load

import matplotlib.pyplot as plt
from bids.layout.writing import build_path

import pytest

from ..core import Report


@pytest.fixture()
def bids_sessions(tmpdir_factory):
    f, _ = plt.subplots()
    svg_dir = tmpdir_factory.mktemp('work') / 'fmriprep'
    svg_dir.ensure_dir()

    pattern = (
        "sub-{subject}[/ses-{session}]/{datatype<anat|func>}/"
        "sub-{subject}[_ses-{session}][_task-{task}][_acq-{acquisition}]"
        "[_ce-{contrast}][_dir-{direction}][_rec-{reconstruction}]"
        "[_mod-{modality}][_run-{run}][_echo-{echo}][_space-{space}]"
        "[_desc-{desc}]_{suffix<dseg|T1w|bold>}.{extension<svg>}"
    )
    subjects = ['01']
    tasks = ['t1', 't2', 't3']
    runs = ['01', '02', None]
    descs = ['aroma', 'bbregister', 'carpetplot', 'rois']
    # create functional data for both sessions
    ses1_combos = product(subjects, ['1'], tasks, runs, descs)
    ses2_combos = product(subjects, ['2'], tasks, [None], descs)
    # have no runs in the second session (ex: dmriprep test data)
    # https://github.com/nipreps/dmriprep/pull/59
    all_combos = list(ses1_combos) + list(ses2_combos)

    for subject, session, task, run, desc in all_combos:
        entities = {
            'subject': subject,
            'session': session,
            'task': task,
            'run': run,
            'desc': desc,
            'extension': 'svg',
            'suffix': 'bold',
            'datatype': 'func'
        }
        bids_path = build_path(entities, pattern)
        file_path = svg_dir / bids_path
        file_path.ensure()
        f.savefig(str(file_path))

    # create anatomical data
    anat_opts = [
        {'desc': 'brain'},
        {'desc': 'conform'},
        {'desc': 'reconall'},
        {'desc': 'rois'},
        {'suffix': 'dseg'},
        {'space': 'MNI152NLin6Asym'},
        {'space': 'MNI152NLin2009cAsym'},
    ]
    anat_combos = product(subjects, anat_opts)
    for subject, anat_opt in anat_combos:
        anat_entities = {
            "subject": subject,
            "datatype": 'anat',
            "suffix": 't1w'
        }
        anat_entities.update(**anat_opt)
        bids_path = build_path(entities, pattern)
        file_path = svg_dir / bids_path
        file_path.ensure()
        f.savefig(str(file_path))

    return svg_dir.dirname


@pytest.fixture()
def test_report1():
    test_data_path = pkgrf(
        'niworkflows',
        os.path.join('data', 'tests', 'work', 'reportlets'))
    out_dir = tempfile.mkdtemp()

    return Report(Path(test_data_path), Path(out_dir), 'fakeiuud',
                  subject_id='01', packagename='fmriprep')


@pytest.fixture()
def test_report2(bids_sessions):
    out_dir = tempfile.mkdtemp()
    return Report(Path(bids_sessions), Path(out_dir), 'fakeiuud',
                  subject_id='01', packagename='fmriprep')


@pytest.mark.parametrize(
    "orderings,expected_entities,expected_value_combos",
    [
        (['session', 'task', 'run'],
         ['task', 'run'],
         [
            ('faketask', None),
            ('faketask2', None),
            ('faketaskwithruns', 1),
            ('faketaskwithruns', 2),
            ('mixedgamblestask', 1),
            ('mixedgamblestask', 2),
            ('mixedgamblestask', 3),
        ]),
        (['run', 'task', 'session'],
         ['run', 'task'],
         [
            (None, 'faketask'),
            (None, 'faketask2'),
            (1, 'faketaskwithruns'),
            (1, 'mixedgamblestask'),
            (2, 'faketaskwithruns'),
            (2, 'mixedgamblestask'),
            (3, 'mixedgamblestask'),
        ]),
        ([''], [], []),
        (['session'], [], []),
        ([], [], []),
        (['madeupentity'], [], []),
    ]
)
def test_process_orderings_small(test_report1, orderings,
                                 expected_entities, expected_value_combos):
    report = test_report1
    report.init_layout()
    entities, value_combos = report._process_orderings(orderings, report.layout)

    assert entities == expected_entities
    assert expected_value_combos == value_combos


@pytest.mark.parametrize(
    "orderings,expected_entities,first_value_combo,last_value_combo",
    [
        (['session', 'task', 'run'],
         ['session', 'task', 'run'],
         ('1', 't1', None),
         ('2', 't3', None),
         ),
        (['run', 'task', 'session'],
         ['run', 'task', 'session'],
         (None, 't1', '1'),
         (2, 't3', '1'),
         ),
        ([''], [], None, None),
        (['session'], ['session'], ('1',), ('2',)),
        ([], [], None, None),
        (['madeupentity'], [], None, None),
    ]
)
def test_process_orderings_large(test_report2, orderings,
                                 expected_entities, first_value_combo,
                                 last_value_combo):
    report = test_report2
    report.init_layout()
    entities, value_combos = report._process_orderings(orderings, report.layout)

    if not value_combos:
        value_combos = [None]

    assert entities == expected_entities
    assert value_combos[0] == first_value_combo
    assert value_combos[-1] == last_value_combo


@pytest.mark.parametrize(
    "ordering",
    [
        ("session"),
        ("task"),
        ("run"),
        ("session,task"),
        ("session,task,run"),
        ("session,task,acquisition,ceagent,reconstruction,direction,run,echo"),
        ("session,task,run,madeupentity"),
    ])
def test_generated_reportlets(bids_sessions, ordering):
    # make independent report
    out_dir = tempfile.mkdtemp()
    report = Report(Path(bids_sessions), Path(out_dir), 'fakeiuud',
                    subject_id='01', packagename='fmriprep')
    config = Path(pkgrf('niworkflows', 'reports/fmriprep.yml'))
    settings = load(config.read_text())
    # change settings to only include some missing ordering
    settings['sections'][3]['ordering'] = ordering
    report.index(settings['sections'])
    # expected number of reportlets
    expected_reportlets_num = len(report.layout.get(extension='svg'))
    # bids_session uses these entities
    needed_entities = ['session', 'task', 'run']
    # the last section is the most recently run
    reportlets_num = len(report.sections[-1].reportlets)
    # if ordering does not contain all the relevent entities
    # then there should be fewer reportlets than expected
    if all(ent in ordering for ent in needed_entities):
        assert reportlets_num == expected_reportlets_num
    else:
        assert reportlets_num < expected_reportlets_num
