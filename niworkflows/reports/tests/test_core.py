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
"""Testing module for niworkflows.reports.core"""

import tempfile
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from bids.layout import BIDSLayout
from bids.layout.writing import build_path
from yaml import safe_load as load

from ... import load_resource
from ..core import Report


@pytest.fixture
def bids_sessions(tmpdir_factory):
    f, _ = plt.subplots()
    svg_dir = tmpdir_factory.mktemp('work') / 'fmriprep'
    svg_dir.ensure_dir()

    pattern = (
        'sub-{subject}[/ses-{session}]/{datatype<figures>}/'
        'sub-{subject}[_ses-{session}][_task-{task}][_acq-{acquisition}]'
        '[_ce-{ceagent}][_dir-{direction}][_rec-{reconstruction}]'
        '[_mod-{modality}][_run-{run}][_echo-{echo}][_space-{space}]'
        '[_desc-{desc}]_{suffix<dseg|T1w|bold>}{extension<.svg>}'
    )
    subjects = ['01']
    tasks = ['t1', 't2', 't3']
    runs = ['01', '02', None]
    ces = ['none', 'Gd']
    descs = ['aroma', 'bbregister', 'carpetplot', 'rois']
    # create functional data for both sessions
    ses1_combos = product(subjects, ['1'], tasks, [None], runs, descs)
    ses2_combos = product(subjects, ['2'], tasks, ces, [None], descs)
    # have no runs in the second session (ex: dmriprep test data)
    # https://github.com/nipreps/dmriprep/pull/59
    all_combos = list(ses1_combos) + list(ses2_combos)

    for subject, session, task, ce, run, desc in all_combos:
        entities = {
            'subject': subject,
            'session': session,
            'task': task,
            'ceagent': ce,
            'run': run,
            'desc': desc,
            'extension': '.svg',
            'suffix': 'bold',
            'datatype': 'figures',
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
        anat_entities = {'subject': subject, 'datatype': 'anat', 'suffix': 't1w'}
        anat_entities.update(**anat_opt)
        bids_path = build_path(entities, pattern)
        file_path = svg_dir / bids_path
        file_path.ensure()
        f.savefig(str(file_path))

    return svg_dir.dirname


@pytest.fixture
def example_workdir():
    from ... import data

    workdir = data.load('tests/work')
    if not workdir.is_dir():
        pytest.skip('Missing example workdir; run this test from a source repository')
    return workdir


@pytest.fixture
def test_report1(tmp_path, example_workdir):
    return Report(
        tmp_path,
        'fakeuuid',
        reportlets_dir=example_workdir / 'reportlets',
        subject_id='01',
        packagename='fmriprep',
    )


@pytest.fixture
def test_report2(tmp_path, bids_sessions):
    return Report(
        tmp_path,
        'fakeuuid',
        reportlets_dir=Path(bids_sessions),
        subject_id='01',
        packagename='fmriprep',
    )


@pytest.mark.parametrize(
    ('orderings', 'expected_entities', 'expected_value_combos'),
    [
        (
            ['session', 'task', 'run'],
            ['task', 'run'],
            [
                ('faketask', None),
                ('faketask2', None),
                ('faketaskwithruns', 1),
                ('faketaskwithruns', 2),
                ('mixedgamblestask', 1),
                ('mixedgamblestask', 2),
                ('mixedgamblestask', 3),
            ],
        ),
        (
            ['run', 'task', 'session'],
            ['run', 'task'],
            [
                (None, 'faketask'),
                (None, 'faketask2'),
                (1, 'faketaskwithruns'),
                (1, 'mixedgamblestask'),
                (2, 'faketaskwithruns'),
                (2, 'mixedgamblestask'),
                (3, 'mixedgamblestask'),
            ],
        ),
        ([''], [], []),
        (['session'], [], []),
        ([], [], []),
        (['madeupentity'], [], []),
    ],
)
def test_process_orderings_small(
    test_report1, orderings, expected_entities, expected_value_combos
):
    report = test_report1
    report.init_layout()
    entities, value_combos = report._process_orderings(orderings, report.layout)

    assert entities == expected_entities
    assert expected_value_combos == value_combos


@pytest.mark.parametrize(
    ('orderings', 'expected_entities', 'first_value_combo', 'last_value_combo'),
    [
        (
            ['session', 'task', 'ceagent', 'run'],
            ['session', 'task', 'ceagent', 'run'],
            ('1', 't1', None, None),
            ('2', 't3', 'none', None),
        ),
        (
            ['run', 'task', 'session'],
            ['run', 'task', 'session'],
            (None, 't1', '1'),
            (2, 't3', '1'),
        ),
        ([''], [], None, None),
        (['session'], ['session'], ('1',), ('2',)),
        ([], [], None, None),
        (['madeupentity'], [], None, None),
    ],
)
def test_process_orderings_large(
    test_report2, orderings, expected_entities, first_value_combo, last_value_combo
):
    report = test_report2
    report.init_layout()
    entities, value_combos = report._process_orderings(orderings, report.layout)

    if not value_combos:
        value_combos = [None]

    assert entities == expected_entities
    assert value_combos[0] == first_value_combo
    assert value_combos[-1] == last_value_combo


@pytest.mark.parametrize(
    'ordering',
    [
        ('session'),
        ('task'),
        ('run'),
        ('session,task'),
        ('session,task,run'),
        ('session,task,ceagent,run'),
        ('session,task,acquisition,ceagent,reconstruction,direction,run,echo'),
        ('session,task,run,madeupentity'),
    ],
)
def test_generated_reportlets(bids_sessions, ordering):
    # make independent report
    out_dir = tempfile.mkdtemp()
    report = Report(
        Path(out_dir),
        'fakeuuid',
        reportlets_dir=Path(bids_sessions),
        subject_id='01',
        packagename='fmriprep',
    )
    settings = load(load_resource.readable('reports/default.yml').read_text())
    # change settings to only include some missing ordering
    settings['sections'][3]['ordering'] = ordering
    report.index(settings['sections'])
    # expected number of reportlets
    expected_reportlets_num = len(report.layout.get(extension='.svg'))
    # bids_session uses these entities
    needed_entities = ['session', 'task', 'ceagent', 'run']
    # the last section is the most recently run
    reportlets_num = len(report.sections[-1].reportlets)
    # get the number of figures in the output directory
    out_layout = BIDSLayout(out_dir, config='figures', validate=False)
    out_figs = len(out_layout.get())
    # if ordering does not contain all the relevant entities
    # then there should be fewer reportlets than expected
    if all(ent in ordering for ent in needed_entities):
        assert reportlets_num == expected_reportlets_num == out_figs
    else:
        assert reportlets_num < expected_reportlets_num == out_figs


@pytest.mark.parametrize(
    ('subject_id', 'out_html'),
    [
        ('sub-01', 'sub-01.html'),
        ('sub-sub1', 'sub-sub1.html'),
        ('01', 'sub-01.html'),
        ('sub1', 'sub-sub1.html'),
    ],
)
def test_subject_id(tmp_path, subject_id, out_html):
    reports = tmp_path / 'reports'
    Path(
        reports
        / 'fmriprep'
        / (subject_id if subject_id.startswith('sub-') else f'sub-{subject_id}')
    ).mkdir(parents=True)

    report = Report(
        str(tmp_path),
        'myuniqueid',
        reportlets_dir=reports,
        subject_id=subject_id,
        packagename='fmriprep',
    )
    assert report.subject_id[:4] != 'sub-'
    assert report.out_filename == out_html
