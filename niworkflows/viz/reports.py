#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Reports builder for BIDS-Apps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


"""
from pathlib import Path
import json
import re
from itertools import product
from collections import defaultdict
from pkg_resources import resource_filename as pkgrf
from bids.layout import BIDSLayout, add_config_paths

import jinja2
from nipype.utils.filemanip import copyfile

add_config_paths(figures=pkgrf('niworkflows', 'viz/figures.json'))
LAYOUT_GET = defaultdict(str('s').format, [('echo', 'es')])
SVG_SNIPPET = """\
<object class="svg-reportlet" type="image/svg+xml" data="./{0}">
Problem loading figure {0}. If the link below works, please try \
reloading the report in your browser.</object>
</div>
<div class="elem-filename">
    Get figure file: <a href="./{0}" target="_blank">{0}</a>
</div>
"""


class Element(object):
    """
    Just a basic component of a report
    """

    def __init__(self, name, title=None):
        self.name = name
        self.title = title


class Reportlet(Element):
    """
    A reportlet has title, description and a list of components with either an
    HTML fragment or a path to an SVG file, and possibly a caption. This is a
    factory class to generate Reportlets reusing the layout from a ``Report``
    object.

    .. testsetup::

    >>> from shutil import copytree
    >>> from tempfile import TemporaryDirectory
    >>> from bids.layout import BIDSLayout
    >>> new_path = Path(__file__).resolve().parent.parent
    >>> test_data_path = new_path / 'data' / 'tests' / 'work'
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)  #noqa
    >>> testdir = Path().resolve()
    >>> data_dir = copytree(test_data_path, testdir / 'work')
    >>> out_figs = (testdir / 'out' / 'fmriprep' )
    >>> bl = BIDSLayout(str(testdir / 'work' / 'reportlets'),
    ...                 config='figures', validate=False)

    .. doctest::

    >>> bl.get(subject='01', desc='reconall')
    [<BIDSFile filename='fmriprep/sub-01/anat/sub-01_desc-reconall_T1w.svg'>]

    >>> len(bl.get(subject='01', space='.*'))
    2

    >>> r = Reportlet(bl, out_figs, config={
    ...     'title': 'Some Title', 'bids': {'datatype': 'anat', 'desc': 'reconall'},
    ...     'description': 'Some description'})
    >>> r.name
    'datatype-anat_desc-reconall'

    >>> r.components[0][0].startswith('<object')
    True

    >>> r = Reportlet(bl, out_figs, config={
    ...     'title': 'Some Title', 'bids': {'datatype': 'anat', 'desc': 'summary'},
    ...     'description': 'Some description'})

    >>> r.components[0][0].startswith('<h3')
    True

    >>> r.components[0][1] is None
    True

    >>> r = Reportlet(bl, out_figs, config={
    ...     'title': 'Some Title', 'bids': {'datatype': 'anat', 'space': '.*'},
    ...     'caption': 'Some description {space}'})
    >>> r.components[0][1]
    'Some description MNI152NLin2009cAsym'

    >>> r.components[1][1]
    'Some description MNI152NLin6Asym'


    >>> r = Reportlet(bl, out_figs, config={
    ...     'title': 'Some Title', 'bids': {'datatype': 'fmap', 'space': '.*'},
    ...     'caption': 'Some description {space}'})
    >>> r.is_empty()
    True

    .. testcleanup::

    >>> tmpdir.cleanup()

    """

    def __init__(self, layout, out_dir, config=None):
        if not config:
            raise RuntimeError('Reportlet must have a config object')

        self.name = config.get('name')
        if not self.name:
            self.name = '_'.join('%s-%s' % i for i in config['bids'].items())

        self.title = config.get('title')
        self.description = config.get('description')

        # Query the BIDS layout of reportlets
        files = layout.get(**config['bids'])

        self.components = []
        for bidsfile in files:
            src = Path(bidsfile.path)
            ext = ''.join(src.suffixes)
            desc_text = config.get('caption')

            contents = None
            if ext == '.html':
                contents = src.read_text().strip()
            elif ext == '.svg':
                entities = bidsfile.entities
                if desc_text:
                    desc_text = desc_text.format(**entities)

                entities['extension'] = 'svg'
                entities['datatype'] = 'figures'
                linked_svg = layout.build_path(entities)
                out_file = out_dir / linked_svg
                out_file.parent.mkdir(parents=True, exist_ok=True)
                copyfile(src, out_file, copy=True, use_hardlink=True)
                contents = SVG_SNIPPET.format(linked_svg)

            if contents:
                self.components.append((contents, desc_text))

    def is_empty(self):
        return len(self.components) == 0


class SubReport(Element):
    """
    SubReports are sections within a Report
    """

    def __init__(self, name, reportlets=None, title=''):
        self.name = name
        self.title = title
        self.reportlets = []
        if reportlets:
            self.reportlets += reportlets


class Report(object):
    """
    The full report object. This object maintains a BIDSLayout to index
    all reportlets.


    .. testsetup::

    >>> from shutil import copytree
    >>> from tempfile import TemporaryDirectory
    >>> from bids.layout import BIDSLayout
    >>> new_path = Path(__file__).resolve().parent.parent
    >>> test_data_path = new_path / 'data' / 'tests' / 'work'
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)  #noqa
    >>> testdir = Path().resolve()
    >>> data_dir = copytree(test_data_path, testdir / 'work')
    >>> out_figs = (testdir / 'out' / 'fmriprep' )

    .. doctest::

    >>> robj = Report(testdir / 'work' / 'reportlets',
    ...               'work/config.json', testdir / 'out',
    ...               'madeoutuuid', subject_id='01')
    >>> robj.layout.get(subject='01', desc='reconall')
    [<BIDSFile filename='fmriprep/sub-01/anat/sub-01_desc-reconall_T1w.svg'>]

    >>> robj.generate_report()
    0
    >>> len((testdir / 'out' / 'niworkflows' / 'sub-01.html').read_text())
    9014

    """

    def __init__(self, reportlets_dir, config, out_dir, run_uuid,
                 subject_id=None, out_filename='report.html',
                 packagename=None):
        self.root = reportlets_dir

        # Initialize a BIDS layout
        self.layout = BIDSLayout(self.root, config='figures', validate=False)

        # Initialize structuring elements
        self.sections = []
        self.errors = []
        self.out_dir = Path(out_dir)
        self.out_filename = out_filename
        self.run_uuid = run_uuid
        self.template_path = None
        self.packagename = packagename
        self.subject_id = subject_id
        if subject_id is not None and subject_id.startswith('sub-'):
            self.subject_id = self.subject_id[4:]

        if self.subject_id is not None:
            self.out_filename = 'sub-{}.html'.format(self.subject_id)

        self._load_config(Path(config))

    def _load_config(self, config):
        settings = json.loads(config.read_text())
        self.packagename = self.packagename or settings.get('package', None)

        if self.packagename is not None:
            self.root = self.root / self.packagename
            self.out_dir = self.out_dir / self.packagename

        if self.subject_id is not None:
            self.root = self.root / 'sub-{}'.format(self.subject_id)

        template_path = Path(settings.get('template_path', 'report.tpl'))
        if not template_path.is_absolute():
            template_path = config.parent / template_path
        self.template_path = template_path.resolve()
        self.index(settings['sections'])

    def index(self, config):
        for subrep_cfg in config:
            orderings = [s for s in subrep_cfg.get('ordering', '').strip().split(',') if s]

            queries = []
            for key in orderings:
                values = getattr(self.layout, 'get_%s%s' % (key, LAYOUT_GET[key]))()
                if values:
                    queries.append((key, values))

            if not queries:
                reportlets = [Reportlet(self.layout, self.out_dir, config=cfg)
                              for cfg in subrep_cfg['reportlets']]
            else:
                reportlets = []
                entities, values = zip(*queries)
                combinations = list(product(*values))

                for c in combinations:
                    for cfg in subrep_cfg['reportlets']:
                        cfg['bids'].update({entities[i]: c[i] for i in range(len(c))})
                        reportlets.append(
                            Reportlet(self.layout, self.out_dir, config=cfg)
                        )

            # Filter out empty reportlets
            reportlets = [r for r in reportlets if not r.is_empty()]
            if reportlets:
                sub_report = SubReport(
                    subrep_cfg['name'], reportlets=reportlets,
                    title=subrep_cfg.get('title'))
                self.sections.append(sub_report)

        error_dir = self.out_dir / self.packagename / 'sub-{}'.format(self.subject_id) / \
            'log' / self.run_uuid
        if error_dir.is_dir():
            from ..utils.misc import read_crashfile
            self.errors = [read_crashfile(str(f)) for f in error_dir.glob('crash*.*')]

    def generate_report(self):
        logs_path = self.out_dir / 'logs'

        boilerplate = []
        boiler_idx = 0

        if (logs_path / 'CITATION.html').exists():
            text = (logs_path / 'CITATION.html').read_text(encoding='UTF-8')
            text = '<div class="boiler-html">%s</div>' % re.compile(
                '<body>(.*?)</body>',
                re.DOTALL | re.IGNORECASE).findall(text)[0].strip()
            boilerplate.append((boiler_idx, 'HTML', text))
            boiler_idx += 1

        if (logs_path / 'CITATION.md').exists():
            text = '<pre>%s</pre>\n' % (logs_path / 'CITATION.md').read_text(encoding='UTF-8')
            boilerplate.append((boiler_idx, 'Markdown', text))
            boiler_idx += 1

        if (logs_path / 'CITATION.tex').exists():
            text = (logs_path / 'CITATION.tex').read_text(encoding='UTF-8')
            text = re.compile(
                r'\\begin{document}(.*?)\\end{document}',
                re.DOTALL | re.IGNORECASE).findall(text)[0].strip()
            text = '<pre>%s</pre>\n' % text
            text += '<h3>Bibliography</h3>\n'
            text += '<pre>%s</pre>\n' % Path(
                pkgrf(self.packagename, 'data/boilerplate.bib')).read_text(encoding='UTF-8')
            boilerplate.append((boiler_idx, 'LaTeX', text))
            boiler_idx += 1

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath=str(self.template_path.parent)),
            trim_blocks=True, lstrip_blocks=True
        )
        report_tpl = env.get_template(self.template_path.name)
        report_render = report_tpl.render(sections=self.sections, errors=self.errors,
                                          boilerplate=boilerplate)

        # Write out report
        (self.out_dir / self.out_filename).write_text(report_render, encoding='UTF-8')
        return len(self.errors)


def run_reports(reportlets_dir, out_dir, subject_label, run_uuid, config):
    """
    Runs the reports

    .. testsetup::

    >>> from shutil import copytree
    >>> from tempfile import TemporaryDirectory
    >>> new_path = Path(__file__).resolve().parent.parent
    >>> test_data_path = new_path / 'data' / 'tests' / 'work'
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)  #noqa
    >>> testdir = Path().resolve()
    >>> data_dir = copytree(test_data_path, testdir / 'work')
    >>> (testdir / 'fmriprep').mkdir(parents=True, exist_ok=True)

    .. doctest::

    >>> run_reports(str(testdir / 'work' / 'reportlets'),
    ...             str(testdir / 'out'), '01', 'madeoutuuid',
    ...             'work/config.json')
    0

    .. testcleanup::

    >>> tmpdir.cleanup()

    """
    report = Report(Path(reportlets_dir), config, out_dir, run_uuid,
                    subject_id=subject_label)
    return report.generate_report()


def generate_reports(subject_list, output_dir, work_dir, run_uuid, config):
    """
    A wrapper to run_reports on a given ``subject_list``
    """
    reports_dir = str(Path(work_dir) / 'reportlets')
    report_errors = [
        run_reports(reports_dir, output_dir, subject_label, run_uuid, config)
        for subject_label in subject_list
    ]

    errno = sum(report_errors)
    if errno:
        import logging
        logger = logging.getLogger('cli')
        error_list = ', '.join('%s (%d)' % (subid, err)
                               for subid, err in zip(subject_list, report_errors) if err)
        logger.error(
            'Preprocessing did not finish successfully. Errors occurred while processing '
            'data from participants: %s. Check the HTML reports for details.', error_list)
    return errno
