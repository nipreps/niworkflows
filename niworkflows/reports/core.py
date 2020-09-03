# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Reports builder for BIDS-Apps.

Generalizes report generation across BIDS-Apps

"""
from pathlib import Path
import re
from itertools import compress
from collections import defaultdict
from pkg_resources import resource_filename as pkgrf
from bids.layout import BIDSLayout, add_config_paths
import jinja2
from nipype.utils.filemanip import copyfile

# Add a new figures spec
try:
    add_config_paths(figures=pkgrf("niworkflows", "data/nipreps.json"))
except ValueError as e:
    if "Configuration 'figures' already exists" != str(e):
        raise

PLURAL_SUFFIX = defaultdict(str("s").format, [("echo", "es")])
SVG_SNIPPET = [
    """\
<object class="svg-reportlet" type="image/svg+xml" data="./{0}">
Problem loading figure {0}. If the link below works, please try \
reloading the report in your browser.</object>
</div>
<div class="elem-filename">
    Get figure file: <a href="./{0}" target="_blank">{0}</a>
</div>
""",
    """\
<img class="svg-reportlet" src="./{0}" style="width: 100%" />
</div>
<div class="elem-filename">
    Get figure file: <a href="./{0}" target="_blank">{0}</a>
</div>
""",
]


class Element(object):
    """Just a basic component of a report"""

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

    >>> cwd = os.getcwd()
    >>> os.chdir(tmpdir)

    >>> from pkg_resources import resource_filename
    >>> from shutil import copytree
    >>> from bids.layout import BIDSLayout
    >>> test_data_path = resource_filename('niworkflows', 'data/tests/work')
    >>> testdir = Path(tmpdir)
    >>> data_dir = copytree(test_data_path, str(testdir / 'work'))
    >>> out_figs = testdir / 'out' / 'fmriprep'
    >>> bl = BIDSLayout(str(testdir / 'work' / 'reportlets'),
    ...                 config='figures', validate=False)

    .. doctest::

    >>> bl.get(subject='01', desc='reconall') # doctest: +ELLIPSIS
    [<BIDSFile filename='.../fmriprep/sub-01/figures/sub-01_desc-reconall_T1w.svg'>]

    >>> len(bl.get(subject='01', space='.*', regex_search=True))
    2

    >>> r = Reportlet(bl, out_figs, config={
    ...     'title': 'Some Title', 'bids': {'datatype': 'figures', 'desc': 'reconall'},
    ...     'description': 'Some description'})
    >>> r.name
    'datatype-figures_desc-reconall'

    >>> r.components[0][0].startswith('<img')
    True

    >>> r = Reportlet(bl, out_figs, config={
    ...     'title': 'Some Title', 'bids': {'datatype': 'figures', 'desc': 'reconall'},
    ...     'description': 'Some description', 'static': False})
    >>> r.name
    'datatype-figures_desc-reconall'

    >>> r.components[0][0].startswith('<object')
    True

    >>> r = Reportlet(bl, out_figs, config={
    ...     'title': 'Some Title', 'bids': {'datatype': 'figures', 'desc': 'summary'},
    ...     'description': 'Some description'})

    >>> r.components[0][0].startswith('<h3')
    True

    >>> r.components[0][1] is None
    True

    >>> r = Reportlet(bl, out_figs, config={
    ...     'title': 'Some Title',
    ...     'bids': {'datatype': 'figures', 'space': '.*', 'regex_search': True},
    ...     'caption': 'Some description {space}'})
    >>> sorted(r.components)[0][1]
    'Some description MNI152NLin2009cAsym'

    >>> sorted(r.components)[1][1]
    'Some description MNI152NLin6Asym'


    >>> r = Reportlet(bl, out_figs, config={
    ...     'title': 'Some Title',
    ...     'bids': {'datatype': 'fmap', 'space': '.*', 'regex_search': True},
    ...     'caption': 'Some description {space}'})
    >>> r.is_empty()
    True

    .. testcleanup::

    >>> os.chdir(cwd)

    """

    def __init__(self, layout, out_dir, config=None):
        if not config:
            raise RuntimeError("Reportlet must have a config object")

        self.name = config.get(
            "name", "_".join("%s-%s" % i for i in sorted(config["bids"].items()))
        )
        self.title = config.get("title")
        self.subtitle = config.get("subtitle")
        self.description = config.get("description")

        # Query the BIDS layout of reportlets
        files = layout.get(**config["bids"])

        self.components = []
        for bidsfile in files:
            src = Path(bidsfile.path)
            ext = "".join(src.suffixes)
            desc_text = config.get("caption")

            contents = None
            if ext == ".html":
                contents = src.read_text().strip()
            elif ext == ".svg":
                entities = dict(bidsfile.entities)
                if desc_text:
                    desc_text = desc_text.format(**entities)

                try:
                    html_anchor = src.relative_to(out_dir)
                except ValueError:
                    html_anchor = src.relative_to(Path(layout.root).parent)
                    dst = out_dir / html_anchor
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    copyfile(src, dst, copy=True, use_hardlink=True)

                contents = SVG_SNIPPET[config.get("static", True)].format(html_anchor)

                # Our current implementations of dynamic reportlets do this themselves,
                # however I'll leave the code here since this is potentially something we
                # will want to transfer from every figure generator to this location.
                # The following code misses setting preserveAspecRatio="xMidYMid meet"
                # if not is_static:
                #     # Remove height and width attributes from initial <svg> tag
                #     svglines = out_file.read_text().splitlines()
                #     expr = re.compile(r' (height|width)=["\'][0-9]+(\.[0-9]*)?[a-z]*["\']')
                #     for l, line in enumerate(svglines[:6]):
                #         if line.strip().startswith('<svg'):
                #             newline = expr.sub('', line)
                #             svglines[l] = newline
                #             out_file.write_text('\n'.join(svglines))
                #             break

            if contents:
                self.components.append((contents, desc_text))

    def is_empty(self):
        return len(self.components) == 0


class SubReport(Element):
    """SubReports are sections within a Report."""

    def __init__(self, name, isnested=False, reportlets=None, title=""):
        self.name = name
        self.title = title
        self.reportlets = reportlets or []
        self.isnested = isnested


class Report:
    """
    The full report object. This object maintains a BIDSLayout to index
    all reportlets.


    .. testsetup::

    >>> cwd = os.getcwd()
    >>> os.chdir(tmpdir)

    >>> from pkg_resources import resource_filename
    >>> from shutil import copytree
    >>> from bids.layout import BIDSLayout
    >>> test_data_path = resource_filename('niworkflows', 'data/tests/work')
    >>> testdir = Path(tmpdir)
    >>> data_dir = copytree(test_data_path, str(testdir / 'work'))
    >>> out_figs = testdir / 'out' / 'fmriprep'

    .. doctest::

    >>> robj = Report(testdir / 'out', 'madeoutuuid', subject_id='01', packagename='fmriprep',
    ...               reportlets_dir=testdir / 'work' / 'reportlets')
    >>> robj.layout.get(subject='01', desc='reconall')  # doctest: +ELLIPSIS
    [<BIDSFile filename='.../figures/sub-01_desc-reconall_T1w.svg'>]

    >>> robj.generate_report()
    0
    >>> len((testdir / 'out' / 'fmriprep' / 'sub-01.html').read_text())
    36693

    .. testcleanup::

    >>> os.chdir(cwd)

    """

    def __init__(
        self,
        out_dir,
        run_uuid,
        config=None,
        out_filename="report.html",
        packagename=None,
        reportlets_dir=None,
        subject_id=None,
    ):
        self.root = Path(reportlets_dir or out_dir)

        # Initialize structuring elements
        self.sections = []
        self.errors = []
        self.out_dir = Path(out_dir)
        self.out_filename = out_filename
        self.run_uuid = run_uuid
        self.packagename = packagename
        self.subject_id = subject_id
        if subject_id is not None:
            self.subject_id = (
                subject_id[4:] if subject_id.startswith("sub-") else subject_id
            )
            self.out_filename = f"sub-{self.subject_id}.html"

        # Default template from niworkflows
        self.template_path = Path(pkgrf("niworkflows", "reports/report.tpl"))
        self._load_config(Path(config or pkgrf("niworkflows", "reports/default.yml")))
        assert self.template_path.exists()

    def _load_config(self, config):
        from yaml import safe_load as load

        settings = load(config.read_text())
        self.packagename = self.packagename or settings.get("package", None)

        if self.packagename is not None:
            self.root = self.root / self.packagename
            self.out_dir = self.out_dir / self.packagename

        if self.subject_id is not None:
            self.root = self.root / "sub-{}".format(self.subject_id)

        if "template_path" in settings:
            self.template_path = config.parent / settings["template_path"]

        self.index(settings["sections"])

    def init_layout(self):
        self.layout = BIDSLayout(self.root, config="figures", validate=False)

    def index(self, config):
        """
        Traverse the reports config definition and instantiate reportlets.

        This method also places figures in their final location.
        """
        # Initialize a BIDS layout
        self.init_layout()
        for subrep_cfg in config:
            # First determine whether we need to split by some ordering
            # (ie. sessions / tasks / runs), which are separated by commas.
            orderings = [
                s for s in subrep_cfg.get("ordering", "").strip().split(",") if s
            ]
            entities, list_combos = self._process_orderings(orderings, self.layout)

            if not list_combos:  # E.g. this is an anatomical reportlet
                reportlets = [
                    Reportlet(self.layout, self.out_dir, config=cfg)
                    for cfg in subrep_cfg["reportlets"]
                ]
            else:
                # Do not use dictionary for queries, as we need to preserve ordering
                # of ordering columns.
                reportlets = []
                for c in list_combos:
                    # do not display entities with the value None.
                    c_filt = list(filter(None, c))
                    ent_filt = list(compress(entities, c))
                    # Set a common title for this particular combination c
                    title = "Reports for: %s." % ", ".join(
                        [
                            '%s <span class="bids-entity">%s</span>'
                            % (ent_filt[i], c_filt[i])
                            for i in range(len(c_filt))
                        ]
                    )
                    for cfg in subrep_cfg["reportlets"]:
                        cfg["bids"].update({entities[i]: c[i] for i in range(len(c))})
                        rlet = Reportlet(self.layout, self.out_dir, config=cfg)
                        if not rlet.is_empty():
                            rlet.title = title
                            title = None
                            reportlets.append(rlet)

            # Filter out empty reportlets
            reportlets = [r for r in reportlets if not r.is_empty()]
            if reportlets:
                sub_report = SubReport(
                    subrep_cfg["name"],
                    isnested=bool(list_combos),
                    reportlets=reportlets,
                    title=subrep_cfg.get("title"),
                )
                self.sections.append(sub_report)

        # Populate errors section
        error_dir = (
            self.out_dir / "sub-{}".format(self.subject_id) / "log" / self.run_uuid
        )
        if error_dir.is_dir():
            from ..utils.misc import read_crashfile

            self.errors = [read_crashfile(str(f)) for f in error_dir.glob("crash*.*")]

    def generate_report(self):
        """Once the Report has been indexed, the final HTML can be generated"""
        logs_path = self.out_dir / "logs"

        boilerplate = []
        boiler_idx = 0

        if (logs_path / "CITATION.html").exists():
            text = (
                re.compile("<body>(.*?)</body>", re.DOTALL | re.IGNORECASE)
                .findall((logs_path / "CITATION.html").read_text())[0]
                .strip()
            )
            boilerplate.append(
                (boiler_idx, "HTML", f'<div class="boiler-html">{text}</div>')
            )
            boiler_idx += 1

        if (logs_path / "CITATION.md").exists():
            text = (logs_path / "CITATION.md").read_text()
            boilerplate.append((boiler_idx, "Markdown", f"<pre>{text}</pre>\n"))
            boiler_idx += 1

        if (logs_path / "CITATION.tex").exists():
            text = (
                re.compile(
                    r"\\begin{document}(.*?)\\end{document}", re.DOTALL | re.IGNORECASE
                )
                .findall((logs_path / "CITATION.tex").read_text())[0]
                .strip()
            )
            boilerplate.append(
                (
                    boiler_idx,
                    "LaTeX",
                    f"""<pre>{text}</pre>
<h3>Bibliography</h3>
<pre>{Path(pkgrf(self.packagename, 'data/boilerplate.bib')).read_text()}</pre>
""",
                )
            )
            boiler_idx += 1

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath=str(self.template_path.parent)),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
        )
        report_tpl = env.get_template(self.template_path.name)
        report_render = report_tpl.render(
            sections=self.sections, errors=self.errors, boilerplate=boilerplate
        )

        # Write out report
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / self.out_filename).write_text(report_render, encoding="UTF-8")
        return len(self.errors)

    @staticmethod
    def _process_orderings(orderings, layout):
        """
        Generate relevant combinations of orderings with observed values.

        Arguments
        ---------
        orderings : :obj:`list` of :obj:`list` of :obj:`str`
            Sections prescribing an ordering to select across sessions, acquisitions, runs, etc.
        layout : :obj:`bids.layout.BIDSLayout`
            The BIDS layout

        Returns
        -------
        entities: :obj:`list` of :obj:`str`
            The relevant orderings that had unique values
        value_combos: :obj:`list` of :obj:`tuple`
            Unique value combinations for the entities

        """
        # get a set of all unique entity combinations
        all_value_combos = {
            tuple(bids_file.get_entities().get(k, None) for k in orderings)
            for bids_file in layout.get()
        }
        # remove the all None member if it exists
        none_member = tuple([None for k in orderings])
        if none_member in all_value_combos:
            all_value_combos.remove(tuple([None for k in orderings]))
        # see what values exist for each entity
        unique_values = [
            {value[idx] for value in all_value_combos} for idx in range(len(orderings))
        ]
        # if all values are None for an entity, we do not want to keep that entity
        keep_idx = [
            False if (len(val_set) == 1 and None in val_set) or not val_set else True
            for val_set in unique_values
        ]
        # the "kept" entities
        entities = list(compress(orderings, keep_idx))
        # the "kept" value combinations
        value_combos = [
            tuple(compress(value_combo, keep_idx)) for value_combo in all_value_combos
        ]
        # sort the value combinations alphabetically from the first entity to the last entity
        value_combos.sort(
            key=lambda entry: tuple(
                str(value) if value is not None else "0" for value in entry
            )
        )

        return entities, value_combos


def run_reports(
    out_dir,
    subject_label,
    run_uuid,
    config=None,
    reportlets_dir=None,
    packagename=None,
):
    """
    Run the reports.

    .. testsetup::

    >>> cwd = os.getcwd()
    >>> os.chdir(tmpdir)

    >>> from pkg_resources import resource_filename
    >>> from shutil import copytree
    >>> test_data_path = resource_filename('niworkflows', 'data/tests/work')
    >>> testdir = Path(tmpdir)
    >>> data_dir = copytree(test_data_path, str(testdir / 'work'))
    >>> (testdir / 'fmriprep').mkdir(parents=True, exist_ok=True)

    .. doctest::

    >>> run_reports(testdir / 'out', '01', 'madeoutuuid', packagename='fmriprep',
    ...             reportlets_dir=testdir / 'work' / 'reportlets')
    0

    .. testcleanup::

    >>> os.chdir(cwd)

    """
    return Report(
        out_dir,
        run_uuid,
        config=config,
        subject_id=subject_label,
        packagename=packagename,
        reportlets_dir=reportlets_dir,
    ).generate_report()


def generate_reports(
    subject_list, output_dir, run_uuid, config=None, work_dir=None, packagename=None
):
    """Execute run_reports on a list of subjects."""
    reportlets_dir = None
    if work_dir is not None:
        reportlets_dir = Path(work_dir) / "reportlets"
    report_errors = [
        run_reports(
            output_dir,
            subject_label,
            run_uuid,
            config=config,
            packagename=packagename,
            reportlets_dir=reportlets_dir,
        )
        for subject_label in subject_list
    ]

    errno = sum(report_errors)
    if errno:
        import logging

        logger = logging.getLogger("cli")
        error_list = ", ".join(
            "%s (%d)" % (subid, err)
            for subid, err in zip(subject_list, report_errors)
            if err
        )
        logger.error(
            "Preprocessing did not finish successfully. Errors occurred while processing "
            "data from participants: %s. Check the HTML reports for details.",
            error_list,
        )
    return errno
