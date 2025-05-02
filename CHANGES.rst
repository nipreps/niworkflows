1.13.1 (May 02, 2025)
=====================
Patch release in the 1.13.x series.

This release adds the ``niworkflows.engine.splicer`` module, to facilitate sugery on wworkflows.

* FIX: Allow passing arguments through tag decorator (#939)
* ENH: Workflow splicer module (#938)
* MNT: Bump astral-sh/setup-uv from 5 to 6 (#937)


1.13.0 (March 20, 2025)
=======================
New feature release in the 1.13.x series.

This release restructures the ANTs brain extraction workflow.
This should be a backwards compatible change, but a minor release is used out of caution.

* FIX: Remove duplicated code, problematic deviation from antsBrainExtraction (#931)
* FIX: Listify numbers (#925)
* ENH: Allow + symbol in label entities (#926, #927)
* STY: Apply various ruff checks (#913, #915, #917, #918, #919, #930)


1.12.2 (January 23, 2025)
=========================
Patch release that includes a new keyword argument to `BIDSDataGrabber` to bypass requiring a T1w image.

* ENH: Parse kwargs to allow no T1w (#922)
* MNT: Bump astral-sh/setup-uv from 4 to 5 (#920)
* STY: Format + add git blame ignore file (#923)
* STY: Apply ruff/Pylint rule PLR0402 (#916)
* STY: Apply ruff/flake8-pie rules (PIE) (#914)
* STY: Apply ruff/flake8-comprehensions preview rule C409 (#912)
* STY: Use f-strings where possible (#836)


1.12.1 (December 12, 2024)
==========================
Patch release that adds a new input to `SpatialNormalization`, but remains backwards compatible.

* ENH: Allow adjusting histogram matching of configuration files (#911)


1.12.0 (December 05, 2024)
==========================
New minor release in the 1.12.x series.

This release includes support for Python 3.13, Numpy 2.0 and Nipype 1.9.

Note that using Python 3.13 requires building traits from the development branch.

* FIX: Pandas deprecations (#906)
* FIX: Numpy 2 compatibility (#905)
* FIX: Check 2D spaces against list from templateflow (#888)
* ENH: Add boldmap suffix to filename patterns. (#899)
* ENH: Add dir entity to T2starmap pattern. (#898)
* TEST: CIFTI2 expected checksum has changed (#895)
* STY: Replace black+flake8 with ruff, add pre-commit (#904)
* STY: use f-string within normalization interface (#892)
  MNT: Enable 3.13 tests (#909)
* MNT: Switch to tox for testing and CI (#900)
* MNT: Drop Python 3.8, update dependencies (#896)
* CI: Fix get_data job on CircleCI (#897)


1.11.0 (July 31, 2024)
======================
New minor release to start the 1.11.x series.

The major changes include:
- a deprecation to T1w-only based template processing
- addition of a new dependency ``acres`` for accessing package data

However, this is still backwards compatible with the 1.10.x series.

* MAINT: Depend on acres for data access
* ENH: Add PrepareDerivative/SaveDerivative interfaces (#885)
* ENH: Make template dimensions support T2w as well (#879)
* ENH: Modify FSSource to output T2 (#868)
* FIX: Set cal_max in the NIfTI header for visualization after ``IntensityClip`` (#878)
* FIX: Remove accidental MRIQC dependency, allow app config to be passed to workflow plugin (#876)
* TEST: Fix plugin invocation, use an initializer that can be verified (#880)


1.10.2 (June 10, 2024)
======================
Bug-fix release in the 1.10.x series

* ENH: Modify FSSource to output T2 (#868)
* STY: Apply ruff/flake8-implicit-str-concat rule ISC001 (#870)
* STY: Remove extraneous quotes (#859)
* STY: Apply assorted ruff/refurb rules (#867)
* MNT: update vendored docs files (#834)
* MNT: Fix deprecation error in CI (#866)

1.10.1 (March 06, 2024)
=======================
Bug-fix release in the 1.10.x series

* ENH: Allow alternative ``extensions`` when creating a BIDS skeleton (#858)
* ENH: Parsing for ``space+cohort`` structure (#857)

1.10.0 (December 13, 2023)
==========================
New feature release in the 1.10.x series.

This release has several improvements targeting ASLPrep and MRIQC, particularly
around BIDS tooling and workflow plugins.
We also drop our use of Convert3D in favor of nitransforms, allowing downstream
projects to remove this non-Python dependency as needed.

Some fixes change workflow structure, so please test before upgrading.

* FIX: Copy header to ensure mask is really aligned (#844)
* FIX: Correctly use ``ignore_initial_volumes`` in ``ConfoundsCorrelationPlot`` (#846)
* FIX: Add DWI to the default queries of BIDS querying (#833)
* ENH: Add ``ignore_initial_volumes`` param to ``ConfoundsCorrelationPlot`` (#843)
* ENH: Add MRIQC's custom multiproc runner that allows passing the pool (#841)
* ENH: Upstream debugging hook from MRIQC (#842)
* ENH: Add ASL support in BIDS utilities (#839)
* ENH: Add hemi entity to transform path patterns (#832)
* RF: Replace Convert3d with nitransforms in MCFLIRT2ITK (#835)
* DOC: fix typo in refmap.py (#837)
* STY: Assorted refurb suggestions (#831)
* MNT: http:// → https:// (#828)
* CI: Bump actions/setup-python from 4 to 5 (#845)
* CI: Bump conda-incubator/setup-miniconda from 2 to 3 (#840)

1.9.0 (October 12, 2023)
========================
New feature release in the 1.9.x series.

This release should be mostly compatible with 1.8.x.
The most significant addition is the :class:`niworkflows.data.Loader`,
which provides some wrapping around :mod:`importlib.resources` to make
accessing resources more convenient.

* FIX: Build config and CI (#826)
* FIX: Compatibility with matplotlib 3.8.0 (#825)
* FIX: Save boldrefs as float32 (#821)
* FIX: Replace deprecated Matplotlib functionality (#820)
* FIX: Test names of paths when calling path.iterdir() (#818)
* ENH: Add PET support to niworkflows (#823)
* ENH: Add data loader class, install at root and data modules (#816)
* MNT: Pyupgrade suggestions (#817)
* MNT: update vendored docs files (#813)
* CI: Bump actions/checkout from 3 to 4 (#824)

1.8.1 (June 29, 2023)
=====================
Patch release in the 1.8.x series.

This release includes a fix to CIFTI volume structure indices, and adds additional Workbench interfaces.

* ENH: Port GIFTI/CIFTI interfaces from fMRIPrep (#811)
* FIX: Generate CIFTI volume structure indices in column-major order (#815)

1.8.0 (June 01, 2023)
=====================
New feature release in the 1.8.x series.

This includes only bug-fixes, however the reselection of "white" surfaces
as the default gray/white boundary instead of "smoothwm" justifies a minor
release.

* FIX: Add desc entity to all path patterns (#808)
* FIX: Inspect if template has resolution instead of assuming (#803)
* FIX: Add missing ``dir-{direction}`` pattern for data sinking DWI reportlets (#807)
* FIX: Move from smoothwm to white (#806)
* FIX: Add option to bypass twopass in AFNI RobustAverage (#801)
* FIX: Use len(segments) to avoid nsegments becoming wrong (#795)
* MNT: Add PendingDeprecationWarning to nireports modules (#810)
* MNT: Configure dependabot to upgrade actions (#796)
* MNT: Drop Python 3.7 support, test 3.11 (#793)
* MNT: Update build system and CI invocations (#792)
* CI: Do not run scheduled builds on forks (#800)
* CI: Bump codespell-project/actions-codespell from 1 to 2 (#805)
* CI: Bump actions/checkout from 2 to 3 (#797)
* CI: Build and upload wheels (#791)

1.7.9 (April 11, 2023)
======================
Patch release in the 1.7.x series.

This release contains a minor fix for degenerate cases that arise on very
low resolution datasets, such as used for CI testing.

* FIX: Use len(segments) to avoid nsegments becoming wrong (#795)

1.7.8 (March 24, 2023)
======================
Patch release in the 1.7.x series.

This release combines the fixes in 1.7.6 and 1.7.7, which were accidentally separately
branched from 1.7.5.

1.7.7 (March 24, 2023)
======================
Patch release that optimizes BIDS metadata handling by reusing cached index databases.

* ENH: Allow passing a ``database_path`` to create a ``BIDSLayout`` (#788)

1.7.6 (March 06, 2023)
======================
Patch release in the 1.7.x series.

* FIX: Use correct image for contour plotting (#785)

1.7.5 (March 06, 2023)
======================
Patch release in the 1.7.x series.

* FIX: Use a more reliable test of NIfTIness (#784)

1.7.4 (February 28, 2023)
=========================
Patch release in the 1.7.x series.

This release again fixes a minor bug in carpet plot generation and supports
the upcoming Pandas release.

* FIX: Check for empty segments in plot_carpet (#783)
* FIX: Use keyword sep for pandas.read_csv (#781)
* STY: Configure codespell and fix typos (#782)

1.7.3 (February 16, 2023)
=========================
Patch release in the 1.7.x series.

This release fixes a minor bug in carpet plot generation and resolves remaining issues
with using custom configurations in DerivativesDataSink.

* ENH: Pass custom configuration into DerivativesDataSink entity parser (#777)
* FIX: Avoid sorting carpetplot rows when decimated ROIs have one voxel (#780)

1.7.2 (February 15, 2023)
=========================
Patch release in the 1.7.x series.

This release includes minor changes involving confound plotting.

* FIX: Restore confound order in correlation plot (#779)
* FIX: Handle plotting only one MELODIC component (#775)

1.7.1 (January 23, 2023)
========================
Patch release in the 1.7.x series.

This release includes updates to the Docker image environment, and a fix for a plotting bug.

  * CI: Update GitHub actions frdom master (#766)
  * CI: Drop set-output (#763)
  * FIX: Remove unused ANTs parameter that was removed in 2.4.1 (#771)
  * FIX: Coerce ``MGHImage`` to ``Nifti1Image`` when plotting (#769)
  * FIX: Create test label images with realistic dtypes (#767)
  * FIX: error for ``np.bool`` with ``numpy 1.24.0`` (#764)
  * MNT: Try bumping docker base to latest nipreps/miniconda (#773)
  * MNT: Use org-level context (#770)
  * REF: Store global variables as DerivativesDataSink attributes (#762)

1.7.0 (November 29, 2022)
=========================
New feature release in the 1.7.x series. This series will support sMRIPrep 0.10,
fMRIPrep 22.1, and NiBabies 22.2.

This will be the last minor release series to support Python 3.7.

* FIX: Use ``np.bool_`` instead of deprecated np.bool (#760)
* FIX: Use nilearn plot_surf over surfplot (#745)
* ENH: Allow DerivativesDataSink to handle output==input (#758)
* ENH: Add ``Resolution`` field to metadata if data has ``res`` entity (#754)
* ENH: Add morphometric suffixes to pattern matching (#753)
* RF: GenerateCifti interface (#756)
* MNT: Loosen dependency versions (#759)
* MNT: Remove non-compliant ``dtseries.json`` metadata (#750)
* MNT: Build FreeSurfer 7.2 (#751)

1.6.4 (September 13, 2022)
==========================
Patch release in the 1.6.x series.

This release includes a fix that resolves an issue with empty TSV files.

* FIX: TSV2JSON should convert empty TSV files to empty JSON files (#746)

1.6.3 (September 08, 2022)
==========================
Patch release in the 1.6.x series.

This release includes a new interface and improvements to existing interfaces.

  * CI: Fix commit check (#744)
  * CI: Extract pip pre checks into separate, skippable job (#743)
  * FIX: Missing outputs in RobustAverage (#741)
  * ENH: ReorientImage interface (#736)
  * ENH: Add minimum_fs_version input to BIDSFreeSurferDir (#739)
  * MNT: Pin traits < 6.4 (#738)
  * MNT: Upgrade versioneer (#737)
  * MAINT: Remove pins that broke prerelease tests (#735)

1.6.2 (July 1, 2022)
====================
Bug-fix release in the 1.6.x series.

This release fixes a bug where BIDS data collection would fail if using ``session``
as a BIDS filter.

* FIX: Ensure we do not clobber entities from bids filters (#731)

1.6.1 (May 27, 2022)
====================
Bug-fix release in the 1.6.x series.

This release allows downstream tools to use nitransforms 22.0.

* MNT: Relax nitransforms dependency #727

1.6.0 (May 25, 2022)
====================
New feature release in the 1.6.x series. This series will support sMRIPrep 0.9,
fMRIPrep 22.0., and nibabies 22.1.

* FIX: Address some reliability issues of the functional masking workflow (#714)
* FIX: Improve reliability of BOLD masking workflow (#712)
* FIX: Account for potential lists of lists in multi-echo cases (#719)
* ENH: Added MRtrix3 gradients to derivative path patterns (#724)
* ENH: Add T2starmap as a functional derivative (#720)
* MAINT: Replace distutils use, upgrade versioneer (#725)
* CI: Let datalad handle git-annex installation (#721)
* CI: Bump environment cache version (#717)
* CI: Fallback to maintenance branch first (#716)

1.5.4 (April 08, 2022)
======================
Bug-fix release in the 1.5.x series.

This change enables a bug-fix in sdcflows where images with slightly
different affines were unable to be concatenated for use with TOPUP.

* ENH: Add ``affine_tolerance`` flag to ``MergeSeries`` (#706)

1.5.3 (April 05, 2022)
======================

A bugfix release incorporating changes from the 1.3.x and 1.4.x
maintenance series.

* FIX: Alternate query for template brain mask (#704)
* FIX: Use copy function that does not preserve mtime when creating fsaverage directories (#703)
* FIX: Test failures (#701)
* ENH: Add optional session filter when collecting data (#678)
* ENH: Specify path pattern for transformation files in dwi datatype (#699)
* REF: Move BIDS skeleton generation into new ``testing`` module (#682)
* MNT: Select magnitude images in collect_data for BIDS 1.5.0 (#594)
* CI: Install package across Python versions and run pytest (#697)

1.5.2 (March 23, 2022)
======================
A bugfix release ensuring compatibility with Python 3.7.

* FIX: Add ``unlink()`` with ``missing_ok`` for Python 3.7 (#695)

1.5.1 (March 11, 2022)
======================
A bugfix release addressing several issues and improving the carpet plot.

* FIX: Set ``pixdim[4]`` to match ``RepetitionTime`` (#679)
* FIX: Unlink destination files before saving derivative (#683)
* FIX: Smarter decision to remap input segmentation to carpetplot (#691)
* FIX: Error handling the segmentation in NIfTI extraction (#689)
* ENH: Add a legend to carpet plots with more than one segment (#690)
* ENH: Miscellaneous improvements to carpetplot (#688)
* ENH: Refactor carpetplot reasigning responsibilities (#687)
* ENH: Initiate a module of nipype interfaces for mathematical morphology (#651)
* ENH: Reorder within-segment rows of carpetplot with hierarchical clustering (#686)
* DOC: Fix doc builds in ``maint/1.3.x`` (#692)

1.5.0 (February 07, 2022)
=========================
First release in the 1.5.x series.
This release upgrades the Dockerfile to use FSL6, and includes some new interfaces.

  * ENH: Additional nibabel interfaces (#680)
  * ENH: Add utility function to generate sample BIDS directory (#677)
  * ENH: Update to FSL6 (#675)
  * FIX: Scipy docs path (#681)
  * TEST: Drop excessively long interface equivalence tests (#674)

1.4.10 (September 14, 2022)
===========================
Bug-fix release in the 1.4.x series

Backports and minor maintenance.

* FIX: TSV2JSON should convert empty TSV files to empty JSON files (#747)
* FIX: Stop relying on deprecated indexing behavior (#730)
* MNT: Update docs and GitHub actions config from master in LTS branch (#748)

1.4.9 (April 21, 2022)
======================
Bug-fix release in the 1.4.x series.

This release includes improvements to the reliability of the BOLD masking workflow,
as well as an fMRIPrep bug-fix when reusing multi-echo anatomicals.

  * FIX: Account for potential lists of lists in multi-echo cases (#719)
  * FIX: Improve reliability of BOLD masking workflow (#712)

1.4.8 (April 08, 2022)
======================
Bug-fix release in the 1.4.x series.

This change enables a bug-fix in sdcflows where images with slightly
different affines were unable to be concatenated for use with TOPUP.

* ENH: Add ``affine_tolerance`` flag to ``MergeSeries`` (#706)

1.4.7 (March 31, 2022)
======================
Bug-fix release in the 1.4.x series.

* FIX: Use copy function that does not preserve mtime when creating fsaverage directories (#703)
* FIX: Add unlink() with missing_ok for Python 3.7 (#695)
* ENH: Select magnitude images in collect_data for BIDS 1.5.0 (#594)
* CI: Install package across Python versions and run pytest  (#697)

1.4.6 (March 09, 2022)
======================
Patch release in the 1.4.x series.

* FIX: Unlink destination files before saving derivative (#683)
* FIX: Set pixdim[4] to match RepetitionTime (#679)

1.4.5 (December 13, 2021)
=========================
Patch release in the 1.4.x series.

In preparation for fMRIPrep 21.0.0 release.

* FIX: Conform entity ordering to BIDS specification for derivatives (#676)
* ENH: Add CIFTI surface plot (#663)

1.4.4 (December 08, 2021)
=========================
Patch release in the 1.4.x series.

This release enables better multi-echo handling in fMRIPrep.

* ENH: Return all bold files from init_bold_reference_wf (#673)

1.4.3 (November 16, 2021)
=========================
Patch release in the 1.4.x series, including improvements to the carpetplot figure.
With thanks to Zaki A. for the contributions.

* ENH: Carpet plot tweaks (#617)
* FIX: Patch ``ants.Registration`` interface temporarily (#654)
* MAINT: Revise Docker image building and dependencies (#655)
* MAINT: Relax matplotlib constraint (#668)
* MAINT: Configure black to avoid excessive rewriting (#666)

1.4.2 (October 15, 2021)
========================
Bug-fix release in the 1.4.x series revising dependencies and including minor improvements to the Docker image, a complete overhaul of the documentation skin, building and deployment & archival.

* DOC: Overhaul - new skin, new deployment & multiversion archival (#656, #657, #658)
* MAINT: Revise Docker image building and dependencies (#655)

1.4.1 (October 07, 2021)
========================
Bug-fix release in the 1.4.x series.

This depends on the latest nitransforms, enabling downstream tools to upgrade.

* FIX: Patch ``ants.Registration`` interface temporarily (#654)

1.4.0 (September 1, 2021)
=========================
First release in the 1.4.x series.
This release includes enhancements and bug-fixes towards the release of the first
beta version of *dMRIPrep*.
It also contains new features that are necessary for the API overhaul that has
happened within the new *SDCFlows 2.x* series.
The new series include a cross-cutting (modalities, species) workflow to generate
EPI references.
Finally other *NiPreps* will also have a first release with specific support for
them: *NiRodents* and *NiBabies* (and their corresponding *fMRIPrep* extensions).

.. admonition:: *NiWorkflows* has been relicensed!

    As of the first release candidate of the 1.4 series, the license has changed
    from BSD 3-clause to the Apache License 2.0.
    Amongst several terms that are changing, the following two premises are relevant
    if you derive code from the new series:

    * *You must give any other recipients of the Work or Derivative Works a copy
      of this License*; and
    * *You must cause any modified files to carry prominent notices stating that
      You changed the files*.

A list of prominent changes can be found below. (With thanks to Lea Waller for the contributions)

* DOC: Ensure copyright notice in all headers' comment (#635)
* FIX: Set slope and intercept to 1/0 if not otherwise provided (#649)
* FIX: ``DerivativesDataSink`` warning when it has multiple source files (#647)
* FIX: ``FSDetectInputs`` mutually exclusive options for ``ReconAll`` (#646)
* FIX: Remove pandas warning about use of keyword arguments (#645)
* FIX: Improve ``SimpleBeforeAfterRPT`` contour visibility (#643)
* FIX: ``DerivativesDataSink`` dismissing entity writing reportlet (#638)
* FIX: Ensure ``IntensityClip`` input is a 3D file (#621)
* FIX: Limit memory usage in ``EstimateReferenceImage`` (#629)
* FIX: Purge removed ``--disable`` flag from *svgo* call (#626)
* FIX: Re-add ``sbref_file`` input to reference volume interface (#624)
* FIX: Update svgutils after breaking API changes (#620)
* FIX: Address issues with ``RobustAverage`` global signal measurement (#607)
* FIX: ``NonsteadyStatesDetector`` wrongly using Nipype's ``is_outlier`` (#605)
* FIX: Change in *svgutils*' API on 0.3.2 breaks reportlets (#599)
* FIX: Check for ``in_segm`` input, not ``seg_file`` (typo) (#592)
* FIX: Use the mask to calculate FOV rather than the fixed image in ``GenerateSamplingReference`` (#583)
* FIX: Allow omission of ``<res>`` for template normalization (#582)
* FIX: Include ``_T2starw`` ``_MTw`` and ``_TSE``-suffixes in ``build_path`` options (#584)
* FIX: ``DerivativesDataSink`` warning when it has multiple source files (#573)
* ENH: Allow ``SimpleBeforeAfterRPT`` plotting of image rotated to cardinal axes (#650)
* ENH: Update visual report specification with fieldmaps (#634)
* ENH: Expose the output of the ``ValidateImage`` node as an output in EPI reference workflow (#636)
* ENH: Add an inversion operation to ``IntensityClip`` (#616)
* ENH: Cross-cutting (modalities, species) workflow to generate EPI references (#610)
* ENH: Add a ``RobustAverage`` interface and split volume selection (#602)
* ENH: Revise some patterns of the ``BIDSLayout`` config to aid *SDCFlows* new API (#585)
* ENH: Upstream *fMRIPrep*'s ``init_bbreg_wf`` to integrate it in *dMRIPrep* (#586)
* MAINT: Pin *svgutils* to previous versions (#596)
* MAINT: Drop gh-pages history, reducing repository size (#622)
* MAINT: Add DS030 dataset, with clipped (55 timepoints) BOLD data (#609)
* MAINT: Migrate ``ds003_downsampled`` to ``nipreps-data`` (#608)
* MAINT: Move mask-regressions test-data to datalad + nipreps-data (#606)
* MAINT: Refactor structure of interfaces (#603)
* MAINT: Drop Python 3.6, test setuptools builds, pip installations, and revise Docker pinned versions (#593)
* MAINT: CircleCI housekeeping (#580)
* RF: Write derivatives once, using deterministic gzip settings (#641)

.. caution::

    The ``niworkflows.interfaces`` submodule has been refactored.
    To migrate from previous series, please modify the following imports:

    * ``images.MatchHeader`` -> ``header.MatchHeader``
    * ``images.ValidateImage`` -> ``header.ValidateImage``
    * ``images.Demean`` -> ``nibabel.Demean``
    * ``images.FilledImageLike`` -> ``nibabel.FilledImageLike``
    * ``images.RegridToZooms`` -> ``nibabel.RegridToZooms``
    * ``masks.ROIsPlot`` -> ``reportlets.masks.ROIsPlot``
    * ``masks.ComputeEPIMask`` -> ``nilearn.ComputeEPIMask``
    * ``mni.RobustMNINormalization`` -> ``norm.SpatialNormalization``
    * New ``niworkflows.interfaces.reportlets`` submodule

      * ``report_base`` -> ``reportlets.base``
      * ``masks`` -> ``reportlets.masks``
      * ``registration`` -> ``reportlets.registration``
      * ``segmentation`` -> ``reportlets.segmentation``

    * ``utils.GenerateSamplingReference`` -> ``nibabel.GenerateSamplingReference``
    * ``utils.CopyXForm`` -> ``header.CopyXForm``
    * ``utils.NormalizeMotionParams`` -> ``confounds.NormalizeMotionParams``
    * ``utils.AddTPMs`` -> ``probmaps.AddTPMs``
    * ``utils.TPM2ROI`` -> ``probmaps.TPM2ROI``
    * ``utils.AddTSVHeader`` -> ``utility.AddTSVHeader``
    * ``utils.JoinTSVColumns`` -> ``utility.JoinTSVColumns``
    * ``utils.DictMerge`` -> ``utility.DictMerge``
    * ``utils.TSV2JSON`` -> ``utility.TSV2JSON``

    Beware that interface aliases at the top ``niworkflows.interfaces`` level have
    been removed:

    * ``ExpandModel``, ``SpikeRegressors`` from ``confounds``
    * ``BET`` -> ``reportlets.masks.BETRPT``
    * ``FAST`` -> ``reportlets.segmentation.FASTRPT``
    * ``FLIRT``, ``ApplyXFM``, ``RobustMNINormalization``, ``Registration``,
      ``ApplyTransforms``, ``SimpleBeforeAfter`` now under ``reportlets.registration``
      as ``FLIRTRPT``, ``ApplyXFMRPT``, ``RobustMNINormalizationRPT``, ``ANTSRegistrationRPT``,
      ``ANTSApplyTransformsRPT``, ``SimpleBeforeAfterRPT``.
    * ``CopyXForm``, ``CopyHeader``, ``SanitizeImage`` now under ``header``
    * ``NormalizeMotionParams`` now under ``confounds``.
    * ``FMRISummary``, ``CompCorVariancePlot``, ``ConfoundsCorrelationPlot`` from ``plotting``

1.3.10 (September 16, 2023)
===========================
Bug-fix release in the 1.3.x series.

Minor maintenance.

* FIX: Compatibility with matplotlib 3.8.0 (#825)
* FIX: Remove unused ANTs parameter that was removed in 2.4.1 (#772)
* MNT: Create a requirements.txt to keep the LTS branch buildable (#798)

1.3.9 (December 21, 2022)
=========================
Bug-fix release in the 1.3.x series.

Minor maintenance.

* FIX: Remove deprecated uses of ``np.bool`` for numpy 1.24 compatibility (#764)
* CI: Update GitHub actions from master (#766)
* CI: Update CircleCI from master (#765)

1.3.8 (September 14, 2022)
==========================
Bug-fix release in the 1.3.x series

Backports and minor maintenance.

* FIX: TSV2JSON should convert empty TSV files to empty JSON files (#747)
* FIX: Stop relying on deprecated indexing behavior (#730)
* MNT: Update docs and GitHub actions config from master in LTS branch (#748)

1.3.7 (March 31, 2022)
======================
Bug-fix release in the 1.3.x series

* FIX: Use copy function that does not preserve mtime when creating fsaverage directories (#703)

1.3.6 (March 09, 2022)
======================
Bug-fix release in the 1.3.x series

* FIX: Set pixdim[4] to match RepetitionTime (#679)
* DOC: Fix doc builds for 1.3.x series (#692)

1.3.5 (October 01, 2021)
========================
Bug-fix release in the 1.3.x series

* FIX: Set slope and intercept to 1/0 if not otherwise provided (#649)
* FIX: DerivativesDataSink warning when it has multiple source files [backport #573] (#647)
* FIX: `FSDetectInputs` mutually exclusive options for `ReconAll` (#646)
* MNT: Update some version pinning and correct for *pandas* warning about keyword arguments (#645)
* CI: Use datalad-managed test data [BACKPORT] (#653)

1.3.4 (June 8, 2021)
====================
Bug-fix release in the 1.3.x series.

* RF: Write derivatives once, using deterministic gzip settings

1.3.3 (April 15, 2021)
======================
Bug-fix release in the 1.3.x series.

* FIX: Limit memory usage in ``EstimateReferenceImage`` (#629)
* FIX: Check for ``in_segm`` input, not ``seg_file`` (#592)
* FIX: Use the mask to calculate FOV rather than the fixed image in ``GenerateSamplingReference`` (#583)
* FIX: Allow omission of ``<res>`` for template normalization (#582)
* MAINT: Pin *svgutils* to 0.3.1 (#596)
* MAINT: Migrate from Travis -> GH Actions (#589)
* MAINT: CircleCI housekeeping (#580)

1.3.2 (November 5, 2020)
========================
Bug-fix release in the 1.3.x series.

* FIX: Cordon off ``.dtseries.json`` contents (#578)
* ENH: Add units to qform overwrite report (#577)

1.3.1 (September 22, 2020)
==========================
Bug-fix release in the 1.3.x series.
Addresses longstanding issues in the anatomical MRI brain extraction workflow.

* FIX: Revision of ``antsBrainExtraction``, better handling edge cases (#567)

1.3.0 (September 11, 2020)
==========================
First release in the 1.3.x series.
This release includes enhancements and bug-fixes towards the release of the first
LTS (*long-term support*) version of *fMRIPrep*.
*PyBIDS* has been revised to use more recent versions, a series of ANTs' interfaces
have been deemed ready to upstream into *Nipype*, and several improvements regarding
multi-echo EPI are included.
With thanks to Basile Pinsard for contributions.

* FIX: Patch ``ApplyTransforms`` spec to permit identity in a chain (#554)
* FIX: Add dots to extensions in PyBIDS' config file (#548)
* FIX: Segmentation plots aligned with cardinal axes (#544)
* FIX: Skip T1w file existence check if ``anat_derivatives`` are provided (#545)
* FIX: Avoid diverting CIFTI dtype from original BOLD (#532)
* ENH: Add ``smooth`` input to ``RegridToZooms`` (#549)
* ENH: Enable ``DerivativesDataSink`` to take multiple source files to derive entities (#547)
* ENH: Allow ``bold_reference_wf`` to accept multiple EPIs/SBRefs (#408)
* ENH: Numerical stability of EPI brain-masks using probabilistic prior (#485)
* ENH: Add a pure-Python interface to resample to specific resolutions (#511)
* MAINT: Upstream all bug-fixes in the 1.2.9 release
* MAINT: Finalize upstreaming of ANTs' interfaces to Nipype (#550)
* MAINT: Update to Python +3.6 (#541)

1.2.9 (September 11, 2020)
==========================
Bug-fix release in the 1.2.x series with very minor problems addressed.

* FIX: Reportlets would crash in edge condition (#566)
* FIX: AROMA metadata ``CsfFraction`` -> ``CSFFraction`` (#563)
* FIX: Add DWI nonstandard spaces (#565)

1.2.8 (September 03, 2020)
==========================
Bug-fix release in the 1.2.x series with a minor improvement of the correlations plot.

* FIX: Improved control over correlations plot (#561)

1.2.7 (August 12, 2020)
=======================
Bug-fix release in the 1.2.x series with a very minor improvement of the reportlets.

* FIX: Pin PyBIDS < 0.11 (and TemplateFlow < 0.6.3) only on the 1.2.x series. (#552)
* FIX: Use ``numpy.linspace`` to calculate mosaic plots' cutting planes (#543)

1.2.6 (June 09, 2020)
=====================
Bug-fix release in the 1.2.x series addressing minor bugs encountered mostly
within *sMRIPrep*.
With thanks to Franziskus Liem for contributions.

* FIX: Error conforming T1w images with differing zooms before ``recon-all`` (#534)
* FIX: Restore and deprecate license argument to ``check_valid_fs_license`` (#538)
* FIX: Allow anatomical derivatives to have ``run-`` entity (#539)

1.2.5 (June 4, 2020)
====================
Bug-fix release that remedies an issue with packaging data

* FIX: Packaging data (#535)

1.2.4 (June 04, 2020)
=====================
Bug-fix release improving the FS license checking

* ENH: Improve FS license checking (#533)

1.2.3 (May 27, 2020)
====================
Bug-fix release addressing some downstream issues in *fMRIPrep*.

* FIX: ``MultiLabel`` interpolations should not use ``float=True`` (#530)
* FIX: Do not break figure-datatype derivatives by sessions (#529)
* MNT: Update comments, minimum versions for setup requirements (#512)

1.2.2 (May 26, 2020)
====================
A bug-fix release remedying a casting issue in DerivativesDataSink.

* FIX: Non-integer data coercion initialization

1.2.1 (May 26, 2020)
====================
A bug-fix release in the 1.2.x series. This ensures consistency of datatype (dataobj, header)
when casting to a new type in DerivativesDataSink.

* FIX: Ensure consistency when changing derivative datatype (#527)

1.2.0 (May 21, 2020)
====================
First release in the 1.2.x series. This release includes a variety of enhancements
and bug fixes, including a large scale refactoring of DerivativesDataSink.

* FIX: Purge greedy lstrip from reports (#521)
* FIX: Add DWI default patterns for dMRIPrep's reportlets (#504)
* FIX: Merge/SplitSeries write to path of input image, instead of cwd (#503)
* FIX: Better generalization and renaming+relocation in the API of ``extract_wm`` (#500)
* FIX: Increase fault tolerance of DerivativesDataSink (#497)
* FIX: Match N4-only workflow outputs to brain extraction workflow (#496)
* FIX: Set default volumetric resolution within OutputReferencesAction to native (#494)
* ENH: Upstream NiTransforms module from fMRIPrep (#525)
* ENH: Improve DerivativesDataSink flexibility (#507) (#514) (#516)
* ENH: Add utility function to quickly check for FS license (#505)
* ENH: Add nibabel-based split and merge interfaces (#489)
* ENH: Show registration reportlets inline within Jupyter notebooks (#493)
* ENH: Ensure subcortical volume in CIFTI is in LAS orientation (#484)
* ENH: Produce carpetplot from CIFTI file (#491)
* ENH: Option to set DerivativesDataSink datatype (#492) (#495)
* MAINT: Revert #496 -- N4-only workflow connections (#498)
* MAINT: Transfer brainmask script from fMRIPrep (#488)

1.1.x series
============
1.1.12 (March 19, 2020)
-----------------------
Bug-fix release in the 1.1.x series.

  * FIX: Update naming patterns in figures.json (#483)
  * FIX: Add CE agent to output figure filename templates (#482)

1.1.11 (March 17, 2020)
-----------------------
Bug-fix release to improve CIFTI compatibility with workbench tools.

  * FIX: Ensure BOLD and label orientations are equal (#477)

1.1.10 (March 11, 2020)
-----------------------
Bug-fix release in the 1.1.x series.

  * ENH: Overwrite attr's string conversion dunders (#475)

1.1.9 (March 05, 2020)
----------------------
Bug-fix release in the 1.1.x series.

This release contains maintenance actions on the CI infrastructure after
migration to the `NiPreps organization <https://www.nipreps.org>`__.

  * FIX: replace mutable ``list`` with ``tuple`` in ANTs' workflow (#473)
  * MAINT: Pacify security patterns found by Codacy (#474)
  * MAINT: Miscellaneous housekeeping (#472)
  * MAINT: Fix test_masks (#470)
  * MAINT: Use docker-registry for caching on CircleCI (#471)
  * MAINT: Revise code coverage collection (#469)
  * MAINT: Transfer to nipreps organization (#468)

1.1.8 (February 26, 2020)
-------------------------
Bug-fix release in the 1.1.x series.

This release includes some minor improvements to formatting of reports and derivative metadata.

* FIX: Check for valid qform before calculating change (#466) @effigies
* ENH: Display errors as summary/details elements (#464) @effigies
* ENH: Add a pure-Python ApplyMask interface, based on NiBabel (#463) @oesteban
* MAINT: Replace ``os`` operations with ``pathlib``, indent JSON sidecars (#467) @mgxd

1.1.7 (February 14, 2020)
-------------------------
Minor improvements to enable fMRIPrep 20.0.0 release.

* ENH: Revise SpatialReference caching for ease of use, accessibility (#461) @mgxd
* ENH: Downgrade log level for superfluous scans (#460) @mgxd
* ENH: Enable optional BIDS entity filtering to data collection utility (#407) @bpinsard

1.1.6 (February 7, 2020)
------------------------
Update NiBabel pinned version.

* MAINT: Update nibabel's pin to >=3.0.1

1.1.5 (February 6, 2020)
------------------------
A refactor of recently introduced spaces/references/spatial-references objects,
and some methods for upstream pipelines.

* ENH: Revision of spaces module for consistency (#457)
* ENH: Add BIDS output version checker (#456)
* ENH: Standard space querying (#455)
* ENH: Add cache to ``SpatialReferences`` (#453)
* ENH: Add helper function for cleaning a directory (#454)
* FIX: Parsing of floats in ``ResampleImageBySpacing`` (#452)

1.1.4 (January 28, 2020)
------------------------
Minor enhancements to better represent spaces/spatial-references,
and increasing the test coverage of the Reports System (with thanks to J. Kent
for the contribution):

* ENH: Separate ``Space`` and ``SpatialReferences`` (#451)
* ENH+TST: Add all valid entities to the default report specification (#447)

1.1.3 (January 16, 2020)
------------------------
A fix/enhancement of the report generation system. With thanks to J. Kent for
the contribution.

* ENH/FIX: parse orderings to account for missing entities (#443)

1.1.2 (December 17, 2019)
-------------------------
Hotfix of 1.1.1

* FIX: ``IntraModalMerge`` - Undefined input name used in ``_run_interface`` (#442)

1.1.1 (December 17, 2019)
-------------------------
A bugfix release to support addressing `nipreps/sdcflows#77
<https://github.com/nipreps/sdcflows/issues/77>`__.
With thanks to Alejandro De La Vega for contributions.

* FIX: ``IntraModalMerge`` failed for dims (x, y, z, 1) (#441) @oesteban
* ENH: Add ``n4_only`` workflow -- to skip brain extraction (#435) @adelavega
* MAINT: Update nibabel to 3.0.0rc1 (#439) @mgxd

1.1.0 (December 9, 2019)
------------------------
The first minor release of the niworkflows 1.x series. Numerous interfaces (GenerateCifti, CiftiNameSource, GiftiNameSource) have been changed to be compatible with HCP grayordinates.

* ENH: CIFTI / fsLR density (#436) @mgxd
* ENH: Expand GenerateCifti & MedialNaNs interfaces to support HCP grayordinates / fsLR surfaces (#417) @mgxd

1.0.x series
============
1.0.3 (December 18, 2019)
-------------------------
Hot-fix release in the 1.0.x series. Backported from 1.1.2.

* FIX: ``IntraModalMerge`` - Undefined input name used in ``_run_interface`` (#442)
* FIX: ``IntraModalMerge`` failed for dims (x, y, z, 1) (#441) @oesteban

1.0.2 (December 9, 2019)
------------------------
Bug-fix release in the 1.0.x series.

* FIX: Permit dummy scans to be 0 (#438) @jdkent
* MNT: Specify junit_family to suppress pytest DeprecationWarning (#432) @effigies

1.0.1 (November 27, 2019)
-------------------------
Bug-fix release in the 1.0.x series.

* FIX: Ensure data type of masked image matches T1.mgz (#430) @effigies

1.0.0 (November 26, 2019)
-------------------------
The first stable release of NIWorkflows.

* CI: Drop setup.py install/develop installs (#428) @effigies
* DOC: Maintenance of the documentation building (#429) @oesteban
* DOC: Generate versioned documentation of the API (#416) @rwblair
* ENH: Add ``copy_header`` inputs to some ANTs interfaces (#401) @oesteban
* ENH: Remove the data getters/utils modules (#427) @oesteban
* ENH: Move nilearn interfaces over from fMRIPrep (#414) @oesteban
* ENH: Reports use the default template from niworkflows, allowing overwrite (#419) @oesteban
* FIX: Update all ``SpatialImage.get_data`` -> ``get_fdata`` (#426) @oesteban
* MAINT: Update ``.gitignore`` and ``.dockerignore`` (#420) @oesteban
* MAINT: use scikit-image LTS for earlier python versions (#418) @mgxd
* MAINT: Pin nipype>=1.3.1, remove link dependencies from ``setup.cfg`` @oesteban

0.10.x series
=============
0.10.4 (October 8, 2019)
------------------------
Patch release with a few small bug fixes and improvements.

* FIX: Remove unused, undocumented output from the bold_reference workflow (#409) @oesteban
* FIX: Do not validate built paths (#405) @effigies
* FIX: Ensure that length of indices matches length of values (#397) @rciric
* ENH: Add a new ``Binarize`` interface using nibabel (#402) @oesteban
* ENH: Enable BIDSFreeSurferDir to take an absolute path as a subjects directory (#398) @effigies
* TEST: Separate LTA length fixing and add doctest (#403) @davhunt

0.10.3 (September 9, 2019)
--------------------------
Patch release with several bugfixes and two improvements on how NIfTI files were
handled. With thanks to David Hunt and Ursula Tooley for contributions.

* ENH: Memory optimized header rewriting (#386) @effigies
* ENH: Warn about copying sform to qform only if qform changes (#365) @utooley
* FIX: Nonpositive values entered to N4 when calculating BOLDrefs (#389) @oesteban
* FIX: Retain newlines in corrected LTA files (#391) @davhunt
* FIX: Handle singleton decompositions (#383) @rciric
* FIX: Revision of previous PR #337 / MELODIC did not converge (#381) @oesteban
* MAINT:Confound metadata maintenance (#382) @rciric
* TEST: Skip tests with non-Python dependencies when missing (#387) @effigies

0.10.2 (July 24, 2019)
----------------------
Patch release culminating the migration of workflows from fMRIPrep.

* TST: Bring EPI brain extraction tests from fMRIPrep (#380) @oesteban

0.10.1 (July 22, 2019)
----------------------
Minor release with bug fixes and pinning the latest stable release of the TemplateFlow client.

* PIN: latest templateflow client (0.4.1) @oesteban
* FIX: Load file with mmap-False when modifying on-disk dtype (#378) @effigies
* FIX: Require scikit-learn because nilearn does not (#376) @effigies

0.10.0 (July 12, 2019)
----------------------
Minor release to allow dependent tools to upgrade to PyBIDS 0.9 series (minimum 0.9.2).
We've also moved to a ``setup.cfg``-based setup to standardize configuration.

* MAINT: Use PyBIDS 0.9.2+ (#369) @effigies
* MAINT: Switch to a ``setup.cfg``-based setup (#375) @effigies

0.9.x series
============
0.9.6 (July 8, 2019)
--------------------
Minor improvements to support some of the requirements generated during the development of fMRIPrep-1.4.1.

* ENH: Improvements to ``RobustMNINormalization`` (#368) @oesteban
* RF: Miscellaneous improvements to allow multiplicity of templates and specs (#366) @oesteban


0.9.5 (June 5, 2019)
--------------------
Minor improvements to allow more flexible template selection within
the brain extraction workflow, in particular to enable using infant and
pediatric templates.

* ENH: Accept template specifications in ``antsBrainExtraction`` (#364) @oesteban


0.9.4 (June 5, 2019)
--------------------
A housekeeping release, including bugfixes and minor enhancements.
With thanks to William H. Thompson for contributions.

* PIN: TemplateFlow to latest (0.3.0), including infant and pediatric templates (#363) @oesteban
* RF: Move BOLD-reference generation workflows to niworkflows (#362) @oesteban
* ENH: Create informative HTML reportlet on missing MELODIC mix (#337) @effigies
* ENH: Signal extraction of parcels/ROIs from single NIfTI file (#303) @wiheto

0.9.3 (May 15, 2019)
--------------------
Hotfix to the new confounds plot showing correlations.

* FIX: Refine implementation of plotting confounds correlations (#360) @oesteban

0.9.2-1 (May 6, 2019)
---------------------
Hotfix to CopyXForm interface to keep backwards compatibility.

* FIX: fields were being replaced in outputs call (b418733) @oesteban

0.9.2 (May 6, 2019)
-------------------
Hotfix addressing x-form issues on our ``antsBrainExtraction``'s interpretation.

* ENH: Ensure consistency of headers along brain extraction workflow (#359) @oesteban


0.9.1-1 (May 3, 2019)
---------------------
A hotfix over latest hotfix.

  * FIX: Minor bug introduced with #358 (`ed7a8e <https://github.com/nipreps/niworkflows/commit/ed7a8e6ca350d06ff5f4d9fe8bd7ed2f06ada9ad>`__) @oesteban

0.9.1 (May 3, 2019)
-------------------
A hotfix release to allow new documentation building of fMRIPrep.

  * FIX: Tolerate missing ANTs at workflow construction (#358) @effigies

0.9.0 (May 3, 2019)
-------------------
A number of new features and bugfixes. This release includes a refactor of the
reports generation system that attempts to better generalize to other BIDS-Apps.
The new reports internally use pybids to find reportlets, and the configuration
file has been migrated to YAML to allow line breaks when captioning reportlets.
The release also provides more infrastructure for fMRIPrep and sMRIPrep, including
some BIDS-related interfaces.

  * ENH: Miscellaneous improvements to the Reports (#357) @oesteban
  * ENH: Add a ``KeySelect`` interface (#347) @oesteban
  * FIX: BusError in ``DerivativesDataSink`` (#356) @effigies
  * Revert "FIX: BusError in ``DerivativesDataSink``" (#355) @effigies
  * FIX: ``GenerateSamplingReference`` failed extension with #348 (#354) @oesteban
  * FIX: Revise tests after sloppy merge of #352 (#353) @oesteban
  * FIX: Reportlets path and output path were wrong (#352) @oesteban
  * FIX: Use safe loader for YAML data input in reports (#351) @oesteban
  * FIX: Allow ``native`` grids (i.e. pass-through) for ``GenerateSamplingReference`` (#348) @oesteban
  * FIX: BusError in ``DerivativesDataSink`` (#350) @effigies
  * ENH: Add new confounds model to reports template (#349) @oesteban
  * ENH/FIX: Migrate default config to YAML, fix ROIs query. (#346) @oesteban
  * REL: Synchronization with latest fMRIPrep changes + minor improvements (#345) @oesteban
  * ENH: ``DerivativesDataSink`` now accepts metadata dictionaries too (#332) @oesteban
  * ENH: Upstream ``init_gifti_surface_wf`` from sMRIPrep (#328) @oesteban
  * FIX: Do not generate 4D references out of 4D single-band references (SBRefs) (#338) @oesteban
  * FIX: Allow pipelining dynamic outputs of ``ReadSidecarJSON`` (#340) @oesteban
  * ENH: Dictionary manipulation / TSV to dict, merge multiple dicts (#341) @rciric
  * ENH: Run a second ``N4BiasFieldCorrection`` node to refine INU correction (#342) @oesteban
  * ENH: Add an ``allowed_entities`` setting in ``DerivativesDataSink`` (#343) @oesteban
  * ENH: Refactor of the Report generation tools (#344) @oesteban
  * PIN: Update dependencies - nilearn!=0.5.0,!=0.5.1 and latest templateflow (0.1.7)

0.8.x series
============
0.8.2 (April 4, 2019)
---------------------
New release to go along with the upcoming MRIQC 0.15.0.

  * ENH: Update CompCor plotting to allow getting NaNs (#326) @rciric
  * ENH: Ensure brain mask's conformity (#324) @oesteban
  * ENH: Add several helper interfaces (#325) @oesteban
  * FIX: "NONE of the components..." banner was printed even when no AROMA file was present (#330) @oesteban


0.8.1 (March 15, 2019)
----------------------
  * FIX: Revising antsBrainExtraction dual workflow (#316) @oesteban
  * ENH: Expose bias-corrected T1w before skull-stripping (#317) @oesteban
  * ENH: ``DerivativesDataSink`` - enable JSON sidecar writing (#318) @oesteban

0.8.0 (March 05, 2019)
----------------------
  * [PIN] Update to TemplateFlow 0.1.0 (#315) @oesteban

0.7.x series
============
0.7.2 (February 19, 2019)
-------------------------
  * [FIX] Scaling of confound fix (#310) @wiheto
  * [FIX] GenerateSamplingReference with correct zooms (#312) @effigies
  * [ENH] AROMA plots - add warning for edge cases (none/all are noise) (#292) @jdkent
  * [ENH] Confound enhancement (#287) @rciric


0.7.1.post1 (February 12, 2019)
-------------------------------
  * [FIX] Do not cast ``run`` BIDS-entity to string (#307) @oesteban


0.7.1 (February 07, 2019)
-------------------------
  * [TST] Add test on ``BIDSInfo`` interface (#302) @oesteban
  * [MNT] Deprecate ``getters`` module (#305) @oesteban
  * [FIX] Improve bounding box computation from masks (#304) @oesteban


0.7.0 (February 04, 2019)
-------------------------
  * [ENH] Implementation of BIDS utilities using pybids (#299) @oesteban
  * [HOTFIX] Only check headers of NIfTI files (#300) @oesteban
  * [ENH] Option to sanitize NIfTI headers when writing derivatives (#298) @oesteban
  * [ENH] Do not save the original name and time stamp of gzip files (#295) @oesteban
  * [CI] Checkout source for coverage reporting (#290) @effigies
  * [CI] Add coverage (#288) @effigies

Old 0.x series
==============
0.6.1 (January 23, 2019)
------------------------
  * [FIX] Allow arbitrary template names in ``RobustMNINormalization`` (#284) @oesteban
  * [FIX] Brain extraction broken connection (#286) @oesteban


0.6.0 (January 18, 2019)
------------------------
  * [RF] Improve readability of parameters files (#276) @oesteban
  * [ENH] Improve niwflows.interfaces.freesurfer (#277) @oesteban
  * [ENH] Make BIDS regex more readable (#278) @oesteban
  * [ENH] Datalad+templateflow integration (#280) @oesteban


0.5.4 (January 23, 2019)
------------------------
  * [HOTFIX] Fix ``UnboundLocalError`` in utils.bids (#285) @oesteban


0.5.3 (January 08, 2019)
------------------------
  * [RF] Improve generalization of Reports generation (#275)
  * [RF] Improve implementation of DerivativesDataSink (#274)
  * [RF] Conform names to updated TemplateFlow, add options conducive to small animal neuroimaging (#271)
  * [FIX] Do not resolve non-existent Paths (#272)

0.5.2.post5 (December 14, 2018)
-------------------------------
  * [FIX] ``read_crashfile`` stopped working after migration (#270)

0.5.2.post4 (December 13, 2018)
-------------------------------
  * [HOTFIX] ``LiterateWorkflow`` returning empty desc (#269)

0.5.2.post3 (December 13, 2018)
-------------------------------
  * [FIX] Summary fMRIPlot chokes when confounds are all-nan (#268)

0.5.2.post2 (December 12, 2018)
-------------------------------
  * [FIX] ``get_metadata_for_nifti`` broken in transfer from fmriprep (#267)

0.5.2.post1 (December 10, 2018)
-------------------------------
A hotfix release that ensures version is correctly reported when installed
via Pypi.

  * [MAINT] Clean-up dependencies (7a76a45)
  * [HOTFIX] Ensure VERSION file is created at deployment (3e3a2f3)
  * [TST] Add tests missed out in #263 (#266)

0.5.2 (December 8, 2018)
-------------------------
With thanks to @wiheto for contributions.

  * [ENH] Upstream work from fMRIPrep (prep. sMRIPrep) (#263)
  * [ENH] Integrate versioneer (#264)
  * [FIX] X axis label for fMRIPlot - better respect TR and default to frame number (#261)

0.5.1 (November 8, 2018)
------------------------
* [FIX] Count non-steady-state volumes even if sbref is passed  (#258)
* [FIX] Remove empty nipype file (#259)

0.5.0 (October 26, 2018)
------------------------
* [RF] Updates for templateflow (#257)

0.4.4 (October 15, 2018)
------------------------
* [ENH] Add "fMRIPrep" template, with new boldref template (#255)
* [ENH/MAINT] Refactor downloads, update NKI (#256)

0.4.3 (September 4, 2018)
-------------------------
* [FIX] Return runtime from EstimateReferenceImage._run_interface (#251)
* [ENH] Add nipype reimplementation of antsBrainExtraction (#244)
* [REF] Use runtime.cwd when possible in interfaces (#249)

0.4.2 (July 5, 2018)
--------------------
* [ENH] Add fs-32k template (#243)
* [FIX] Avoid mmap when overwriting input in copyxform (#247)
* [PIN] nipype 1.1.0 (#248)

0.4.1 (June 7, 2018)
--------------------
* [FIX] Standardize DTK template name

0.4.0 (May 31, 2018)
--------------------
* [ENH] Resume external nipype dependency at version 1.0.4 (#241)
* [REF] Use nipype's ReportCapableInterface mixin (#238)
* [MNT] Enable running tests in parallel (#240)

0.3.13 (May 11, 2018)
---------------------
* [PIN] Update Nipype to current master in nipy/nipype

0.3.12 (May 05, 2018)
---------------------
With thanks to @danlurie for this new feature.

* [ENH] Constrained cost-function masking for T1-MNI registration (#233)

0.3.8 (April 20, 2018)
----------------------
* [PIN] Update nipype PIN to current master

0.3.7 (March 22, 2018)
----------------------
* [ENH] fMRI summary plot to take ``_confounds.tsv`` (#230)

0.3.6 (March 14, 2018)
----------------------
Celebrating the 30th Anniversary of Pi Day!

* [ENH] Migrate the summary plot to niworkflows (#229)
* [ENH] Migrate carpetplot from MRIQC (#223)

0.3.5 (February 28, 2018)
-------------------------
With thanks to @mgxd for the new atlas.

* [PIN] Nipype-1.0.2
* [ENH] Add OASIS joint-fusion label atlas (#228)

0.3.4 (February 22, 2018)
-------------------------
* [ENH] Remove extensions from the nifti header (`#226 <https://github.com/nipreps/niworkflows/pull/226>`_)
* [FIX] Fixing conda version (`#227 <https://github.com/nipreps/niworkflows/pull/227>`_)
* [TST] Speed-up long tests (`#225 <https://github.com/nipreps/niworkflows/pull/225>`_)
* [TST] Migrate to CircleCI 2.0 (`#224 <https://github.com/nipreps/niworkflows/pull/224>`_)

0.3.3
-----
* [ENH] Added SanitizeImage interface (https://github.com/nipreps/niworkflows/pull/221)

0.3.1
-----
* [FIX] broken normalization retries (https://github.com/nipreps/niworkflows/pull/220)

0.3.0
-----
* [PIN] Nipype 1.0.0

0.2.8
-----
* [PIN] Pinning nipype to oesteban/nipype (including
  nipy/nipype#2383, nipy/nipype#2384, nipy/nipype#2376)

0.2.7
-----
* [PIN] Pinning nipype to nipy/nipype (including
  https://github.com/nipy/nipype/pull/2373)

0.2.6
-----
* [PIN] Pinning nipype to oesteban/nipype (including
  https://github.com/nipy/nipype/pull/2368)

0.2.5
-----
* [PIN] Pinning nipype to nipy/nipype@master

0.2.4
-----
* [FIX] Regression of nipreps/fmriprep#868 - updated nipy/nipype#2325
  to fix it.

0.2.3
-----
* [PIN] Upgrade internal Nipype to current master + current nipy/nipype#2325
* [ENH] Thinner lines in tissue segmentation (#215)
* [ENH] Use nearest for coreg visualization (#214)

0.2.2
-----
* [PIN] Upgrade internal Nipype to current master + nipy/nipype#2325

0.2.1
-----
* [ENH] Add new ROIsPlot interface (#211)
* [PIN] Upgrade internal Nipype to current master.

0.2.0
-----
* [ENH] Generate SVGs only (#210)
* [PIN] Upgrade internal Nipype to master after the v0.14.0 release.

0.1.11
------

* [ENH] Update internal Nipype including merging nipy/nipype#2285 before nipype itself does.

0.1.10
------

* [ENH] Lower priority of "Affines do not match" warning (#209)
* [FIX] Increase tolerance in GenerateSamplingReference (#207)
* [ENH] Upgrade internal Nipype

0.1.9
-----
* [ENH] Display surface contours for MRICoregRPT if available (#204)
* [ENH] Crop BOLD sampling reference to reduce output file size (#205)
* [ENH] Close file descriptors where possible to avoid OS limits (#208)
* [ENH] Upgrade internal Nipype

0.1.8
-----
* [ENH] Add NKI template data grabber (#200)
* [ENH] Enable sbref to be passed to EstimateReferenceImage (#199)
* [ENH] Add utilities for fixing NIfTI qform/sform matrices (#202)
* [ENH] Upgrade internal Nipype

0.1.7
-----
* [ENH] Reporting interface for `mri_coreg`
* [ENH] Upgrade internal Nipype

0.1.6
-----
* [ENH] Add BIDS example getters (#189)
* [ENH] Add NormalizeMotionParams interface (#190)
* [ENH] Add ICA-AROMA reporting interface (#193)
* [FIX] Correctly handle temporal units in MELODIC plotting (#192)
* [ENH] Upgrade internal Nipype

0.1.5
-----
* [ENH] Do not enforce float precision for ANTs (#187)
* [ENH] Clear header extensions when making ref image (#188)
* [ENH] Upgrade internal Nipype

0.1.4
-----
* [ENH] Upgrade internal Nipype

0.1.3
-----
* [ENH] Upgrade internal Nipype

0.1.2
-----
* Hotfix release (updated manifest)

0.1.1
-----
* Hotfix release (updated manifest)

0.1.0
-----
* [ENH] Improve dependency management for users unable to use Docker/Singularity containers (#174)
* [DEP] Removed RobustMNINormalization `testing` input; use `flavor-'testing'` instead (#172)

0.0.7
-----
* [ENH] Use AffineInitializer in RobustMNIRegistration (#169, #171)
* [ENH] Add CopyHeader interface (#168)
* [ENH] Add 3dUnifize to skull-stripping workflow (#167, #170)
* [ENH] Give access to num_threads in N4BiasFieldCorrection (#166)
* [ENH] Add a simple interface for visualising masks (#161)
* [ENH] Add a family of faster registration settings (#157)
* [ENH] More flexible settings for RobustMNIRegistration (#155)
* [ENH] Add EstimateReferenceImage interface (#148)
* [ENH] Add a SimpleBeforeAfter report capable interface (#144)
* [ENH] Add MELODIC report interface (#134)

0.0.6
-----
* [FIX] Python 2.7 issues and testing (#130, #135)
* [ENH] Compress surface segmentation reports (#133)
* [ENH] Write bias image in skull-stripping workflow (#131)
* [FIX] BBRegisterRPT: Use `inputs.subjects_dir` to find structurals (#128)
* [ENH] Fetch full 2009c from OSF (#126)
* [ENH] Coregistration tweaks (#125)
* [FIX] Be more robust in detecting SVGO (#124)
* [ENH] Enable Lanczos interpolation (#122)

0.0.3
-----
* Add parcellation derived from Harvard-Oxford template, to be
  used with the nonlinear-asym-09c template for the carpetplot
* Add headmask and normalize tpms in mni_icbm152_nlin_asym_09c
* Update MNI ICBM152 templates (linear and nonlinear-asym)
* Add MNI152 2009c nonlinear-symetric template (LAS)
* Add MNI152 nonlinear-symmetric template
* Add MNI EPI template and parcellation
* Switch data downloads from GDrive to OSF
* Fixed installer, now compatible with python 3

0.0.2
-----
* Added MRI reorient workflow (based on AFNI)


0.0.1
-----
* Added skull-stripping workflow based on AFNI
* Rewritten most of the shablona-derived names and description files
* Copied project structure from Shablona
