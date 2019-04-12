0.8.2 (April 4, 2019)
=====================

New release to go along with the upcoming MRIQC 0.15.0.

  * ENH: Update CompCor plotting to allow getting NaNs (#326) @rciric
  * ENH: Ensure brain mask's conformity (#324) @oesteban
  * ENH: Add several helper interfaces (#325) @oesteban
  * FIX: "NONE of the components..." banner was printed even when no AROMA file was present (#330) @oesteban

  
0.8.1 (March 15, 2019)
======================

  * FIX: Revising antsBrainExtraction dual workflow (#316) @oesteban
  * ENH: Expose bias-corrected T1w before skull-stripping (#317) @oesteban
  * ENH: ``DerivativesDataSink`` - enable JSON sidecar writing (#318) @oesteban

0.8.0 (March 05, 2019)
======================

  * [PIN] Update to TemplateFlow 0.1.0 (#315) @oesteban

0.7.2 (February 19, 2019)
=========================

  * [FIX] Scaling of confound fix (#310) @wiheto
  * [FIX] GenerateSamplingReference with correct zooms (#312) @effigies
  * [ENH] AROMA plots - add warning for edge cases (none/all are noise) (#292) @jdkent
  * [ENH] Confound enhancement (#287) @rciric


0.7.1.post1 (February 12, 2019)
===============================

  * [FIX] Do not cast ``run`` BIDS-entity to string (#307) @oesteban


0.7.1 (February 07, 2019)
=========================

  * [TST] Add test on ``BIDSInfo`` interface (#302) @oesteban
  * [MNT] Deprecate ``getters`` module (#305) @oesteban
  * [FIX] Improve bounding box computation from masks (#304) @oesteban


0.7.0 (February 04, 2019)
=========================

  * [ENH] Implementation of BIDS utilities using pybids (#299) @oesteban
  * [HOTFIX] Only check headers of NIfTI files (#300) @oesteban
  * [ENH] Option to sanitize NIfTI headers when writing derivatives (#298) @oesteban
  * [ENH] Do not save the original name and time stamp of gzip files (#295) @oesteban
  * [CI] Checkout source for coverage reporting (#290) @effigies
  * [CI] Add coverage (#288) @effigies


0.6.1 (January 23, 2019)
========================

  * [FIX] Allow arbitrary template names in ``RobustMNINormalization`` (#284) @oesteban
  * [FIX] Brain extraction broken connection (#286) @oesteban


0.6.0 (January 18, 2019)
========================

  * [RF] Improve readability of parameters files (#276) @oesteban
  * [ENH] Improve niwflows.interfaces.freesurfer (#277) @oesteban
  * [ENH] Make BIDS regex more readable (#278) @oesteban
  * [ENH] Datalad+templateflow integration (#280) @oesteban 


0.5.4 (January 23, 2019)
========================

  * [HOTFIX] Fix ``UnboundLocalError`` in utils.bids (#285) @oesteban


0.5.3 (January 08, 2019)
========================

  * [RF] Improve generalization of Reports generation (#275)
  * [RF] Improve implementation of DerivativesDataSink (#274)
  * [RF] Conform names to updated TemplateFlow, add options conducive to small animal neuroimaging (#271)
  * [FIX] Do not resolve non-existent Paths (#272)

0.5.2.post5 (December 14, 2018)
===============================

  * [FIX] ``read_crashfile`` stopped working after migration (#270)

0.5.2.post4 (December 13, 2018)
===============================

  * [HOTFIX] ``LiterateWorkflow`` returning empty desc (#269)

0.5.2.post3 (December 13, 2018)
===============================

  * [FIX] Summary fMRIPlot chokes when confounds are all-nan (#268)

0.5.2.post2 (December 12, 2018)
===============================

  * [FIX] ``get_metadata_for_nifti`` broken in transfer from fmriprep (#267)

0.5.2.post1 (December 10, 2018)
===============================

A hotfix release that ensures version is correctly reported when installed
via Pypi.

  * [MAINT] Clean-up dependencies (7a76a45)
  * [HOTFIX] Ensure VERSION file is created at deployment (3e3a2f3)
  * [TST] Add tests missed out in #263 (#266)

0.5.2 (December 8, 2018)
=========================

With thanks to @wiheto for contributions.

  * [ENH] Upstream work from fMRIPrep (prep. sMRIPrep) (#263)
  * [ENH] Integrate versioneer (#264)
  * [FIX] X axis label for fMRIPlot - better respect TR and default to frame number (#261)

0.5.1 (November 8, 2018)
========================

* [FIX] Count non-steady-state volumes even if sbref is passed  (#258)
* [FIX] Remove empty nipype file (#259)

0.5.0 (October 26, 2018)
========================

* [RF] Updates for templateflow (#257)

0.4.4 (October 15, 2018)
========================

* [ENH] Add "fMRIPrep" template, with new boldref template (#255)
* [ENH/MAINT] Refactor downloads, update NKI (#256)

0.4.3 (September 4, 2018)
=========================

* [FIX] Return runtime from EstimateReferenceImage._run_interface (#251)
* [ENH] Add nipype reimplementation of antsBrainExtraction (#244)
* [REF] Use runtime.cwd when possible in interfaces (#249)

0.4.2 (July 5, 2018)
====================

* [ENH] Add fs-32k template (#243)
* [FIX] Avoid mmap when overwriting input in copyxform (#247)
* [PIN] nipype 1.1.0 (#248)

0.4.1 (June 7, 2018)
====================

* [FIX] Standardize DTK template name 

0.4.0 (May 31, 2018)
====================

* [ENH] Resume external nipype dependency at version 1.0.4 (#241)
* [REF] Use nipype's ReportCapableInterface mixin (#238)
* [MNT] Enable running tests in parallel (#240)

0.3.13 (May 11, 2018)
=====================

* [PIN] Update Nipype to current master in nipy/nipype

0.3.12 (May 05, 2018)
=====================

With thanks to @danlurie for this new feature.

* [ENH] Constrained cost-function masking for T1-MNI registration (#233)

0.3.8 (April 20, 2018)
======================

* [PIN] Update nipype PIN to current master

0.3.7 (March 22, 2018)
======================

* [ENH] fMRI summary plot to take ``_confounds.tsv`` (#230)

0.3.6 (March 14, 2018)
======================

Celebrating the 30th Anniversary of Pi Day!

* [ENH] Migrate the summary plot to niworkflows (#229)
* [ENH] Migrate carpetplot from MRIQC (#223)

0.3.5 (February 28, 2018)
=========================

With thanks to @mgxd for the new atlas.

* [PIN] Nipype-1.0.2
* [ENH] Add OASIS joint-fusion label atlas (#228)

0.3.4 (February 22, 2018)
=========================

* [ENH] Remove extensions from the nifti header (`#226 <https://github.com/poldracklab/niworkflows/pull/226>`_)
* [FIX] Fixing conda version (`#227 <https://github.com/poldracklab/niworkflows/pull/227>`_)
* [TST] Speed-up long tests (`#225 <https://github.com/poldracklab/niworkflows/pull/225>`_)
* [TST] Migrate to CircleCI 2.0 (`#224 <https://github.com/poldracklab/niworkflows/pull/224>`_)


Version 0.3.3
=============

* [ENH] Added SanitizeImage interface (https://github.com/poldracklab/niworkflows/pull/221)

Version 0.3.1
=============

* [FIX] broken normalization retries (https://github.com/poldracklab/niworkflows/pull/220)

Version 0.3.0
=============

* [PIN] Nipype 1.0.0

Version 0.2.8
=============

* [PIN] Pinning nipype to oesteban/nipype (including
  nipy/nipype#2383, nipy/nipype#2384, nipy/nipype#2376)

Version 0.2.7
=============

* [PIN] Pinning nipype to nipy/nipype (including
  https://github.com/nipy/nipype/pull/2373)

Version 0.2.6
=============

* [PIN] Pinning nipype to oesteban/nipype (including
  https://github.com/nipy/nipype/pull/2368)

Version 0.2.5
=============

* [PIN] Pinning nipype to nipy/nipype@master

Version 0.2.4
=============

* [FIX] Regression of poldracklab/fmriprep#868 - updated nipy/nipype#2325
  to fix it.

Version 0.2.3
=============

* [PIN] Upgrade internal Nipype to current master + current nipy/nipype#2325
* [ENH] Thinner lines in tissue segmentation (#215)
* [ENH] Use nearest for coreg visualization (#214)

Version 0.2.2
=============

* [PIN] Upgrade internal Nipype to current master + nipy/nipype#2325

Version 0.2.1
=============

* [ENH] Add new ROIsPlot interface (#211)
* [PIN] Upgrade internal Nipype to current master.

Version 0.2.0
=============

* [ENH] Generate SVGs only (#210)
* [PIN] Upgrade internal Nipype to master after the v0.14.0 release.

Version 0.1.11
=============-

* [ENH] Update internal Nipype including merging nipy/nipype#2285 before nipype itself does.

Version 0.1.10
=============-

* [ENH] Lower priority of "Affines do not match" warning (#209)
* [FIX] Increase tolerance in GenerateSamplingReference (#207)
* [ENH] Upgrade internal Nipype

Version 0.1.9
=============

* [ENH] Display surface contours for MRICoregRPT if available (#204)
* [ENH] Crop BOLD sampling reference to reduce output file size (#205)
* [ENH] Close file descriptors where possible to avoid OS limits (#208)
* [ENH] Upgrade internal Nipype

Version 0.1.8
=============

* [ENH] Add NKI template data grabber (#200)
* [ENH] Enable sbref to be passed to EstimateReferenceImage (#199)
* [ENH] Add utilities for fixing NIfTI qform/sform matrices (#202)
* [ENH] Upgrade internal Nipype

Version 0.1.7
=============

* [ENH] Reporting interface for `mri_coreg`
* [ENH] Upgrade internal Nipype

Version 0.1.6
=============

* [ENH] Add BIDS example getters (#189)
* [ENH] Add NormalizeMotionParams interface (#190)
* [ENH] Add ICA-AROMA reporting interface (#193)
* [FIX] Correctly handle temporal units in MELODIC plotting (#192)
* [ENH] Upgrade internal Nipype

Version 0.1.5
=============

* [ENH] Do not enforce float precision for ANTs (#187)
* [ENH] Clear header extensions when making ref image (#188)
* [ENH] Upgrade internal Nipype

Version 0.1.4
=============

* [ENH] Upgrade internal Nipype

Version 0.1.3
=============

* [ENH] Upgrade internal Nipype

Version 0.1.2
=============

* Hotfix release (updated manifest)

Version 0.1.1
=============

* Hotfix release (updated manifest)

Version 0.1.0
=============

* [ENH] Improve dependency management for users unable to use Docker/Singularity containers (#174)
* [DEP] Removed RobustMNINormalization `testing` input; use `flavor='testing'` instead (#172)

Version 0.0.7
=============

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

Version 0.0.6
=============

* [FIX] Python 2.7 issues and testing (#130, #135)
* [ENH] Compress surface segmentation reports (#133)
* [ENH] Write bias image in skull-stripping workflow (#131)
* [FIX] BBRegisterRPT: Use `inputs.subjects_dir` to find structurals (#128)
* [ENH] Fetch full 2009c from OSF (#126)
* [ENH] Coregistration tweaks (#125)
* [FIX] Be more robust in detecting SVGO (#124)
* [ENH] Enable Lanczos interpolation (#122)

Version 0.0.5
=============


Version 0.0.3
=============

* Add parcellation derived from Harvard-Oxford template, to be
  used with the nonlinear-asym-09c template for the carpetplot
* Add headmask and normalize tpms in mni_icbm152_nlin_asym_09c
* Update MNI ICBM152 templates (linear and nonlinear-asym)
* Add MNI152 2009c nonlinear-symetric template (LAS)
* Add MNI152 nonlinear-symmetric template
* Add MNI EPI template and parcellation
* Switch data downloads from GDrive to OSF
* Fixed installer, now compatible with python 3

Version 0.0.2
=============

* Added MRI reorient workflow (based on AFNI)


Version 0.0.1
=============

* Added skull-stripping workflow based on AFNI
* Rewritten most of the shablona-derived names and description files
* Copied project structure from Shablona
