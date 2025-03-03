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
"""Interfaces for handling BIDS-like neuroimaging structures."""

import os
import re
import shutil
import sys
from collections import defaultdict
from contextlib import suppress
from json import dumps, loads
from pathlib import Path

import nibabel as nb
import numpy as np
import templateflow as tf
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    Directory,
    DynamicTraitedSpec,
    File,
    InputMultiObject,
    OutputMultiObject,
    SimpleInterface,
    Str,
    TraitedSpec,
    Undefined,
    isdefined,
    traits,
)
from nipype.interfaces.io import add_traits
from nipype.utils.filemanip import hash_infile

from .. import data
from ..utils.bids import _init_layout, relative_to_root
from ..utils.images import set_consumables, unsafe_write_nifti_header_and_data
from ..utils.misc import _copy_any, unlink

regz = re.compile(r'\.gz$')
_pybids_spec = loads(data.load.readable('nipreps.json').read_text())
BIDS_DERIV_ENTITIES = _pybids_spec['entities']
BIDS_DERIV_PATTERNS = tuple(_pybids_spec['default_path_patterns'])

STANDARD_SPACES = tf.api.templates()
LOGGER = logging.getLogger('nipype.interface')


if sys.version_info < (3, 10):  # PY39
    builtin_zip = zip

    def zip(*args, strict=False):  # noqa: A001
        if strict and any(len(args[0]) != len(arg) for arg in args):
            raise ValueError('strict_zip() requires all arguments to have the same length')
        return builtin_zip(*args)


def _none():
    return None


# Automatically coerce certain suffixes (DerivativesDataSink)
DEFAULT_DTYPES = defaultdict(
    _none,
    (
        ('mask', 'uint8'),
        ('dseg', 'int16'),
        ('probseg', 'float32'),
        ('boldref', 'float32'),
    ),
)


class _BIDSBaseInputSpec(BaseInterfaceInputSpec):
    bids_dir = traits.Either(
        (None, Directory(exists=True)), usedefault=True, desc='optional bids directory'
    )
    bids_validate = traits.Bool(True, usedefault=True, desc='enable BIDS validator')
    index_db = Directory(exists=True, desc='a PyBIDS layout cache directory')


class _BIDSInfoInputSpec(_BIDSBaseInputSpec):
    in_file = File(mandatory=True, desc='input file, part of a BIDS tree')


class _BIDSInfoOutputSpec(DynamicTraitedSpec):
    subject = traits.Str()
    session = traits.Str()
    task = traits.Str()
    acquisition = traits.Str()
    reconstruction = traits.Str()
    run = traits.Int()
    suffix = traits.Str()


class BIDSInfo(SimpleInterface):
    """
    Extract BIDS entities from a BIDS-conforming path.

    This interface uses only the basename, not the path, to determine the
    subject, session, task, run, acquisition or reconstruction.

    .. testsetup::

        >>> data_dir_canary()

    >>> bids_info = BIDSInfo(bids_dir=str(datadir / 'ds054'), bids_validate=False)
    >>> bids_info.inputs.in_file = '''\
sub-01/func/ses-retest/sub-01_ses-retest_task-covertverbgeneration_bold.nii.gz'''
    >>> res = bids_info.run()
    >>> res.outputs
    <BLANKLINE>
    acquisition = <undefined>
    reconstruction = <undefined>
    run = <undefined>
    session = retest
    subject = 01
    suffix = bold
    task = covertverbgeneration
    <BLANKLINE>

    >>> bids_info = BIDSInfo(bids_dir=str(datadir / 'ds054'), bids_validate=False)
    >>> bids_info.inputs.in_file = '''\
sub-01/func/ses-retest/sub-01_ses-retest_task-covertverbgeneration_rec-MB_acq-AP_run-01_bold.nii.gz'''
    >>> res = bids_info.run()
    >>> res.outputs
    <BLANKLINE>
    acquisition = AP
    reconstruction = MB
    run = 1
    session = retest
    subject = 01
    suffix = bold
    task = covertverbgeneration
    <BLANKLINE>

    >>> bids_info = BIDSInfo(bids_dir=str(datadir / 'ds054'), bids_validate=False)
    >>> bids_info.inputs.in_file = '''\
sub-01/func/ses-retest/sub-01_ses-retest_task-covertverbgeneration_acq-AP_run-01_bold.nii.gz'''
    >>> res = bids_info.run()
    >>> res.outputs
    <BLANKLINE>
    acquisition = AP
    reconstruction = <undefined>
    run = 1
    session = retest
    subject = 01
    suffix = bold
    task = covertverbgeneration
    <BLANKLINE>

    >>> bids_info = BIDSInfo(bids_validate=False)
    >>> bids_info.inputs.in_file = str(
    ...     datadir / 'ds114' / 'sub-01' / 'ses-retest' /
    ...     'func' / 'sub-01_ses-retest_task-covertverbgeneration_bold.nii.gz')
    >>> res = bids_info.run()
    >>> res.outputs
    <BLANKLINE>
    acquisition = <undefined>
    reconstruction = <undefined>
    run = <undefined>
    session = retest
    subject = 01
    suffix = bold
    task = covertverbgeneration
    <BLANKLINE>

    >>> bids_info = BIDSInfo(bids_validate=False)
    >>> bids_info.inputs.in_file = '''\
sub-01/func/ses-retest/sub-01_ses-retest_task-covertverbgeneration_bold.nii.gz'''
    >>> res = bids_info.run()
    >>> res.outputs
    <BLANKLINE>
    acquisition = <undefined>
    reconstruction = <undefined>
    run = <undefined>
    session = retest
    subject = 01
    suffix = bold
    task = covertverbgeneration
    <BLANKLINE>

    """

    input_spec = _BIDSInfoInputSpec
    output_spec = _BIDSInfoOutputSpec

    def _run_interface(self, runtime):
        from bids.layout import parse_file_entities

        bids_dir = self.inputs.bids_dir
        in_file = self.inputs.in_file
        if bids_dir is not None:
            try:
                in_file = str(Path(in_file).relative_to(bids_dir))
            except ValueError:
                pass
        params = parse_file_entities(in_file)
        self._results = {key: params.get(key, Undefined) for key in _BIDSInfoOutputSpec().get()}
        return runtime


class _BIDSDataGrabberInputSpec(BaseInterfaceInputSpec):
    subject_data = traits.Dict(Str, traits.Any)
    subject_id = Str()


class _BIDSDataGrabberOutputSpec(TraitedSpec):
    out_dict = traits.Dict(desc='output data structure')
    fmap = OutputMultiObject(desc='output fieldmaps')
    bold = OutputMultiObject(desc='output functional images')
    sbref = OutputMultiObject(desc='output sbrefs')
    t1w = OutputMultiObject(desc='output T1w images')
    roi = OutputMultiObject(desc='output ROI images')
    t2w = OutputMultiObject(desc='output T2w images')
    flair = OutputMultiObject(desc='output FLAIR images')
    pet = OutputMultiObject(desc='output PET images')
    dwi = OutputMultiObject(desc='output DWI images')
    asl = OutputMultiObject(desc='output ASL images')


class BIDSDataGrabber(SimpleInterface):
    """
    Collect files from a BIDS directory structure.

    .. testsetup::

        >>> data_dir_canary()

    >>> bids_src = BIDSDataGrabber(anat_only=False)
    >>> bids_src.inputs.subject_data = bids_collect_data(
    ...     str(datadir / 'ds114'), '01', bids_validate=False)[0]
    >>> bids_src.inputs.subject_id = '01'
    >>> res = bids_src.run()
    >>> res.outputs.t1w  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['.../ds114/sub-01/ses-retest/anat/sub-01_ses-retest_T1w.nii.gz',
     '.../ds114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz']

    """

    input_spec = _BIDSDataGrabberInputSpec
    output_spec = _BIDSDataGrabberOutputSpec
    _require_funcs = True

    def __init__(self, *args, **kwargs):
        anat_only = kwargs.pop('anat_only', None)
        anat_derivatives = kwargs.pop('anat_derivatives', None)
        require_t1w = kwargs.pop('require_t1w', True)
        super().__init__(*args, **kwargs)
        if anat_only is not None:
            self._require_funcs = not anat_only
        self._require_t1w = require_t1w and anat_derivatives is None

    def _run_interface(self, runtime):
        bids_dict = self.inputs.subject_data

        self._results['out_dict'] = bids_dict
        self._results.update(bids_dict)

        if self._require_t1w and not bids_dict['t1w']:
            raise FileNotFoundError(
                f'No T1w images found for subject sub-{self.inputs.subject_id}'
            )

        if self._require_funcs and not bids_dict['bold']:
            raise FileNotFoundError(
                f'No functional images found for subject sub-{self.inputs.subject_id}'
            )

        for imtype in ['bold', 't2w', 'flair', 'fmap', 'sbref', 'roi', 'pet', 'asl']:
            if not bids_dict[imtype]:
                LOGGER.info('No "%s" images found for sub-%s', imtype, self.inputs.subject_id)

        return runtime


class _PrepareDerivativeInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    check_hdr = traits.Bool(True, usedefault=True, desc='fix headers of NIfTI outputs')
    compress = InputMultiObject(
        traits.Either(None, traits.Bool),
        usedefault=True,
        desc='whether ``in_file`` should be compressed (True), uncompressed (False) '
        'or left unmodified (None, default).',
    )
    data_dtype = Str(
        desc='NumPy datatype to coerce NIfTI data to, or `source` to match the input file dtype'
    )
    dismiss_entities = InputMultiObject(
        traits.Either(None, Str),
        usedefault=True,
        desc='a list entities that will not be propagated from the source file',
    )
    in_file = InputMultiObject(File(exists=True), mandatory=True, desc='the object to be saved')
    meta_dict = traits.Dict(Str, desc='an input dictionary containing metadata')
    source_file = InputMultiObject(
        File(exists=False), mandatory=True, desc='the source file(s) to extract entities from'
    )


class _PrepareDerivativeOutputSpec(TraitedSpec):
    out_file = OutputMultiObject(File(exists=True), desc='derivative file path')
    out_meta = traits.Dict(Str, desc='derivative metadata')
    out_path = OutputMultiObject(Str, desc='relative path in target directory')
    fixed_hdr = traits.List(traits.Bool, desc='whether derivative header was fixed')


class PrepareDerivative(SimpleInterface):
    """
    Prepare derivative files and metadata.

    Collects entities from source files and inputs, filters them for allowed entities,
    and constructs a relative path within a BIDS dataset.

    For each file, the interface will determine if any changes to the file contents
    are needed, including:

    - Compression (or decompression) of the file
    - Coercion of the data type
    - Fixing the NIfTI header
      - Align qform and sform affines and codes
      - Set zooms and units

    If the input file needs to be modified, the interface will write a new file
    and return the path to it. If no changes are needed, the interface will return
    the path to the input file.

    .. testsetup::

        >>> data_dir_canary()

    >>> import tempfile
    >>> tmpdir = Path(tempfile.mkdtemp())
    >>> tmpfile = tmpdir / 'a_temp_file.nii.gz'
    >>> tmpfile.open('w').close()  # "touch" the file
    >>> t1w_source = bids_collect_data(
    ...     str(datadir / 'ds114'), '01', bids_validate=False)[0]['t1w'][0]
    >>> prep = PrepareDerivative(check_hdr=False)
    >>> prep.inputs.in_file = str(tmpfile)
    >>> prep.inputs.source_file = t1w_source
    >>> prep.inputs.desc = 'denoised'
    >>> prep.inputs.compress = False
    >>> res = prep.run()
    >>> res.outputs.out_file  # doctest: +ELLIPSIS
    '.../a_temp_file.nii.gz'
    >>> res.outputs.out_path  # doctest: +ELLIPSIS
    'sub-01/ses-retest/anat/sub-01_ses-retest_desc-denoised_T1w.nii'

    >>> tmpfile = tmpdir / 'a_temp_file.nii'
    >>> tmpfile.open('w').close()  # "touch" the file
    >>> prep = PrepareDerivative(check_hdr=False, allowed_entities=("custom",))
    >>> prep.inputs.in_file = str(tmpfile)
    >>> prep.inputs.source_file = t1w_source
    >>> prep.inputs.custom = 'noise'
    >>> res = prep.run()
    >>> res.outputs.out_file  # doctest: +ELLIPSIS
    '.../a_temp_file.nii'
    >>> res.outputs.out_path  # doctest: +ELLIPSIS
    'sub-01/ses-retest/anat/sub-01_ses-retest_custom-noise_T1w.nii'

    >>> prep = PrepareDerivative(check_hdr=False, allowed_entities=("custom",))
    >>> prep.inputs.in_file = [str(tmpfile), str(tmpfile)]
    >>> prep.inputs.source_file = t1w_source
    >>> prep.inputs.custom = [1, 2]
    >>> prep.inputs.compress = True
    >>> res = prep.run()
    >>> res.outputs.out_file  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['.../a_temp_file.nii', '.../a_temp_file.nii']
    >>> res.outputs.out_path  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['sub-01/ses-retest/anat/sub-01_ses-retest_custom-1_T1w.nii.gz',
     'sub-01/ses-retest/anat/sub-01_ses-retest_custom-2_T1w.nii.gz']

    >>> prep = PrepareDerivative(check_hdr=False, allowed_entities=("custom1", "custom2"))
    >>> prep.inputs.in_file = [str(tmpfile)] * 2
    >>> prep.inputs.source_file = t1w_source
    >>> prep.inputs.custom1 = [1, 2]
    >>> prep.inputs.custom2 = "b"
    >>> res = prep.run()
    >>> res.outputs.out_file  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['.../a_temp_file.nii', '.../a_temp_file.nii']
    >>> res.outputs.out_path  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['sub-01/ses-retest/anat/sub-01_ses-retest_custom1-1_custom2-b_T1w.nii',
     'sub-01/ses-retest/anat/sub-01_ses-retest_custom1-2_custom2-b_T1w.nii']

    When multiple source files are passed, only common entities are passed down.
    For example, if two T1w images from different sessions are used to generate
    a single image, the session entity is removed automatically.

    >>> bids_dir = tmpdir / 'bidsroot'
    >>> multi_source = [
    ...     bids_dir / 'sub-02/ses-A/anat/sub-02_ses-A_T1w.nii.gz',
    ...     bids_dir / 'sub-02/ses-B/anat/sub-02_ses-B_T1w.nii.gz']
    >>> for source_file in multi_source:
    ...     source_file.parent.mkdir(parents=True, exist_ok=True)
    ...     _ = source_file.write_text("")
    >>> prep = PrepareDerivative(check_hdr=False)
    >>> prep.inputs.in_file = str(tmpfile)
    >>> prep.inputs.source_file = list(map(str, multi_source))
    >>> prep.inputs.desc = 'preproc'
    >>> res = prep.run()
    >>> res.outputs.out_path  # doctest: +ELLIPSIS
    'sub-02/anat/sub-02_desc-preproc_T1w.nii'

    If, on the other hand, only one is used, the session is preserved:

    >>> prep.inputs.source_file = str(multi_source[0])
    >>> res = prep.run()
    >>> res.outputs.out_path  # doctest: +ELLIPSIS
    'sub-02/ses-A/anat/sub-02_ses-A_desc-preproc_T1w.nii'

    >>> bids_dir = tmpdir / 'bidsroot' / 'sub-02' / 'ses-noanat' / 'func'
    >>> bids_dir.mkdir(parents=True, exist_ok=True)
    >>> tricky_source = bids_dir / 'sub-02_ses-noanat_task-rest_run-01_bold.nii.gz'
    >>> tricky_source.open('w').close()
    >>> prep = PrepareDerivative(check_hdr=False)
    >>> prep.inputs.in_file = str(tmpfile)
    >>> prep.inputs.source_file = str(tricky_source)
    >>> prep.inputs.desc = 'preproc'
    >>> res = prep.run()
    >>> res.outputs.out_path  # doctest: +ELLIPSIS
    'sub-02/ses-noanat/func/sub-02_ses-noanat_task-rest_run-01_desc-preproc_bold.nii'

    >>> bids_dir = tmpdir / 'bidsroot' / 'sub-02' / 'ses-noanat' / 'func'
    >>> bids_dir.mkdir(parents=True, exist_ok=True)
    >>> tricky_source = bids_dir / 'sub-02_ses-noanat_task-rest_run-01_bold.nii.gz'
    >>> tricky_source.open('w').close()
    >>> prep = PrepareDerivative(check_hdr=False)
    >>> prep.inputs.in_file = str(tmpfile)
    >>> prep.inputs.source_file = str(tricky_source)
    >>> prep.inputs.desc = 'preproc'
    >>> prep.inputs.RepetitionTime = 0.75
    >>> res = prep.run()
    >>> res.outputs.out_meta  # doctest: +ELLIPSIS
    {'RepetitionTime': 0.75}

    >>> bids_dir = tmpdir / 'bidsroot' / 'sub-02' / 'ses-noanat' / 'func'
    >>> bids_dir.mkdir(parents=True, exist_ok=True)
    >>> tricky_source = bids_dir / 'sub-02_ses-noanat_task-rest_run-01_bold.nii.gz'
    >>> tricky_source.open('w').close()
    >>> prep = PrepareDerivative(check_hdr=False, SkullStripped=True)
    >>> prep.inputs.in_file = str(tmpfile)
    >>> prep.inputs.source_file = str(tricky_source)
    >>> prep.inputs.desc = 'preproc'
    >>> prep.inputs.space = 'MNI152NLin6Asym'
    >>> prep.inputs.resolution = '01'
    >>> prep.inputs.RepetitionTime = 0.75
    >>> res = prep.run()
    >>> res.outputs.out_meta  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    {'SkullStripped': True,
     'RepetitionTime': 0.75,
     'Resolution': 'Template MNI152NLin6Asym (1.0x1.0x1.0 mm^3)...'}

    >>> bids_dir = tmpdir / 'bidsroot' / 'sub-02' / 'ses-noanat' / 'func'
    >>> bids_dir.mkdir(parents=True, exist_ok=True)
    >>> tricky_source = bids_dir / 'sub-02_ses-noanat_task-rest_run-01_bold.nii.gz'
    >>> tricky_source.open('w').close()
    >>> prep = PrepareDerivative(check_hdr=False, SkullStripped=True)
    >>> prep.inputs.in_file = str(tmpfile)
    >>> prep.inputs.source_file = str(tricky_source)
    >>> prep.inputs.desc = 'preproc'
    >>> prep.inputs.resolution = 'native'
    >>> prep.inputs.space = 'MNI152NLin6Asym'
    >>> prep.inputs.RepetitionTime = 0.75
    >>> prep.inputs.meta_dict = {'RepetitionTime': 1.75, 'SkullStripped': False, 'Z': 'val'}
    >>> res = prep.run()
    >>> res.outputs.out_meta  # doctest: +ELLIPSIS
    {'RepetitionTime': 0.75, 'SkullStripped': True, 'Z': 'val'}

    """

    input_spec = _PrepareDerivativeInputSpec
    output_spec = _PrepareDerivativeOutputSpec
    _config_entities = frozenset({e['name'] for e in BIDS_DERIV_ENTITIES})
    _config_entities_dict = BIDS_DERIV_ENTITIES
    _standard_spaces = STANDARD_SPACES
    _file_patterns = BIDS_DERIV_PATTERNS
    _default_dtypes = DEFAULT_DTYPES

    def __init__(self, allowed_entities=None, **inputs):
        """Initialize the SimpleInterface and extend inputs with custom entities."""
        self._allowed_entities = set(allowed_entities or []).union(set(self._config_entities))

        self._metadata = {}
        self._static_traits = self.input_spec.class_editable_traits() + sorted(
            self._allowed_entities
        )
        for dynamic_input in set(inputs) - set(self._static_traits):
            self._metadata[dynamic_input] = inputs.pop(dynamic_input)

        # First regular initialization (constructs InputSpec object)
        super().__init__(**inputs)
        add_traits(self.inputs, self._allowed_entities)
        for k in self._allowed_entities.intersection(list(inputs.keys())):
            # Add additional input fields (self.inputs is an object)
            setattr(self.inputs, k, inputs[k])

    def _run_interface(self, runtime):
        from bids.layout import Config, parse_file_entities
        from bids.layout.writing import build_path
        from bids.utils import listify

        # Metadata applies to all files, and is not subject to change
        metadata = {
            # Lowest precedence: metadata provided as a dictionary
            **(self.inputs.meta_dict or {}),
            # Middle precedence: metadata passed to constructor
            **self._metadata,
            # Highest precedence: metadata set as inputs
            **(
                {
                    k: getattr(self.inputs, k)
                    for k in self.inputs.copyable_trait_names()
                    if k not in self._static_traits
                }
            ),
        }

        in_file = listify(self.inputs.in_file)

        # Initialize entities with those from the source file.
        custom_config = Config(
            name='custom',
            entities=self._config_entities_dict,
            default_path_patterns=self._file_patterns,
        )
        in_entities = [
            parse_file_entities(
                str(relative_to_root(source_file)),
                config=['bids', 'derivatives', custom_config],
            )
            for source_file in self.inputs.source_file
        ]
        out_entities = {
            k: v
            for k, v in in_entities[0].items()
            if all(ent.get(k) == v for ent in in_entities[1:])
        }
        for drop_entity in listify(self.inputs.dismiss_entities or []):
            out_entities.pop(drop_entity, None)

        # Override extension with that of the input file(s)
        out_entities['extension'] = [
            # _splitext does not accept .surf.gii (for instance)
            ''.join(Path(orig_file).suffixes).lstrip('.')
            for orig_file in in_file
        ]

        compress = listify(self.inputs.compress) or [None]
        if len(compress) == 1:
            compress = compress * len(in_file)
        for i, ext in enumerate(out_entities['extension']):
            if compress[i] is not None:
                ext = regz.sub('', ext)
                out_entities['extension'][i] = f'{ext}.gz' if compress[i] else ext

        # Override entities with those set as inputs
        for key in self._allowed_entities:
            value = getattr(self.inputs, key)
            if value is not None and isdefined(value):
                out_entities[key] = value

        # Clean up native resolution with space
        if out_entities.get('resolution') == 'native' and out_entities.get('space'):
            out_entities.pop('resolution', None)

        # Expand templateflow resolutions
        resolution = out_entities.get('resolution')
        space = out_entities.get('space')
        if resolution:
            # Standard spaces
            if space in self._standard_spaces:
                res = _get_tf_resolution(space, resolution)
            else:  # TODO: Nonstandard?
                res = 'Unknown'
            metadata['Resolution'] = res

        if len(set(out_entities['extension'])) == 1:
            out_entities['extension'] = out_entities['extension'][0]

        # Insert custom (non-BIDS) entities from allowed_entities.
        custom_entities = set(out_entities) - set(self._config_entities)
        patterns = self._file_patterns
        if custom_entities:
            # Example: f"{key}-{{{key}}}" -> "task-{task}"
            custom_pat = '_'.join(f'{key}-{{{key}}}' for key in sorted(custom_entities))
            patterns = [pat.replace('_{suffix', f'_{custom_pat}_{{suffix') for pat in patterns]

        # Build the output path(s)
        dest_files = build_path(out_entities, path_patterns=patterns)
        if not dest_files:
            raise ValueError(f'Could not build path with entities {out_entities}.')

        # Make sure the interpolated values is embedded in a list, and check
        dest_files = listify(dest_files)
        if len(in_file) != len(dest_files):
            raise ValueError(
                f'Input files ({len(in_file)}) not matched '
                f'by interpolated patterns ({len(dest_files)}).'
            )

        # Prepare SimpleInterface outputs object
        self._results['out_file'] = []
        self._results['fixed_hdr'] = [False] * len(in_file)
        self._results['out_path'] = dest_files
        self._results['out_meta'] = metadata

        for i, (orig_file, dest_file) in enumerate(zip(in_file, dest_files)):
            # Set data and header iff changes need to be made. If these are
            # still None when it's time to write, just copy.
            new_data, new_header = None, None

            is_nifti = False
            with suppress(nb.filebasedimages.ImageFileError):
                is_nifti = isinstance(nb.load(orig_file), nb.Nifti1Image)

            new_compression = False
            if is_nifti:
                new_compression = os.fspath(orig_file).endswith('.gz') ^ os.fspath(
                    dest_file
                ).endswith('.gz')

            data_dtype = self.inputs.data_dtype or self._default_dtypes[self.inputs.suffix]
            if is_nifti and any((self.inputs.check_hdr, data_dtype)):
                nii = nb.load(orig_file)

                if self.inputs.check_hdr:
                    hdr = nii.header
                    curr_units = tuple(None if u == 'unknown' else u for u in hdr.get_xyzt_units())
                    curr_codes = (int(hdr['qform_code']), int(hdr['sform_code']))

                    # Default to mm, use sec if data type is bold
                    units = (
                        curr_units[0] or 'mm',
                        'sec' if out_entities['suffix'] == 'bold' else None,
                    )
                    xcodes = (1, 1)  # Derivative in its original scanner space
                    if self.inputs.space:
                        xcodes = (4, 4) if self.inputs.space in self._standard_spaces else (2, 2)

                    curr_zooms = zooms = hdr.get_zooms()
                    if 'RepetitionTime' in self.inputs.get():
                        zooms = curr_zooms[:3] + (self.inputs.RepetitionTime,)

                    if (curr_codes, curr_units, curr_zooms) != (xcodes, units, zooms):
                        self._results['fixed_hdr'][i] = True
                        new_header = hdr.copy()
                        new_header.set_qform(nii.affine, xcodes[0])
                        new_header.set_sform(nii.affine, xcodes[1])
                        new_header.set_xyzt_units(*units)
                        new_header.set_zooms(zooms)

                if data_dtype == 'source':  # match source dtype
                    try:
                        data_dtype = nb.load(self.inputs.source_file[0]).get_data_dtype()
                    except Exception:  # noqa: BLE001
                        LOGGER.warning(
                            'Could not get data type of file %s', self.inputs.source_file[0]
                        )
                        data_dtype = None

                if data_dtype:
                    data_dtype = np.dtype(data_dtype)
                    orig_dtype = nii.get_data_dtype()
                    if orig_dtype != data_dtype:
                        LOGGER.warning(
                            'Changing %s dtype from %s to %s',
                            Path(dest_file).name,
                            orig_dtype,
                            data_dtype,
                        )
                        # coerce dataobj to new data dtype
                        if np.issubdtype(data_dtype, np.integer):
                            new_data = np.rint(nii.dataobj).astype(data_dtype)
                        else:
                            new_data = np.asanyarray(nii.dataobj, dtype=data_dtype)
                        # and set header to match
                        if new_header is None:
                            new_header = nii.header.copy()
                        new_header.set_data_dtype(data_dtype)
                del nii

            if new_data is new_header is None and not new_compression:
                out_file = orig_file
            else:
                out_file = Path(runtime.cwd) / Path(dest_file).name

                orig_img = nb.load(orig_file)

                if new_header is None:
                    new_header = orig_img.header.copy()

                if new_data is None:
                    set_consumables(new_header, orig_img.dataobj)
                    new_data = orig_img.dataobj.get_unscaled()
                else:
                    # Without this, we would be writing nans
                    # This is our punishment for hacking around nibabel defaults
                    new_header.set_slope_inter(slope=1.0, inter=0.0)
                unsafe_write_nifti_header_and_data(
                    fname=out_file, header=new_header, data=new_data
                )
                del orig_img

            self._results['out_file'].append(str(out_file))

        return runtime


class _SaveDerivativeInputSpec(TraitedSpec):
    base_directory = Directory(
        exists=True, mandatory=True, desc='Path to the base directory for storing data.'
    )
    in_file = InputMultiObject(File(exists=True), mandatory=True, desc='the object to be saved')
    metadata = traits.Dict(Str, desc='metadata to be saved alongside the file')
    relative_path = InputMultiObject(
        traits.Str, desc='path to the file relative to the base directory'
    )


class _SaveDerivativeOutputSpec(TraitedSpec):
    out_file = OutputMultiObject(File, desc='written file path')
    out_meta = OutputMultiObject(File, desc='written JSON sidecar path')


class SaveDerivative(SimpleInterface):
    """Save a prepared derivative file.

    This interface is intended to be used after the PrepareDerivative interface.
    Its main purpose is to copy data to the output directory if an identical copy
    is not already present.

    This ensures that changes to the output directory metadata (e.g., mtime) do not
    trigger unnecessary recomputations in the workflow.
    """

    input_spec = _SaveDerivativeInputSpec
    output_spec = _SaveDerivativeOutputSpec
    _always_run = True

    def _run_interface(self, runtime):
        self._results['out_file'] = []
        self._results['out_meta'] = []

        for in_file, relative_path in zip(
            self.inputs.in_file,
            self.inputs.relative_path,
            strict=True,
        ):
            out_file = Path(self.inputs.base_directory) / relative_path
            out_file.parent.mkdir(exist_ok=True, parents=True)

            if not out_file.exists() or hash_infile(in_file) != hash_infile(out_file):
                _copy_any(in_file, out_file)

            if self.inputs.metadata:
                sidecar = out_file.parent / f'{out_file.name.split(".", 1)[0]}.json'
                sidecar.unlink(missing_ok=True)
                sidecar.write_text(dumps(self.inputs.metadata, indent=2))
                self._results['out_meta'].append(str(sidecar))
            self._results['out_file'].append(str(out_file))

        return runtime


class _DerivativesDataSinkInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    base_directory = traits.Directory(desc='Path to the base directory for storing data.')
    check_hdr = traits.Bool(True, usedefault=True, desc='fix headers of NIfTI outputs')
    compress = InputMultiObject(
        traits.Either(None, traits.Bool),
        usedefault=True,
        desc='whether ``in_file`` should be compressed (True), uncompressed (False) '
        'or left unmodified (None, default).',
    )
    data_dtype = Str(
        desc='NumPy datatype to coerce NIfTI data to, or `source` to match the input file dtype'
    )
    dismiss_entities = InputMultiObject(
        traits.Either(None, Str),
        usedefault=True,
        desc='a list entities that will not be propagated from the source file',
    )
    in_file = InputMultiObject(File(exists=True), mandatory=True, desc='the object to be saved')
    meta_dict = traits.Dict(Str, desc='an input dictionary containing metadata')
    source_file = InputMultiObject(
        File(exists=False), mandatory=True, desc='the source file(s) to extract entities from'
    )


class _DerivativesDataSinkOutputSpec(TraitedSpec):
    out_file = OutputMultiObject(File(exists=True, desc='written file path'))
    out_meta = OutputMultiObject(File(exists=True, desc='written JSON sidecar path'))
    compression = OutputMultiObject(
        traits.Either(None, traits.Bool),
        desc='whether ``in_file`` should be compressed (True), uncompressed (False) '
        'or left unmodified (None).',
    )
    fixed_hdr = traits.List(traits.Bool, desc='whether derivative header was fixed')


class DerivativesDataSink(SimpleInterface):
    """
    Store derivative files.

    Saves the ``in_file`` into a BIDS-Derivatives folder provided
    by ``base_directory``, given the input reference ``source_file``.

    .. testsetup::

        >>> data_dir_canary()

    >>> import tempfile
    >>> tmpdir = Path(tempfile.mkdtemp())
    >>> tmpfile = tmpdir / 'a_temp_file.nii.gz'
    >>> tmpfile.open('w').close()  # "touch" the file
    >>> t1w_source = bids_collect_data(
    ...     str(datadir / 'ds114'), '01', bids_validate=False)[0]['t1w'][0]
    >>> dsink = DerivativesDataSink(base_directory=str(tmpdir), check_hdr=False)
    >>> dsink.inputs.in_file = str(tmpfile)
    >>> dsink.inputs.source_file = t1w_source
    >>> dsink.inputs.desc = 'denoised'
    >>> dsink.inputs.compress = False
    >>> res = dsink.run()
    >>> res.outputs.out_file  # doctest: +ELLIPSIS
    '.../niworkflows/sub-01/ses-retest/anat/sub-01_ses-retest_desc-denoised_T1w.nii'

    >>> tmpfile = tmpdir / 'a_temp_file.nii'
    >>> tmpfile.open('w').close()  # "touch" the file
    >>> dsink = DerivativesDataSink(base_directory=str(tmpdir), check_hdr=False,
    ...                             allowed_entities=("custom",))
    >>> dsink.inputs.in_file = str(tmpfile)
    >>> dsink.inputs.source_file = t1w_source
    >>> dsink.inputs.custom = 'noise'
    >>> res = dsink.run()
    >>> res.outputs.out_file  # doctest: +ELLIPSIS
    '.../niworkflows/sub-01/ses-retest/anat/sub-01_ses-retest_custom-noise_T1w.nii'

    >>> dsink = DerivativesDataSink(base_directory=str(tmpdir), check_hdr=False,
    ...                             allowed_entities=("custom",))
    >>> dsink.inputs.in_file = [str(tmpfile), str(tmpfile)]
    >>> dsink.inputs.source_file = t1w_source
    >>> dsink.inputs.custom = [1, 2]
    >>> dsink.inputs.compress = True
    >>> res = dsink.run()
    >>> res.outputs.out_file  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['.../niworkflows/sub-01/ses-retest/anat/sub-01_ses-retest_custom-1_T1w.nii.gz',
     '.../niworkflows/sub-01/ses-retest/anat/sub-01_ses-retest_custom-2_T1w.nii.gz']

    >>> dsink = DerivativesDataSink(base_directory=str(tmpdir), check_hdr=False,
    ...                             allowed_entities=("custom1", "custom2"))
    >>> dsink.inputs.in_file = [str(tmpfile)] * 2
    >>> dsink.inputs.source_file = t1w_source
    >>> dsink.inputs.custom1 = [1, 2]
    >>> dsink.inputs.custom2 = "b"
    >>> res = dsink.run()
    >>> res.outputs.out_file  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['.../niworkflows/sub-01/ses-retest/anat/sub-01_ses-retest_custom1-1_custom2-b_T1w.nii',
     '.../niworkflows/sub-01/ses-retest/anat/sub-01_ses-retest_custom1-2_custom2-b_T1w.nii']

    When multiple source files are passed, only common entities are passed down.
    For example, if two T1w images from different sessions are used to generate
    a single image, the session entity is removed automatically.

    >>> bids_dir = tmpdir / 'bidsroot'
    >>> multi_source = [
    ...     bids_dir / 'sub-02/ses-A/anat/sub-02_ses-A_T1w.nii.gz',
    ...     bids_dir / 'sub-02/ses-B/anat/sub-02_ses-B_T1w.nii.gz']
    >>> for source_file in multi_source:
    ...     source_file.parent.mkdir(parents=True, exist_ok=True)
    ...     _ = source_file.write_text("")
    >>> dsink = DerivativesDataSink(base_directory=str(tmpdir), check_hdr=False)
    >>> dsink.inputs.in_file = str(tmpfile)
    >>> dsink.inputs.source_file = list(map(str, multi_source))
    >>> dsink.inputs.desc = 'preproc'
    >>> res = dsink.run()
    >>> res.outputs.out_file  # doctest: +ELLIPSIS
    '.../niworkflows/sub-02/anat/sub-02_desc-preproc_T1w.nii'

    If, on the other hand, only one is used, the session is preserved:

    >>> dsink.inputs.source_file = str(multi_source[0])
    >>> res = dsink.run()
    >>> res.outputs.out_file  # doctest: +ELLIPSIS
    '.../niworkflows/sub-02/ses-A/anat/sub-02_ses-A_desc-preproc_T1w.nii'

    >>> bids_dir = tmpdir / 'bidsroot' / 'sub-02' / 'ses-noanat' / 'func'
    >>> bids_dir.mkdir(parents=True, exist_ok=True)
    >>> tricky_source = bids_dir / 'sub-02_ses-noanat_task-rest_run-01_bold.nii.gz'
    >>> tricky_source.open('w').close()
    >>> dsink = DerivativesDataSink(base_directory=str(tmpdir), check_hdr=False)
    >>> dsink.inputs.in_file = str(tmpfile)
    >>> dsink.inputs.source_file = str(tricky_source)
    >>> dsink.inputs.desc = 'preproc'
    >>> res = dsink.run()
    >>> res.outputs.out_file  # doctest: +ELLIPSIS
    '.../niworkflows/sub-02/ses-noanat/func/sub-02_ses-noanat_task-rest_run-01_\
desc-preproc_bold.nii'

    >>> bids_dir = tmpdir / 'bidsroot' / 'sub-02' / 'ses-noanat' / 'func'
    >>> bids_dir.mkdir(parents=True, exist_ok=True)
    >>> tricky_source = bids_dir / 'sub-02_ses-noanat_task-rest_run-01_bold.nii.gz'
    >>> tricky_source.open('w').close()
    >>> dsink = DerivativesDataSink(base_directory=str(tmpdir), check_hdr=False)
    >>> dsink.inputs.in_file = str(tmpfile)
    >>> dsink.inputs.source_file = str(tricky_source)
    >>> dsink.inputs.desc = 'preproc'
    >>> dsink.inputs.RepetitionTime = 0.75
    >>> res = dsink.run()
    >>> res.outputs.out_meta  # doctest: +ELLIPSIS
    '.../niworkflows/sub-02/ses-noanat/func/sub-02_ses-noanat_task-rest_run-01_\
desc-preproc_bold.json'

    >>> Path(res.outputs.out_meta).read_text().splitlines()[1]
    '  "RepetitionTime": 0.75'

    >>> bids_dir = tmpdir / 'bidsroot' / 'sub-02' / 'ses-noanat' / 'func'
    >>> bids_dir.mkdir(parents=True, exist_ok=True)
    >>> tricky_source = bids_dir / 'sub-02_ses-noanat_task-rest_run-01_bold.nii.gz'
    >>> tricky_source.open('w').close()
    >>> dsink = DerivativesDataSink(base_directory=str(tmpdir), check_hdr=False,
    ...                             SkullStripped=True)
    >>> dsink.inputs.in_file = str(tmpfile)
    >>> dsink.inputs.source_file = str(tricky_source)
    >>> dsink.inputs.desc = 'preproc'
    >>> dsink.inputs.space = 'MNI152NLin6Asym'
    >>> dsink.inputs.resolution = '01'
    >>> dsink.inputs.RepetitionTime = 0.75
    >>> res = dsink.run()
    >>> res.outputs.out_meta  # doctest: +ELLIPSIS
    '.../niworkflows/sub-02/ses-noanat/func/sub-02_ses-noanat_task-rest_run-01_\
space-MNI152NLin6Asym_res-01_desc-preproc_bold.json'

    >>> lines = Path(res.outputs.out_meta).read_text().splitlines()
    >>> lines[1]
    '  "RepetitionTime": 0.75,'

    >>> lines[2]  # doctest: +ELLIPSIS
    '  "Resolution": "Template MNI152NLin6Asym (1.0x1.0x1.0 mm^3)...'

    >>> lines[3]
    '  "SkullStripped": true'

    >>> bids_dir = tmpdir / 'bidsroot' / 'sub-02' / 'ses-noanat' / 'func'
    >>> bids_dir.mkdir(parents=True, exist_ok=True)
    >>> tricky_source = bids_dir / 'sub-02_ses-noanat_task-rest_run-01_bold.nii.gz'
    >>> tricky_source.open('w').close()
    >>> dsink = DerivativesDataSink(base_directory=str(tmpdir), check_hdr=False,
    ...                             SkullStripped=True)
    >>> dsink.inputs.in_file = str(tmpfile)
    >>> dsink.inputs.source_file = str(tricky_source)
    >>> dsink.inputs.desc = 'preproc'
    >>> dsink.inputs.resolution = 'native'
    >>> dsink.inputs.space = 'MNI152NLin6Asym'
    >>> dsink.inputs.RepetitionTime = 0.75
    >>> dsink.inputs.meta_dict = {'RepetitionTime': 1.75, 'SkullStripped': False, 'Z': 'val'}
    >>> res = dsink.run()
    >>> res.outputs.out_meta  # doctest: +ELLIPSIS
    '.../niworkflows/sub-02/ses-noanat/func/sub-02_ses-noanat_task-rest_run-01_\
space-MNI152NLin6Asym_desc-preproc_bold.json'

    >>> lines = Path(res.outputs.out_meta).read_text().splitlines()
    >>> lines[1]
    '  "RepetitionTime": 0.75,'

    >>> lines[2]
    '  "SkullStripped": true,'

    >>> lines[3]
    '  "Z": "val"'

    """

    input_spec = _DerivativesDataSinkInputSpec
    output_spec = _DerivativesDataSinkOutputSpec
    out_path_base = 'niworkflows'
    _always_run = True
    _config_entities = frozenset({e['name'] for e in BIDS_DERIV_ENTITIES})
    _config_entities_dict = BIDS_DERIV_ENTITIES
    _standard_spaces = STANDARD_SPACES
    _file_patterns = BIDS_DERIV_PATTERNS
    _default_dtypes = DEFAULT_DTYPES

    def __init__(self, allowed_entities=None, out_path_base=None, **inputs):
        """Initialize the SimpleInterface and extend inputs with custom entities."""
        self._allowed_entities = set(allowed_entities or []).union(set(self._config_entities))
        if out_path_base:
            self.out_path_base = out_path_base

        self._metadata = {}
        self._static_traits = self.input_spec.class_editable_traits() + sorted(
            self._allowed_entities
        )
        for dynamic_input in set(inputs) - set(self._static_traits):
            self._metadata[dynamic_input] = inputs.pop(dynamic_input)

        # First regular initialization (constructs InputSpec object)
        super().__init__(**inputs)
        add_traits(self.inputs, self._allowed_entities)
        for k in self._allowed_entities.intersection(list(inputs.keys())):
            # Add additional input fields (self.inputs is an object)
            setattr(self.inputs, k, inputs[k])

    def _run_interface(self, runtime):
        from bids.layout import Config, parse_file_entities
        from bids.layout.writing import build_path
        from bids.utils import listify

        # Ready the output folder
        base_directory = runtime.cwd
        if isdefined(self.inputs.base_directory):
            base_directory = self.inputs.base_directory
        base_directory = Path(base_directory).absolute()
        out_path = base_directory / self.out_path_base
        out_path.mkdir(exist_ok=True, parents=True)

        # Ensure we have a list
        in_file = listify(self.inputs.in_file)

        # Read in the dictionary of metadata
        if isdefined(self.inputs.meta_dict):
            meta = self.inputs.meta_dict
            # inputs passed in construction take priority
            meta.update(self._metadata)
            self._metadata = meta

        # Initialize entities with those from the source file.
        custom_config = Config(
            name='custom',
            entities=self._config_entities_dict,
            default_path_patterns=self._file_patterns,
        )
        in_entities = [
            parse_file_entities(
                str(relative_to_root(source_file)),
                config=['bids', 'derivatives', custom_config],
            )
            for source_file in self.inputs.source_file
        ]
        out_entities = {
            k: v
            for k, v in in_entities[0].items()
            if all(ent.get(k) == v for ent in in_entities[1:])
        }
        for drop_entity in listify(self.inputs.dismiss_entities or []):
            out_entities.pop(drop_entity, None)

        # Override extension with that of the input file(s)
        out_entities['extension'] = [
            # _splitext does not accept .surf.gii (for instance)
            ''.join(Path(orig_file).suffixes).lstrip('.')
            for orig_file in in_file
        ]

        compress = listify(self.inputs.compress) or [None]
        if len(compress) == 1:
            compress = compress * len(in_file)
        for i, ext in enumerate(out_entities['extension']):
            if compress[i] is not None:
                ext = regz.sub('', ext)
                out_entities['extension'][i] = f'{ext}.gz' if compress[i] else ext

        # Override entities with those set as inputs
        for key in self._allowed_entities:
            value = getattr(self.inputs, key)
            if value is not None and isdefined(value):
                out_entities[key] = value

        # Clean up native resolution with space
        if out_entities.get('resolution') == 'native' and out_entities.get('space'):
            out_entities.pop('resolution', None)

        # Expand templateflow resolutions
        resolution = out_entities.get('resolution')
        space = out_entities.get('space')
        if resolution:
            # Standard spaces
            if space in self._standard_spaces:
                res = _get_tf_resolution(space, resolution)
            else:  # TODO: Nonstandard?
                res = 'Unknown'
            self._metadata['Resolution'] = res

        if len(set(out_entities['extension'])) == 1:
            out_entities['extension'] = out_entities['extension'][0]

        # Insert custom (non-BIDS) entities from allowed_entities.
        custom_entities = set(out_entities) - set(self._config_entities)
        patterns = self._file_patterns
        if custom_entities:
            # Example: f"{key}-{{{key}}}" -> "task-{task}"
            custom_pat = '_'.join(f'{key}-{{{key}}}' for key in sorted(custom_entities))
            patterns = [pat.replace('_{suffix', f'_{custom_pat}_{{suffix') for pat in patterns]

        # Prepare SimpleInterface outputs object
        self._results['out_file'] = []
        self._results['compression'] = []
        self._results['fixed_hdr'] = [False] * len(in_file)

        dest_files = build_path(out_entities, path_patterns=patterns)
        if not dest_files:
            raise ValueError(f'Could not build path with entities {out_entities}.')

        # Make sure the interpolated values is embedded in a list, and check
        dest_files = listify(dest_files)
        if len(in_file) != len(dest_files):
            raise ValueError(
                f'Input files ({len(in_file)}) not matched '
                f'by interpolated patterns ({len(dest_files)}).'
            )

        for i, (orig_file, dest_file) in enumerate(zip(in_file, dest_files)):
            out_file = out_path / dest_file
            out_file.parent.mkdir(exist_ok=True, parents=True)
            self._results['out_file'].append(str(out_file))
            self._results['compression'].append(str(dest_file).endswith('.gz'))

            # An odd but possible case is that an input file is in the location of
            # the output and we have made no changes to it.
            # The primary use case is pre-computed derivatives where the output
            # directory will be filled in.
            # From a provenance perspective, I would rather inputs and outputs be
            # cleanly separated, but that is better handled by warnings at the CLI
            # level than a crash in a datasink.
            try:
                if os.path.samefile(orig_file, out_file):
                    continue
            except FileNotFoundError:
                pass

            # Set data and header iff changes need to be made. If these are
            # still None when it's time to write, just copy.
            new_data, new_header = None, None

            is_nifti = False
            with suppress(nb.filebasedimages.ImageFileError):
                is_nifti = isinstance(nb.load(orig_file), nb.Nifti1Image)

            data_dtype = self.inputs.data_dtype or self._default_dtypes[self.inputs.suffix]
            if is_nifti and any((self.inputs.check_hdr, data_dtype)):
                nii = nb.load(orig_file)

                if self.inputs.check_hdr:
                    hdr = nii.header
                    curr_units = tuple(None if u == 'unknown' else u for u in hdr.get_xyzt_units())
                    curr_codes = (int(hdr['qform_code']), int(hdr['sform_code']))

                    # Default to mm, use sec if data type is bold
                    units = (
                        curr_units[0] or 'mm',
                        'sec' if out_entities['suffix'] == 'bold' else None,
                    )
                    xcodes = (1, 1)  # Derivative in its original scanner space
                    if self.inputs.space:
                        xcodes = (4, 4) if self.inputs.space in self._standard_spaces else (2, 2)

                    curr_zooms = zooms = hdr.get_zooms()
                    if 'RepetitionTime' in self.inputs.get():
                        zooms = curr_zooms[:3] + (self.inputs.RepetitionTime,)

                    if (curr_codes, curr_units, curr_zooms) != (xcodes, units, zooms):
                        self._results['fixed_hdr'][i] = True
                        new_header = hdr.copy()
                        new_header.set_qform(nii.affine, xcodes[0])
                        new_header.set_sform(nii.affine, xcodes[1])
                        new_header.set_xyzt_units(*units)
                        new_header.set_zooms(zooms)

                if data_dtype == 'source':  # match source dtype
                    try:
                        data_dtype = nb.load(self.inputs.source_file[0]).get_data_dtype()
                    except Exception:  # noqa: BLE001
                        LOGGER.warning(
                            'Could not get data type of file %s', self.inputs.source_file[0]
                        )
                        data_dtype = None

                if data_dtype:
                    data_dtype = np.dtype(data_dtype)
                    orig_dtype = nii.get_data_dtype()
                    if orig_dtype != data_dtype:
                        LOGGER.warning(
                            'Changing %s dtype from %s to %s', out_file, orig_dtype, orig_dtype
                        )
                        # coerce dataobj to new data dtype
                        if np.issubdtype(data_dtype, np.integer):
                            new_data = np.rint(nii.dataobj).astype(data_dtype)
                        else:
                            new_data = np.asanyarray(nii.dataobj, dtype=data_dtype)
                        # and set header to match
                        if new_header is None:
                            new_header = nii.header.copy()
                        new_header.set_data_dtype(data_dtype)
                del nii

            unlink(out_file, missing_ok=True)
            if new_data is new_header is None:
                _copy_any(orig_file, str(out_file))
            else:
                orig_img = nb.load(orig_file)
                if new_data is None:
                    set_consumables(new_header, orig_img.dataobj)
                    new_data = orig_img.dataobj.get_unscaled()
                else:
                    # Without this, we would be writing nans
                    # This is our punishment for hacking around nibabel defaults
                    new_header.set_slope_inter(slope=1.0, inter=0.0)
                unsafe_write_nifti_header_and_data(
                    fname=out_file, header=new_header, data=new_data
                )
                del orig_img

        if len(self._results['out_file']) == 1:
            meta_fields = self.inputs.copyable_trait_names()
            self._metadata.update(
                {k: getattr(self.inputs, k) for k in meta_fields if k not in self._static_traits}
            )
            if self._metadata:
                sidecar = out_file.parent / f'{out_file.name.split(".", 1)[0]}.json'
                unlink(sidecar, missing_ok=True)
                sidecar.write_text(dumps(self._metadata, sort_keys=True, indent=2))
                self._results['out_meta'] = str(sidecar)
        return runtime


class _ReadSidecarJSONInputSpec(_BIDSBaseInputSpec):
    in_file = File(exists=True, mandatory=True, desc='the input nifti file')


class _ReadSidecarJSONOutputSpec(_BIDSInfoOutputSpec):
    out_dict = traits.Dict()


class ReadSidecarJSON(SimpleInterface):
    """
    Read JSON sidecar files of a BIDS tree.

    .. testsetup::

        >>> data_dir_canary()

    >>> fmap = str(datadir / 'ds054' / 'sub-100185' / 'fmap' /
    ...            'sub-100185_phasediff.nii.gz')

    >>> meta = ReadSidecarJSON(in_file=fmap, bids_dir=str(datadir / 'ds054'),
    ...                        bids_validate=False).run()
    >>> meta.outputs.subject
    '100185'
    >>> meta.outputs.suffix
    'phasediff'
    >>> meta.outputs.out_dict['Manufacturer']
    'SIEMENS'
    >>> meta = ReadSidecarJSON(in_file=fmap, fields=['Manufacturer'],
    ...                        bids_dir=str(datadir / 'ds054'),
    ...                        bids_validate=False).run()
    >>> meta.outputs.out_dict['Manufacturer']
    'SIEMENS'
    >>> meta.outputs.Manufacturer
    'SIEMENS'
    >>> meta.outputs.OtherField  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    AttributeError:
    >>> meta = ReadSidecarJSON(
    ...     in_file=fmap, fields=['MadeUpField'],
    ...     bids_dir=str(datadir / 'ds054'),
    ...     bids_validate=False).run()  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    KeyError:
    >>> meta = ReadSidecarJSON(in_file=fmap, fields=['MadeUpField'],
    ...                        undef_fields=True,
    ...                        bids_dir=str(datadir / 'ds054'),
    ...                        bids_validate=False).run()
    >>> meta.outputs.MadeUpField
    <undefined>

    """

    input_spec = _ReadSidecarJSONInputSpec
    output_spec = _ReadSidecarJSONOutputSpec
    layout = None
    _always_run = True

    def __init__(self, fields=None, undef_fields=False, **inputs):
        from bids.utils import listify

        super().__init__(**inputs)
        self._fields = listify(fields or [])
        self._undef_fields = undef_fields

    def _outputs(self):
        base = super()._outputs()
        if self._fields:
            base = add_traits(base, self._fields)
        return base

    def _run_interface(self, runtime):
        self.layout = self.inputs.bids_dir or self.layout
        self.layout = _init_layout(
            self.inputs.in_file,
            self.layout,
            self.inputs.bids_validate,
            database_path=(self.inputs.index_db if isdefined(self.inputs.index_db) else None),
        )

        # Fill in BIDS entities of the output ("*_id")
        output_keys = list(_BIDSInfoOutputSpec().get().keys())
        params = self.layout.parse_file_entities(self.inputs.in_file)
        self._results = {key: params.get(key.split('_')[0], Undefined) for key in output_keys}

        # Fill in metadata
        metadata = self.layout.get_metadata(self.inputs.in_file)
        self._results['out_dict'] = metadata

        # Set dynamic outputs if fields input is present
        for fname in self._fields:
            if not self._undef_fields and fname not in metadata:
                raise KeyError(
                    f'Metadata field "{fname}" not found for file {self.inputs.in_file}'
                )
            self._results[fname] = metadata.get(fname, Undefined)
        return runtime


class _BIDSFreeSurferDirInputSpec(BaseInterfaceInputSpec):
    derivatives = Directory(exists=True, mandatory=True, desc='BIDS derivatives directory')
    freesurfer_home = Directory(
        exists=True, mandatory=True, desc='FreeSurfer installation directory'
    )
    subjects_dir = traits.Either(
        traits.Str(),
        Directory(),
        default='freesurfer',
        usedefault=True,
        desc='Name of FreeSurfer subjects directory',
    )
    spaces = traits.List(traits.Str, desc='Set of output spaces to prepare')
    overwrite_fsaverage = traits.Bool(
        False, usedefault=True, desc='Overwrite fsaverage directories, if present'
    )
    minimum_fs_version = traits.Enum('7.0.0', desc='Minimum FreeSurfer version for compatibility')


class _BIDSFreeSurferDirOutputSpec(TraitedSpec):
    subjects_dir = traits.Directory(exists=True, desc='FreeSurfer subjects directory')


class BIDSFreeSurferDir(SimpleInterface):
    """
    Prepare a FreeSurfer subjects directory for use in a BIDS context.

    Constructs a subjects directory path, creating if necessary, and copies
    fsaverage subjects (if necessary or forced via ``overwrite_fsaverage``)
    into from the local FreeSurfer distribution.

    If ``subjects_dir`` is an absolute path, then it is returned as the output
    ``subjects_dir``.
    If it is a relative path, it will be resolved relative to the
    ```derivatives`` directory.`

    Regardless of the path, if ``fsaverage`` spaces are provided, they will be
    verified to exist, or copied from ``$FREESURFER_HOME/subjects``, if missing.

    The output ``subjects_dir`` is intended to be passed to ``ReconAll`` and
    other FreeSurfer interfaces.

    """

    input_spec = _BIDSFreeSurferDirInputSpec
    output_spec = _BIDSFreeSurferDirOutputSpec
    _always_run = True

    def _run_interface(self, runtime):
        subjects_dir = Path(self.inputs.subjects_dir)
        if not subjects_dir.is_absolute():
            subjects_dir = Path(self.inputs.derivatives) / subjects_dir
        subjects_dir.mkdir(parents=True, exist_ok=True)
        self._results['subjects_dir'] = str(subjects_dir)

        orig_subjects_dir = Path(self.inputs.freesurfer_home) / 'subjects'

        # Source is target, so just quit
        if subjects_dir == orig_subjects_dir:
            return runtime

        spaces = list(self.inputs.spaces)
        # Always copy fsaverage, for proper recon-all functionality
        if 'fsaverage' not in spaces:
            spaces.append('fsaverage')

        for space in spaces:
            # Skip non-freesurfer spaces and fsnative
            if not space.startswith('fsaverage'):
                continue
            source = orig_subjects_dir / space
            dest = subjects_dir / space

            # Edge case, but give a sensible error
            if not source.exists():
                if dest.exists():
                    continue
                else:
                    raise FileNotFoundError(f"Expected to find '{source}' to copy")

            if (
                space == 'fsaverage'
                and dest.exists()
                and self.inputs.minimum_fs_version == '7.0.0'
            ):
                label = dest / 'label' / 'rh.FG1.mpm.vpnl.label'  # new in FS7
                if not label.exists():
                    # remove previous output and let us recopy
                    shutil.rmtree(dest)

            # Finesse is overrated. Either leave it alone or completely clobber it.
            if dest.exists() and self.inputs.overwrite_fsaverage:
                shutil.rmtree(dest)
            if not dest.exists():
                try:
                    # Use copy instead of copy2; copy calls copymode while copy2 calls
                    # copystat, which will preserve atime/mtime.
                    # atime should *not* be copied to avoid triggering processes that
                    # sweep un-accessed files.
                    # If we want to preserve mtime, that will require a new copy function.
                    shutil.copytree(source, dest, copy_function=shutil.copy)
                except FileExistsError:
                    LOGGER.warning(
                        '%s exists; if multiple jobs are running in parallel'
                        ', this can be safely ignored',
                        dest,
                    )

        return runtime


def _get_tf_resolution(space: str, resolution: str) -> str:
    """
    Query templateflow template information to elaborate on template resolution.

    Examples
    --------
    >>> _get_tf_resolution('MNI152NLin2009cAsym', '01') # doctest: +ELLIPSIS
    'Template MNI152NLin2009cAsym (1.0x1.0x1.0 mm^3)...'
    >>> _get_tf_resolution('MNI152NLin2009cAsym', '1') # doctest: +ELLIPSIS
    'Template MNI152NLin2009cAsym (1.0x1.0x1.0 mm^3)...'
    >>> _get_tf_resolution('MNI152NLin2009cAsym', '10')
    'Unknown'
    """
    metadata = tf.api.get_metadata(space)
    resolutions = metadata.get('res', {})
    res_meta = None

    # Due to inconsistencies, resolution keys may or may not be zero-padded
    padded_res = f'{resolution:0>2}'
    for r in (resolution, padded_res):
        if r in resolutions:
            res_meta = resolutions[r]
    if res_meta is None:
        return 'Unknown'

    def _fmt_xyz(coords: list) -> str:
        xyz = 'x'.join([str(c) for c in coords])
        return f'{xyz} mm^3'

    return (
        f'Template {space} ({_fmt_xyz(res_meta["zooms"])}),'
        f' curated by TemplateFlow {tf.__version__}'
    )
