# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interfaces for handling BIDS-like neuroimaging structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Fetch some example data:
    >>> import os
    >>> from niworkflows import data
    >>> data_root = data.get_bids_examples(variant='BIDS-examples-1-enh-ds054')
    >>> os.chdir(data_root)
Disable warnings:
    >>> from nipype import logging
    >>> logging.getLogger('nipype.interface').setLevel('ERROR')
"""

from pathlib import Path
import os
import re
import gzip
from shutil import copyfileobj

from nipype import logging
from nipype.interfaces.base import (
    isdefined, TraitedSpec, BaseInterfaceInputSpec,
    SimpleInterface, traits)
from nipype.interfaces.base.traits_extension import (
    File, Directory, InputMultiPath, OutputMultiPath, Str
)
from nipype.utils.filemanip import copyfile

LOGGER = logging.getLogger('nipype.interface')
BIDS_NAME = re.compile(
    r'^(.*\/)?'
    '(?P<subject_id>(sub|tpl)-[a-zA-Z0-9]+)'
    '(_(?P<session_id>ses-[a-zA-Z0-9]+))?'
    '(_(?P<task_id>task-[a-zA-Z0-9]+))?'
    '(_(?P<acq_id>acq-[a-zA-Z0-9]+))?'
    '(_(?P<rec_id>rec-[a-zA-Z0-9]+))?'
    '(_(?P<run_id>run-[a-zA-Z0-9]+))?')


class DerivativesDataSinkInputSpec(BaseInterfaceInputSpec):
    base_directory = Directory(
        desc='Path to the base directory for storing data.')
    in_file = InputMultiPath(File(exists=True), mandatory=True,
                             desc='the object to be saved')
    source_file = File(exists=False, mandatory=True, desc='the input func file')
    space = Str('', usedefault=True, desc='Label for space field')
    desc = Str('', usedefault=True, desc='Label for description field')
    suffix = Str('', usedefault=True, desc='suffix appended to source_file')
    keep_dtype = traits.Bool(False, usedefault=True, desc='keep datatype suffix')
    extra_values = traits.List(Str)
    compress = traits.Bool(desc="force compression (True) or uncompression (False)"
                                " of the output file (default: same as input)")


class DerivativesDataSinkOutputSpec(TraitedSpec):
    out_file = OutputMultiPath(File(exists=True, desc='written file path'))
    compression = OutputMultiPath(
        traits.Bool, desc='whether ``in_file`` was compressed/uncompressed '
                          'or `it was copied directly.')


class DerivativesDataSink(SimpleInterface):
    """
    Saves the `in_file` into a BIDS-Derivatives folder provided
    by `base_directory`, given the input reference `source_file`.
    >>> from pathlib import Path
    >>> import tempfile
    >>> from fmriprep.utils.bids import collect_data
    >>> tmpdir = Path(tempfile.mkdtemp())
    >>> tmpfile = tmpdir / 'a_temp_file.nii.gz'
    >>> tmpfile.open('w').close()  # "touch" the file
    >>> dsink = DerivativesDataSink(base_directory=str(tmpdir))
    >>> dsink.inputs.in_file = str(tmpfile)
    >>> dsink.inputs.source_file = collect_data('ds114', '01')[0]['t1w'][0]
    >>> dsink.inputs.keep_dtype = True
    >>> dsink.inputs.suffix = 'target-mni'
    >>> res = dsink.run()
    >>> res.outputs.out_file  # doctest: +ELLIPSIS
    '.../fmriprep/sub-01/ses-retest/anat/sub-01_ses-retest_target-mni_T1w.nii.gz'
    >>> bids_dir = tmpdir / 'bidsroot' / 'sub-02' / 'ses-noanat' / 'func'
    >>> bids_dir.mkdir(parents=True, exist_ok=True)
    >>> tricky_source = bids_dir / 'sub-02_ses-noanat_task-rest_run-01_bold.nii.gz'
    >>> tricky_source.open('w').close()
    >>> dsink = DerivativesDataSink(base_directory=str(tmpdir))
    >>> dsink.inputs.in_file = str(tmpfile)
    >>> dsink.inputs.source_file = str(tricky_source)
    >>> dsink.inputs.keep_dtype = True
    >>> dsink.inputs.desc = 'preproc'
    >>> res = dsink.run()
    >>> res.outputs.out_file  # doctest: +ELLIPSIS
    '.../fmriprep/sub-02/ses-noanat/func/sub-02_ses-noanat_task-rest_run-01_\
desc-preproc_bold.nii.gz'
    """
    input_spec = DerivativesDataSinkInputSpec
    output_spec = DerivativesDataSinkOutputSpec
    out_path_base = "fmriprep"
    _always_run = True

    def __init__(self, out_path_base=None, **inputs):
        super(DerivativesDataSink, self).__init__(**inputs)
        self._results['out_file'] = []
        if out_path_base:
            self.out_path_base = out_path_base

    def _run_interface(self, runtime):
        src_fname, _ = _splitext(self.inputs.source_file)
        src_fname, dtype = src_fname.rsplit('_', 1)
        _, ext = _splitext(self.inputs.in_file[0])
        if self.inputs.compress is True and not ext.endswith('.gz'):
            ext += '.gz'
        elif self.inputs.compress is False and ext.endswith('.gz'):
            ext = ext[:-3]

        m = BIDS_NAME.search(src_fname)

        mod = Path(self.inputs.source_file).parent.name

        base_directory = runtime.cwd
        if isdefined(self.inputs.base_directory):
            base_directory = self.inputs.base_directory

        base_directory = Path(base_directory).resolve()
        out_path = base_directory / self.out_path_base / \
            '{subject_id}'.format(**m.groupdict())

        if m.groupdict().get('session_id') is not None:
            out_path = out_path / '{session_id}'.format(**m.groupdict())

        out_path = out_path / '{}'.format(mod)
        out_path.mkdir(exist_ok=True, parents=True)
        base_fname = str(out_path / src_fname)

        formatstr = '{bname}{space}{desc}{extra}{suffix}{dtype}{ext}'
        if len(self.inputs.in_file) > 1 and not isdefined(self.inputs.extra_values):
            formatstr = '{bname}{space}{desc}{suffix}{i:04d}{dtype}{ext}'

        space = '_space-{}'.format(self.inputs.space) if self.inputs.space else ''
        desc = '_desc-{}'.format(self.inputs.desc) if self.inputs.desc else ''
        suffix = '_{}'.format(self.inputs.suffix) if self.inputs.suffix else ''
        dtype = '' if not self.inputs.keep_dtype else ('_%s' % dtype)

        self._results['compression'] = []
        for i, fname in enumerate(self.inputs.in_file):
            extra = ''
            if isdefined(self.inputs.extra_values):
                extra = '_{}'.format(self.inputs.extra_values[i])
            out_file = formatstr.format(
                bname=base_fname,
                space=space,
                desc=desc,
                extra=extra,
                suffix=suffix,
                i=i,
                dtype=dtype,
                ext=ext,
            )
            self._results['out_file'].append(out_file)
            self._results['compression'].append(_copy_any(fname, out_file))
        return runtime


def _splitext(fname):
    basename = Path(fname).name
    ext = Path(basename).suffix if not basename.endswith('.gz') else \
        ''.join(Path(basename).suffixes[-2:])
    return basename[: -len(ext)], ext


def _copy_any(src, dst):
    src_isgz = src.endswith('.gz')
    dst_isgz = dst.endswith('.gz')
    if src_isgz == dst_isgz:
        copyfile(src, dst, copy=True, use_hardlink=True)
        return False  # Make sure we do not reuse the hardlink later

    # Unlink target (should not exist)
    if os.path.exists(dst):
        os.unlink(dst)

    src_open = gzip.open if src_isgz else open
    dst_open = gzip.open if dst_isgz else open
    with src_open(src, 'rb') as f_in:
        with dst_open(dst, 'wb') as f_out:
            copyfileobj(f_in, f_out)
    return True
