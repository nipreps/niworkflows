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
"""ITK files handling."""

import os
from mimetypes import guess_type
from tempfile import TemporaryDirectory

import nibabel as nb
import nitransforms as nt
import numpy as np
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    OutputMultiObject,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix

from .fixes import _FixTraitApplyTransformsInputSpec

LOGGER = logging.getLogger('nipype.interface')


class _MCFLIRT2ITKInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiObject(
        File(exists=True), mandatory=True, desc='list of MAT files from MCFLIRT'
    )
    in_reference = File(exists=True, mandatory=True, desc='input image for spatial reference')
    in_source = File(exists=True, mandatory=True, desc='input image for spatial source')
    num_threads = traits.Int(nohash=True, desc='number of parallel processes')


class _MCFLIRT2ITKOutputSpec(TraitedSpec):
    out_file = File(desc='the output ITKTransform file')


class MCFLIRT2ITK(SimpleInterface):
    """Convert a list of MAT files from MCFLIRT into an ITK Transform file."""

    input_spec = _MCFLIRT2ITKInputSpec
    output_spec = _MCFLIRT2ITKOutputSpec

    def _run_interface(self, runtime):
        if isdefined(self.inputs.num_threads):
            LOGGER.warning('Multithreading is deprecated. Remove the num_threads input.')

        source = nb.load(self.inputs.in_source)
        reference = nb.load(self.inputs.in_reference)
        affines = [
            nt.linear.load(mat, fmt='fsl', reference=reference, moving=source)
            for mat in self.inputs.in_files
        ]

        affarray = nt.io.itk.ITKLinearTransformArray.from_ras(
            np.stack([a.matrix for a in affines], axis=0),
        )

        self._results['out_file'] = os.path.join(runtime.cwd, 'mat2itk.txt')
        affarray.to_filename(self._results['out_file'])

        return runtime


class _MultiApplyTransformsInputSpec(_FixTraitApplyTransformsInputSpec):
    input_image = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc='input time-series as a list of volumes after splitting through the fourth dimension',
    )
    num_threads = traits.Int(1, usedefault=True, nohash=True, desc='number of parallel processes')
    save_cmd = traits.Bool(
        True, usedefault=True, desc='write a log of command lines that were applied'
    )
    copy_dtype = traits.Bool(False, usedefault=True, desc='copy dtype from inputs to outputs')


class _MultiApplyTransformsOutputSpec(TraitedSpec):
    out_files = OutputMultiObject(File(), desc='the output ITKTransform file')
    log_cmdline = File(desc='a list of command lines used to apply transforms')


class MultiApplyTransforms(SimpleInterface):
    """Apply the corresponding list of input transforms."""

    input_spec = _MultiApplyTransformsInputSpec
    output_spec = _MultiApplyTransformsOutputSpec

    def _run_interface(self, runtime):
        # Get all inputs from the ApplyTransforms object
        ifargs = self.inputs.get()

        # Extract number of input images and transforms
        in_files = ifargs.pop('input_image')
        num_files = len(in_files)
        transforms = ifargs.pop('transforms')
        # Get number of parallel jobs
        num_threads = ifargs.pop('num_threads')
        save_cmd = ifargs.pop('save_cmd')

        # Remove certain keys
        for key in ['environ', 'ignore_exception', 'terminal_output', 'output_image']:
            ifargs.pop(key, None)

        # Get a temp folder ready
        tmp_folder = TemporaryDirectory(prefix='tmp-', dir=runtime.cwd)

        xfms_list = _arrange_xfms(transforms, num_files, tmp_folder)
        if len(xfms_list) != num_files:
            raise ValueError('Number of files and entries in the transforms list do not match')

        # Inputs are ready to run in parallel
        if num_threads < 1:
            num_threads = None

        if num_threads == 1:
            out_files = [
                _applytfms((in_file, in_xfm, ifargs, i, runtime.cwd))
                for i, (in_file, in_xfm) in enumerate(zip(in_files, xfms_list))
            ]
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=num_threads) as pool:
                out_files = list(
                    pool.map(
                        _applytfms,
                        [
                            (in_file, in_xfm, ifargs, i, runtime.cwd)
                            for i, (in_file, in_xfm) in enumerate(zip(in_files, xfms_list))
                        ],
                    )
                )
        tmp_folder.cleanup()

        # Collect output file names, after sorting by index
        self._results['out_files'] = [el[0] for el in out_files]

        if save_cmd:
            self._results['log_cmdline'] = os.path.join(runtime.cwd, 'command.txt')
            with open(self._results['log_cmdline'], 'w') as cmdfile:
                print('\n-------\n'.join([el[1] for el in out_files]), file=cmdfile)
        return runtime


def _applytfms(args):
    """
    Applies ANTs' antsApplyTransforms to the input image.
    All inputs are zipped in one tuple to make it digestible by
    multiprocessing's map
    """
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms

    in_file, in_xform, ifargs, index, newpath = args
    out_file = fname_presuffix(
        in_file, suffix=f'_xform-{index:05d}', newpath=newpath, use_ext=True
    )

    copy_dtype = ifargs.pop('copy_dtype', False)
    xfm = ApplyTransforms(
        input_image=in_file, transforms=in_xform, output_image=out_file, **ifargs
    )
    xfm.terminal_output = 'allatonce'
    xfm.resource_monitor = False
    runtime = xfm.run().runtime

    if copy_dtype:
        nii = nb.load(out_file, mmap=False)
        in_dtype = nb.load(in_file).get_data_dtype()

        # Overwrite only iff dtypes don't match
        if in_dtype != nii.get_data_dtype():
            nii.set_data_dtype(in_dtype)
            nii.to_filename(out_file)

    return (out_file, runtime.cmdline)


def _arrange_xfms(transforms, num_files, tmp_folder):
    """
    Convenience method to arrange the list of transforms that should be applied
    to each input file
    """
    base_xform = ['#Insight Transform File V1.0', '#Transform 0']
    # Initialize the transforms matrix
    xfms_T = []
    for i, tf_file in enumerate(transforms):
        if tf_file == 'identity':
            xfms_T.append([tf_file] * num_files)
            continue

        # If it is a deformation field, copy to the tfs_matrix directly
        if guess_type(tf_file)[0] != 'text/plain':
            xfms_T.append([tf_file] * num_files)
            continue

        with open(tf_file) as tf_fh:
            tfdata = tf_fh.read().strip()

        # If it is not an ITK transform file, copy to the tfs_matrix directly
        if not tfdata.startswith('#Insight Transform File'):
            xfms_T.append([tf_file] * num_files)
            continue

        # Count number of transforms in ITK transform file
        nxforms = tfdata.count('#Transform')

        # Remove first line
        tfdata = tfdata.split('\n')[1:]

        # If it is a ITK transform file with only 1 xform, copy to the tfs_matrix directly
        if nxforms == 1:
            xfms_T.append([tf_file] * num_files)
            continue

        if nxforms != num_files:
            raise RuntimeError(
                f'Number of transforms ({nxforms}) found in the ITK file does not'
                f' match the number of input image files ({num_files}).'
            )

        # At this point splitting transforms will be necessary, generate a base name
        out_base = fname_presuffix(
            tf_file, suffix=f'_pos-{i:03d}_xfm-{{:05d}}', newpath=tmp_folder.name
        ).format
        # Split combined ITK transforms file
        split_xfms = []
        for xform_i in range(nxforms):
            # Find start token to extract
            startidx = tfdata.index(f'#Transform {xform_i}')
            next_xform = base_xform + tfdata[startidx + 1 : startidx + 4] + ['']
            xfm_file = out_base(xform_i)
            with open(xfm_file, 'w') as out_xfm:
                out_xfm.write('\n'.join(next_xform))
            split_xfms.append(xfm_file)
        xfms_T.append(split_xfms)

    # Transpose back (only Python 3)
    return list(map(list, zip(*xfms_T)))
