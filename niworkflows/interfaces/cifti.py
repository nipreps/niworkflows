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
"""Handling connectivity: combines FreeSurfer surfaces with subcortical volumes."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import nibabel as nb
import numpy as np
import templateflow.api as tf
from nibabel import cifti2 as ci
from nilearn.image import resample_to_img
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from nipype.utils.filemanip import split_filename

from niworkflows.interfaces.nibabel import reorient_image

CIFTI_STRUCT_WITH_LABELS = {  # CITFI structures with corresponding labels
    # SURFACES
    'CIFTI_STRUCTURE_CORTEX_LEFT': None,
    'CIFTI_STRUCTURE_CORTEX_RIGHT': None,
    # SUBCORTICAL
    'CIFTI_STRUCTURE_ACCUMBENS_LEFT': (26,),
    'CIFTI_STRUCTURE_ACCUMBENS_RIGHT': (58,),
    'CIFTI_STRUCTURE_AMYGDALA_LEFT': (18,),
    'CIFTI_STRUCTURE_AMYGDALA_RIGHT': (54,),
    'CIFTI_STRUCTURE_BRAIN_STEM': (16,),
    'CIFTI_STRUCTURE_CAUDATE_LEFT': (11,),
    'CIFTI_STRUCTURE_CAUDATE_RIGHT': (50,),
    'CIFTI_STRUCTURE_CEREBELLUM_LEFT': (8,),  # HCP MNI152
    'CIFTI_STRUCTURE_CEREBELLUM_RIGHT': (47,),  # HCP MNI152
    'CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_LEFT': (28,),
    'CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_RIGHT': (60,),
    'CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT': (17,),
    'CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT': (53,),
    'CIFTI_STRUCTURE_PALLIDUM_LEFT': (13,),
    'CIFTI_STRUCTURE_PALLIDUM_RIGHT': (52,),
    'CIFTI_STRUCTURE_PUTAMEN_LEFT': (12,),
    'CIFTI_STRUCTURE_PUTAMEN_RIGHT': (51,),
    'CIFTI_STRUCTURE_THALAMUS_LEFT': (10,),
    'CIFTI_STRUCTURE_THALAMUS_RIGHT': (49,),
}


class _GenerateCiftiInputSpec(BaseInterfaceInputSpec):
    bold_file = File(mandatory=True, exists=True, desc='input BOLD file')
    volume_target = traits.Enum(
        'MNI152NLin6Asym',
        usedefault=True,
        desc='CIFTI volumetric output space',
    )
    surface_target = traits.Enum(
        'fsLR',
        usedefault=True,
        desc='CIFTI surface target space',
    )
    grayordinates = traits.Enum('91k', '170k', usedefault=True, desc='Final CIFTI grayordinates')
    TR = traits.Float(mandatory=True, desc='Repetition time')
    surface_bolds = traits.List(
        File(exists=True),
        mandatory=True,
        desc='list of surface BOLD GIFTI files (length 2 with order [L,R])',
    )


class _GenerateCiftiOutputSpec(TraitedSpec):
    out_file = File(desc='generated CIFTI file')
    out_metadata = File(desc='CIFTI metadata JSON')


class GenerateCifti(SimpleInterface):
    """
    Generate a HCP-style CIFTI image from BOLD file in target spaces.
    """

    input_spec = _GenerateCiftiInputSpec
    output_spec = _GenerateCiftiOutputSpec

    def _run_interface(self, runtime):
        surface_labels, volume_labels, metadata = _prepare_cifti(self.inputs.grayordinates)
        self._results['out_file'] = _create_cifti_image(
            self.inputs.bold_file,
            volume_labels,
            self.inputs.surface_bolds,
            surface_labels,
            self.inputs.TR,
            metadata,
        )
        metadata_file = Path('bold.dtseries.json').absolute()
        metadata_file.write_text(json.dumps(metadata, indent=2))
        self._results['out_metadata'] = str(metadata_file)
        return runtime


class _CiftiNameSourceInputSpec(BaseInterfaceInputSpec):
    space = traits.Str(
        mandatory=True,
        desc='the space identifier',
    )
    density = traits.Str(desc='density label')


class _CiftiNameSourceOutputSpec(TraitedSpec):
    out_name = traits.Str(desc='(partial) filename formatted according to template')


class CiftiNameSource(SimpleInterface):
    """
    Construct new filename based on unique label of spaces used to generate a CIFTI file.

    Examples
    --------
    >>> namer = CiftiNameSource()
    >>> namer.inputs.space = 'fsLR'
    >>> res = namer.run()
    >>> res.outputs.out_name
    'space-fsLR_bold.dtseries'

    >>> namer.inputs.density = '32k'
    >>> res = namer.run()
    >>> res.outputs.out_name
    'space-fsLR_den-32k_bold.dtseries'

    """

    input_spec = _CiftiNameSourceInputSpec
    output_spec = _CiftiNameSourceOutputSpec

    def _run_interface(self, runtime):
        entities = [('space', self.inputs.space)]
        if self.inputs.density:
            entities.append(('den', self.inputs.density))

        out_name = '_'.join([f'{k}-{v}' for k, v in entities] + ['bold.dtseries'])
        self._results['out_name'] = out_name
        return runtime


def _prepare_cifti(grayordinates: str) -> tuple[list, str, dict]:
    """
    Fetch the required templates needed for CIFTI-2 generation, based on input surface density.

    Parameters
    ----------
    grayordinates :
        Total CIFTI grayordinates (91k, 170k)

    Returns
    -------
    surface_labels
        Surface label files for vertex inclusion/exclusion.
    volume_label
        Volumetric label file of subcortical structures.
    metadata
        Dictionary with BIDS metadata.

    Examples
    --------
    >>> surface_labels, volume_labels, metadata = _prepare_cifti('91k')
    >>> surface_labels  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['.../tpl-fsLR_hemi-L_den-32k_desc-nomedialwall_dparc.label.gii', \
     '.../tpl-fsLR_hemi-R_den-32k_desc-nomedialwall_dparc.label.gii']
    >>> volume_labels  # doctest: +ELLIPSIS
    '.../tpl-MNI152NLin6Asym_res-02_atlas-HCP_dseg.nii.gz'
    >>> metadata # doctest: +ELLIPSIS
    {'Density': '91,282 grayordinates corresponding to all of the grey matter sampled at a \
2mm average vertex spacing... 'SpatialReference': {'VolumeReference': ...

    """

    grayord_key = {
        '91k': {'surface-den': '32k', 'tf-res': '02', 'grayords': '91,282', 'res-mm': '2mm'},
        '170k': {'surface-den': '59k', 'tf-res': '06', 'grayords': '170,494', 'res-mm': '1.6mm'},
    }
    if grayordinates not in grayord_key:
        raise NotImplementedError(f'Grayordinates {grayordinates} is not supported.')

    tf_vol_res = grayord_key[grayordinates]['tf-res']
    total_grayords = grayord_key[grayordinates]['grayords']
    res_mm = grayord_key[grayordinates]['res-mm']
    surface_density = grayord_key[grayordinates]['surface-den']
    # Fetch templates
    surface_labels = [
        str(
            tf.get(
                'fsLR',
                density=surface_density,
                hemi=hemi,
                desc='nomedialwall',
                suffix='dparc',
                raise_empty=True,
            )
        )
        for hemi in ('L', 'R')
    ]
    volume_label = str(
        tf.get(
            'MNI152NLin6Asym', suffix='dseg', atlas='HCP', resolution=tf_vol_res, raise_empty=True
        )
    )

    tf_url = 'https://templateflow.s3.amazonaws.com'
    volume_url = f'{tf_url}/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-{tf_vol_res}_T1w.nii.gz'
    surfaces_url = (  # midthickness is the default, but varying levels of inflation are all valid
        f'{tf_url}/tpl-fsLR/tpl-fsLR_den-{surface_density}_hemi-%s_midthickness.surf.gii'
    )
    metadata = {
        'Density': (
            f'{total_grayords} grayordinates corresponding to all of the grey matter sampled at a '
            f'{res_mm} average vertex spacing on the surface and as {res_mm} voxels subcortically'
        ),
        'SpatialReference': {
            'VolumeReference': volume_url,
            'CIFTI_STRUCTURE_LEFT_CORTEX': surfaces_url % 'L',
            'CIFTI_STRUCTURE_RIGHT_CORTEX': surfaces_url % 'R',
        },
    }
    return surface_labels, volume_label, metadata


def _create_cifti_image(
    bold_file: str,
    volume_label: str,
    bold_surfs: tuple[str, str],
    surface_labels: tuple[str, str],
    tr: float,
    metadata: dict | None = None,
):
    """
    Generate CIFTI image in target space.

    Parameters
    ----------
    bold_file
        BOLD volumetric timeseries
    volume_label
        Subcortical label file
    bold_surfs
        BOLD surface timeseries (L,R)
    surface_labels
        Surface label files used to remove medial wall (L,R)
    tr
        BOLD repetition time
    metadata
        Metadata to include in CIFTI header

    Returns
    -------
    out :
        BOLD data saved as CIFTI dtseries
    """
    bold_img = nb.load(bold_file)
    label_img = nb.load(volume_label)
    if label_img.shape != bold_img.shape[:3]:
        warnings.warn('Resampling bold volume to match label dimensions', stacklevel=1)
        bold_img = resample_to_img(bold_img, label_img)

    # ensure images match HCP orientation (LAS)
    bold_img = reorient_image(bold_img, target_ornt='LAS')
    label_img = reorient_image(label_img, target_ornt='LAS')

    bold_data = bold_img.get_fdata(dtype='float32')
    timepoints = bold_img.shape[3]
    label_data = np.asanyarray(label_img.dataobj).astype('int16')

    # Create brain models
    idx_offset = 0
    brainmodels = []
    bm_ts = np.empty((timepoints, 0), dtype='float32')

    for structure, labels in CIFTI_STRUCT_WITH_LABELS.items():
        if labels is None:  # surface model
            model_type = 'CIFTI_MODEL_TYPE_SURFACE'
            # use the corresponding annotation
            hemi = structure.split('_')[-1]
            # currently only supports L/R cortex
            surf_ts = nb.load(bold_surfs[hemi == 'RIGHT'])
            surf_verts = len(surf_ts.darrays[0].data)
            labels = nb.load(surface_labels[hemi == 'RIGHT'])
            medial = np.nonzero(labels.darrays[0].data)[0]
            # extract values across volumes
            ts = np.array([tsarr.data[medial] for tsarr in surf_ts.darrays])

            vert_idx = ci.Cifti2VertexIndices(medial)
            bm = ci.Cifti2BrainModel(
                index_offset=idx_offset,
                index_count=len(vert_idx),
                model_type=model_type,
                brain_structure=structure,
                vertex_indices=vert_idx,
                n_surface_vertices=surf_verts,
            )
            idx_offset += len(vert_idx)
            bm_ts = np.column_stack((bm_ts, ts))
        else:
            model_type = 'CIFTI_MODEL_TYPE_VOXELS'
            vox = []
            ts = []
            for label in labels:
                # nonzero returns indices in row-major (C) order
                # NIfTI uses column-major (Fortran) order, so HCP generates indices in F order
                # Therefore flip the data and label the indices backwards
                k, j, i = np.nonzero(label_data.T == label)
                if k.size == 0:  # skip label if nothing matches
                    continue
                ts.append(bold_data[i, j, k])
                vox.append(np.stack([i, j, k]).T)

            vox_indices_ijk = ci.Cifti2VoxelIndicesIJK(np.concatenate(vox))
            bm = ci.Cifti2BrainModel(
                index_offset=idx_offset,
                index_count=len(vox_indices_ijk),
                model_type=model_type,
                brain_structure=structure,
                voxel_indices_ijk=vox_indices_ijk,
            )
            idx_offset += len(vox_indices_ijk)
            bm_ts = np.column_stack((bm_ts, np.concatenate(ts).T))
        # add each brain structure to list
        brainmodels.append(bm)

    # add volume information
    brainmodels.append(
        ci.Cifti2Volume(
            bold_img.shape[:3],
            ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(-3, bold_img.affine),
        )
    )

    # generate Matrix information
    series_map = ci.Cifti2MatrixIndicesMap(
        (0,),
        'CIFTI_INDEX_TYPE_SERIES',
        number_of_series_points=timepoints,
        series_exponent=0,
        series_start=0.0,
        series_step=tr,
        series_unit='SECOND',
    )
    geometry_map = ci.Cifti2MatrixIndicesMap(
        (1,), 'CIFTI_INDEX_TYPE_BRAIN_MODELS', maps=brainmodels
    )
    # provide some metadata to CIFTI matrix
    if not metadata:
        metadata = {
            'surface': 'fsLR',
            'volume': 'MNI152NLin6Asym',
        }
    # generate and save CIFTI image
    matrix = ci.Cifti2Matrix()
    matrix.append(series_map)
    matrix.append(geometry_map)
    matrix.metadata = ci.Cifti2MetaData(metadata)
    hdr = ci.Cifti2Header(matrix)
    img = ci.Cifti2Image(dataobj=bm_ts, header=hdr)
    img.set_data_dtype(bold_img.get_data_dtype())
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_DENSE_SERIES')

    out_file = f'{split_filename(bold_file)[1]}.dtseries.nii'
    ci.save(img, out_file)
    return Path.cwd() / out_file
