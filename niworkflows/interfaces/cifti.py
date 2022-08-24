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
from pathlib import Path
import json
import warnings

import nibabel as nb
from nibabel import cifti2 as ci
import numpy as np
from nilearn.image import resample_to_img
from nipype.utils.filemanip import split_filename
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    traits,
    SimpleInterface,
    Directory,
)
import templateflow.api as tf

from niworkflows.interfaces.nibabel import reorient_image

CIFTI_SURFACES = ("fsaverage5", "fsaverage6", "fsLR")
CIFTI_VOLUMES = ("MNI152NLin2009cAsym", "MNI152NLin6Asym")
CIFTI_STRUCT_WITH_LABELS = {  # CITFI structures with corresponding labels
    # SURFACES
    "CIFTI_STRUCTURE_CORTEX_LEFT": None,
    "CIFTI_STRUCTURE_CORTEX_RIGHT": None,
    # SUBCORTICAL
    "CIFTI_STRUCTURE_ACCUMBENS_LEFT": (26,),
    "CIFTI_STRUCTURE_ACCUMBENS_RIGHT": (58,),
    "CIFTI_STRUCTURE_AMYGDALA_LEFT": (18,),
    "CIFTI_STRUCTURE_AMYGDALA_RIGHT": (54,),
    "CIFTI_STRUCTURE_BRAIN_STEM": (16,),
    "CIFTI_STRUCTURE_CAUDATE_LEFT": (11,),
    "CIFTI_STRUCTURE_CAUDATE_RIGHT": (50,),
    "CIFTI_STRUCTURE_CEREBELLUM_LEFT": (6, 8,),  # DKT31  # HCP MNI152
    "CIFTI_STRUCTURE_CEREBELLUM_RIGHT": (45, 47,),  # DKT31  # HCP MNI152
    "CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_LEFT": (28,),
    "CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_RIGHT": (60,),
    "CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT": (17,),
    "CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT": (53,),
    "CIFTI_STRUCTURE_PALLIDUM_LEFT": (13,),
    "CIFTI_STRUCTURE_PALLIDUM_RIGHT": (52,),
    "CIFTI_STRUCTURE_PUTAMEN_LEFT": (12,),
    "CIFTI_STRUCTURE_PUTAMEN_RIGHT": (51,),
    "CIFTI_STRUCTURE_THALAMUS_LEFT": (10,),
    "CIFTI_STRUCTURE_THALAMUS_RIGHT": (49,),
}
CIFTI_VARIANTS = {
    "HCP grayordinates": ("fsLR", "MNI152NLin6Asym"),
    "fMRIPrep grayordinates": ("fsaverage", "MNI152NLin2009cAsym"),
}


class _GenerateCiftiInputSpec(BaseInterfaceInputSpec):
    bold_file = File(mandatory=True, exists=True, desc="input BOLD file")
    volume_target = traits.Enum(
        "MNI152NLin6Asym",
        "MNI152NLin2009cAsym",
        usedefault=True,
        desc="CIFTI volumetric output space",
    )
    surface_target = traits.Enum(
        "fsLR",
        "fsaverage5",
        "fsaverage6",
        usedefault=True,
        desc="CIFTI surface target space",
    )
    surface_density = traits.Enum(
        "10k", "32k", "41k", "59k", desc="Surface vertices density."
    )
    TR = traits.Float(mandatory=True, desc="Repetition time")
    surface_bolds = traits.List(
        File(exists=True),
        mandatory=True,
        desc="list of surface BOLD GIFTI files" " (length 2 with order [L,R])",
    )
    subjects_dir = Directory(mandatory=True, desc="FreeSurfer SUBJECTS_DIR")


class _GenerateCiftiOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="generated CIFTI file")
    out_metadata = File(exists=True, desc="variant metadata JSON")
    variant = traits.Str(desc="Name of variant space")
    density = traits.Str(desc="Total number of grayordinates")


class GenerateCifti(SimpleInterface):
    """
    Generate CIFTI image from BOLD file in target spaces.

    Currently supports ``fsLR``, ``fsaverage5``, or ``fsaverage6`` for template surfaces and
    ``MNI152NLin6Asym`` or ``MNI152NLin2009cAsym`` as template volumes.

    """

    input_spec = _GenerateCiftiInputSpec
    output_spec = _GenerateCiftiOutputSpec

    def _run_interface(self, runtime):
        annotation_files, label_file = _get_cifti_data(
            self.inputs.surface_target,
            self.inputs.volume_target,
            self.inputs.subjects_dir,
            self.inputs.surface_density,
        )
        out_metadata, variant, density = _get_cifti_variant(
            self.inputs.surface_target,
            self.inputs.volume_target,
            self.inputs.surface_density,
        )
        self._results.update({"out_metadata": out_metadata, "variant": variant})
        if density:
            self._results["density"] = density

        self._results["out_file"] = _create_cifti_image(
            self.inputs.bold_file,
            label_file,
            self.inputs.surface_bolds,
            annotation_files,
            self.inputs.TR,
            (self.inputs.surface_target, self.inputs.volume_target),
        )
        return runtime


class _CiftiNameSourceInputSpec(BaseInterfaceInputSpec):
    variant = traits.Str(
        mandatory=True,
        desc="unique label of spaces used in combination to generate CIFTI file",
    )
    density = traits.Str(desc="density label")


class _CiftiNameSourceOutputSpec(TraitedSpec):
    out_name = traits.Str(desc="(partial) filename formatted according to template")


class CiftiNameSource(SimpleInterface):
    """
    Construct new filename based on unique label of spaces used to generate a CIFTI file.

    Examples
    --------
    >>> namer = CiftiNameSource()
    >>> namer.inputs.variant = 'HCP grayordinates'
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
        suffix = ""
        if "hcp" in self.inputs.variant.lower():
            suffix += "space-fsLR_"
        if self.inputs.density:
            suffix += "den-{}_".format(self.inputs.density)

        suffix += "bold.dtseries"
        self._results["out_name"] = suffix
        return runtime


def _get_cifti_data(surface, volume, subjects_dir=None, density=None):
    """
    Fetch surface and volumetric label files for CIFTI creation.

    Parameters
    ----------
    surface : str
        Target surface space
    volume : str
        Target volume space
    subjects_dir : str, optional
        Path to FreeSurfer subjects directory (required `fsaverage5`/`fsaverage6` surfaces)
    density : str, optional
        Surface density (required for `fsLR` surfaces)

    Returns
    -------
    annotation_files : list
        Surface annotation files to allow removal of medial wall
    label_file : str
        Volumetric label file of subcortical structures

    Examples
    --------
    >>> annots, label = _get_cifti_data('fsLR', 'MNI152NLin6Asym', density='32k')
    >>> annots  # doctest: +ELLIPSIS
    ['.../tpl-fsLR_hemi-L_den-32k_desc-nomedialwall_dparc.label.gii', \
     '.../tpl-fsLR_hemi-R_den-32k_desc-nomedialwall_dparc.label.gii']
    >>> label  # doctest: +ELLIPSIS
    '.../tpl-MNI152NLin6Asym_res-02_atlas-HCP_dseg.nii.gz'

    """
    if surface not in CIFTI_SURFACES or volume not in CIFTI_VOLUMES:
        raise NotImplementedError(
            "Variant (surface: {0}, volume: {1}) is not supported".format(
                surface, volume
            )
        )

    tpl_kwargs = {"suffix": "dseg"}
    # fMRIPrep grayordinates
    if volume == "MNI152NLin2009cAsym":
        tpl_kwargs.update({"resolution": "2", "desc": "DKT31"})
        annotation_files = sorted(
            (subjects_dir / surface / "label").glob("*h.aparc.annot")
        )
    # HCP grayordinates
    elif volume == "MNI152NLin6Asym":
        # templateflow specific resolutions (2mm, 1.6mm)
        res = {"32k": "2", "59k": "6"}[density]
        tpl_kwargs.update({"atlas": "HCP", "resolution": res})
        annotation_files = [
            str(f)
            for f in tf.get(
                "fsLR", density=density, desc="nomedialwall", suffix="dparc"
            )
        ]

    if len(annotation_files) != 2:
        raise IOError("Invalid number of surface annotation files")
    label_file = str(tf.get(volume, **tpl_kwargs))
    return annotation_files, label_file


def _get_cifti_variant(surface, volume, density=None):
    """
    Identify CIFTI variant and return metadata.

    Parameters
    ----------
    surface : str
        Target surface space
    volume : str
        Target volume space
    density : str, optional
        Surface density (required for `fsLR` surfaces)

    Returns
    -------
    out_metadata : str
        JSON file with variant metadata
    variant : str
        Name of CIFTI variant

    Examples
    --------
    >>> metafile, variant, _ = _get_cifti_variant('fsaverage5', 'MNI152NLin2009cAsym')
    >>> str(metafile)  # doctest: +ELLIPSIS
    '.../dtseries_variant.json'
    >>> variant
    'fMRIPrep grayordinates'

    >>> _, variant, grayords = _get_cifti_variant('fsLR', 'MNI152NLin6Asym', density='59k')
    >>> variant
    'HCP grayordinates'
    >>> grayords
    '170k'

    """
    if surface in ("fsaverage5", "fsaverage6"):
        density = {"fsaverage5": "10k", "fsaverage6": "41k"}[surface]
        surface = "fsaverage"

    for variant, targets in CIFTI_VARIANTS.items():
        if all(target in targets for target in (surface, volume)):
            break
        variant = None
    if variant is None:
        raise NotImplementedError(
            "No corresponding variant for (surface: {0}, volume: {1})".format(
                surface, volume
            )
        )

    grayords = None
    out_metadata = Path.cwd() / "dtseries_variant.json"
    out_json = {
        "space": variant,
        "surface": surface,
        "volume": volume,
        "surface_density": density,
    }
    if surface == "fsLR":
        grayords = {"32k": "91k", "59k": "170k"}[density]
        out_json["grayordinates"] = grayords

    out_metadata.write_text(json.dumps(out_json, indent=2))
    return out_metadata, variant, grayords


def _create_cifti_image(
    bold_file, label_file, bold_surfs, annotation_files, tr, targets
):
    """
    Generate CIFTI image in target space.

    Parameters
    ----------
    bold_file : str
        BOLD volumetric timeseries
    label_file : str
        Subcortical label file
    bold_surfs : list
        BOLD surface timeseries [L,R]
    annotation_files : list
        Surface label files used to remove medial wall
    tr : float
        BOLD repetition time
    targets : tuple or list
        Surface and volumetric output spaces

    Returns
    -------
    out :
        BOLD data saved as CIFTI dtseries
    """
    bold_img = nb.load(bold_file)
    label_img = nb.load(label_file)
    if label_img.shape != bold_img.shape[:3]:
        warnings.warn("Resampling bold volume to match label dimensions")
        bold_img = resample_to_img(bold_img, label_img)

    # ensure images match HCP orientation (LAS)
    bold_img = reorient_image(bold_img, target_ornt="LAS")
    label_img = reorient_image(label_img, target_ornt="LAS")

    bold_data = bold_img.get_fdata(dtype="float32")
    timepoints = bold_img.shape[3]
    label_data = np.asanyarray(label_img.dataobj).astype("int16")

    # Create brain models
    idx_offset = 0
    brainmodels = []
    bm_ts = np.empty((timepoints, 0), dtype="float32")

    for structure, labels in CIFTI_STRUCT_WITH_LABELS.items():
        if labels is None:  # surface model
            model_type = "CIFTI_MODEL_TYPE_SURFACE"
            # use the corresponding annotation
            hemi = structure.split("_")[-1]
            # currently only supports L/R cortex
            surf_ts = nb.load(bold_surfs[hemi == "RIGHT"])
            surf_verts = len(surf_ts.darrays[0].data)
            if annotation_files[0].endswith(".annot"):
                annot = nb.freesurfer.read_annot(annotation_files[hemi == "RIGHT"])
                # remove medial wall
                medial = np.nonzero(annot[0] != annot[2].index(b"unknown"))[0]
            else:
                annot = nb.load(annotation_files[hemi == "RIGHT"])
                medial = np.nonzero(annot.darrays[0].data)[0]
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
            model_type = "CIFTI_MODEL_TYPE_VOXELS"
            vox = []
            ts = None
            for label in labels:
                ijk = np.nonzero(label_data == label)
                if ijk[0].size == 0:  # skip label if nothing matches
                    continue
                ts = (
                    bold_data[ijk]
                    if ts is None
                    else np.concatenate((ts, bold_data[ijk]))
                )
                vox += [
                    [ijk[0][idx], ijk[1][idx], ijk[2][idx]] for idx in range(len(ts))
                ]

            vox = ci.Cifti2VoxelIndicesIJK(vox)
            bm = ci.Cifti2BrainModel(
                index_offset=idx_offset,
                index_count=len(vox),
                model_type=model_type,
                brain_structure=structure,
                voxel_indices_ijk=vox,
            )
            idx_offset += len(vox)
            bm_ts = np.column_stack((bm_ts, ts.T))
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
        "CIFTI_INDEX_TYPE_SERIES",
        number_of_series_points=timepoints,
        series_exponent=0,
        series_start=0.0,
        series_step=tr,
        series_unit="SECOND",
    )
    geometry_map = ci.Cifti2MatrixIndicesMap(
        (1,), "CIFTI_INDEX_TYPE_BRAIN_MODELS", maps=brainmodels
    )
    # provide some metadata to CIFTI matrix
    meta = {
        "surface": targets[0],
        "volume": targets[1],
    }
    # generate and save CIFTI image
    matrix = ci.Cifti2Matrix()
    matrix.append(series_map)
    matrix.append(geometry_map)
    matrix.metadata = ci.Cifti2MetaData(meta)
    hdr = ci.Cifti2Header(matrix)
    img = ci.Cifti2Image(dataobj=bm_ts, header=hdr)
    img.set_data_dtype(bold_img.get_data_dtype())
    img.nifti_header.set_intent("NIFTI_INTENT_CONNECTIVITY_DENSE_SERIES")

    out_file = "{}.dtseries.nii".format(split_filename(bold_file)[1])
    ci.save(img, out_file)
    return Path.cwd() / out_file
