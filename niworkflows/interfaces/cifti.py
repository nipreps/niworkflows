# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Handling connectivity: combines FreeSurfer surfaces with subcortical volumes."""
import os
from glob import glob
import json

import nibabel as nb
from nibabel import cifti2 as ci
import numpy as np
from nilearn.image import resample_to_img

from nipype.utils.filemanip import split_filename
from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec, File, traits,
    SimpleInterface, Directory
)
from templateflow.api import get as get_template

# CITFI structures with corresponding FS labels
CIFTI_STRUCT_WITH_LABELS = {
    # SURFACES
    'CIFTI_STRUCTURE_CORTEX_LEFT': None,
    'CIFTI_STRUCTURE_CORTEX_RIGHT': None,

    # SUBCORTICAL
    'CIFTI_STRUCTURE_ACCUMBENS_LEFT': [26],
    'CIFTI_STRUCTURE_ACCUMBENS_RIGHT': [58],
    'CIFTI_STRUCTURE_AMYGDALA_LEFT': [18],
    'CIFTI_STRUCTURE_AMYGDALA_RIGHT': [54],
    'CIFTI_STRUCTURE_BRAIN_STEM': [16],
    'CIFTI_STRUCTURE_CAUDATE_LEFT': [11],
    'CIFTI_STRUCTURE_CAUDATE_RIGHT': [50],
    'CIFTI_STRUCTURE_CEREBELLUM_LEFT': [
        6,  # DKT31
        8,  # HCP MNI152
    ],
    'CIFTI_STRUCTURE_CEREBELLUM_RIGHT': [
        45,  # DKT31
        47,  # HCP MNI152
    ],
    'CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_LEFT': [28],
    'CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_RIGHT': [60],
    'CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT': [17],
    'CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT': [53],
    'CIFTI_STRUCTURE_PALLIDUM_LEFT': [13],
    'CIFTI_STRUCTURE_PALLIDUM_RIGHT': [52],
    'CIFTI_STRUCTURE_PUTAMEN_LEFT': [12],
    'CIFTI_STRUCTURE_PUTAMEN_RIGHT': [51],
    'CIFTI_STRUCTURE_THALAMUS_LEFT': [10],
    'CIFTI_STRUCTURE_THALAMUS_RIGHT': [49],
}


class _GenerateCiftiInputSpec(BaseInterfaceInputSpec):
    bold_file = File(mandatory=True, exists=True, desc="input BOLD file")
    volume_target = traits.Enum("MNI152NLin6Asym", "MNI152NLin2009cAsym", usedefault=True,
                                desc="CIFTI volumetric output space")
    surface_target = traits.Enum("fsLR", "fsaverage5", "fsaverage6", usedefault=True,
                                 desc="CIFTI surface target space")
    density = traits.Enum('32k', '59k', usedefault=True,
                          help='surface hemisphere vertices')
    TR = traits.Float(mandatory=True, desc="Repetition time")
    surface_bolds = traits.List(File(exists=True), mandatory=True,
                                desc="list of surface BOLD GIFTI files"
                                     " (length 2 with order [L,R])")
    subjects_dir = Directory(mandatory=True, desc="FreeSurfer SUBJECTS_DIR")


class _GenerateCiftiOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="generated CIFTI file")
    variant = traits.Str(desc="combination of target spaces label")
    variant_key = File(exists=True, desc='file storing variant space information')


class GenerateCifti(SimpleInterface):
    """
    Generate CIFTI image from BOLD file in target spaces.

    Currently supports ``fsLR``, ``fsaverage5``, or ``fsaverage6`` for template surfaces and
    ``MNI152NLin6Asym`` or ``MNI152NLin2009cAsym`` as template volumes.

    """

    input_spec = _GenerateCiftiInputSpec
    output_spec = _GenerateCiftiOutputSpec

    _surfaces = ("fsaverage5", "fsaverage6", "fsLR")
    _volumes = ("MNI152NLin2009cAsym", "MNI152NLin6Asym")

    def _run_interface(self, runtime):
        targets = self._define_variant()
        annotation_files, label_file = self._fetch_data()
        self._results["out_file"] = self._create_cifti_image(
            self.inputs.bold_file,
            label_file,
            self.inputs.surface_bolds,
            targets,
            self.inputs.TR,
            annotation_files=annotation_files)
        return runtime

    def _define_variant(self):
        """Assign arbitrary label to combination of CIFTI spaces."""
        space = None
        variants = {
            'HCP grayordinates': ['fsLR', 'MNI152NLin6Asym'],
            'space1': ['fsaverage5', 'MNI152NLin2009cAsym'],
            'space2': ['fsaverage6', 'MNI152NLin2009cAsym'],
        }
        for sp, targets in variants.items():
            if all(
                target in targets for target in
                [self.inputs.surface_target, self.inputs.volume_target]
            ):
                space = sp
        if space is None:
            raise NotImplementedError

        variant_key = os.path.abspath('dtseries_variant.json')
        out_json = {
            'space': space,
            'surface': variants[space][0],
            'volume': variants[space][1]
        }
        if self.inputs.surface_target == 'fsLR':
            # 91k == 2mm resolution, 170k == 1.6mm resolution
            out_json['grayordinates'] = '91k' if self.inputs.density == '32k' else '170k'
        with open(variant_key, 'w') as fp:
            json.dump(out_json, fp)
        self._results['variant_key'] = variant_key
        self._results['variant'] = space
        return variants[space]

    def _fetch_data(self):
        """Converts inputspec to files"""
        if (
            self.inputs.surface_target not in self._surfaces or
            self.inputs.volume_target not in self._volumes
        ):
            raise NotImplementedError(
                "Target space (surface: {0}, volume: {1}) is not supported".format(
                    self.inputs.surface_target, self.inputs.volume_target
                )
            )

        tpl_kwargs = {'suffix': 'dseg'}
        annotation_files = None
        if self.inputs.volume_target == "MNI152NLin2009cAsym":
            tpl_kwargs.update({
                'resolution': '2',
                'desc': 'DKT31',
            })
            annotation_files = sorted(
                glob(os.path.join(self.inputs.subjects_dir,
                                  self.inputs.surface_target,
                                  'label',
                                  '*h.aparc.annot'))
            )

        elif self.inputs.volume_target == 'MNI152NLin6Asym':
            res = '2' if self.inputs.density == '32k' else '5'

            tpl_kwargs.update({
                'atlas': 'HCP',
                'resolution': res,
            })
            annotation_files = [
                str(f) for f in get_template(
                    'fsLR', density=self.inputs.density, desc='nomedialwall', suffix='dparc'
                )
            ]

        if len(annotation_files) != 2:
            raise IOError("Invalid number of surface annotation files")

        label_file = str(get_template(self.inputs.volume_target, **tpl_kwargs))
        return annotation_files, label_file

    @staticmethod
    def _create_cifti_image(bold_file, label_file, bold_surfs,
                            targets, tr, annotation_files):
        """
        Generate CIFTI image in target space.

        Parameters
            bold_file : str
                BOLD volumetric timeseries
            label_file : str
                Subcortical label file
            bold_surfs : list
                BOLD surface timeseries [L,R]
            targets : list
                Surface and volumetric output spaces
            tr : float
                BOLD repetition time
            annotation_files : list
                Surface label files used to remove medial wall

        Returns
            out :
                BOLD data saved as CIFTI dtseries
        """
        bold_img = nb.load(bold_file)
        label_img = nb.load(label_file)
        if label_img.shape != bold_img.shape[:3]:
            bold_img = resample_to_img(bold_img, label_img)

        bold_data = bold_img.get_fdata(dtype='float32')
        timepoints = bold_img.shape[3]
        label_data = np.asanyarray(label_img.dataobj).astype('int16')

        # Create brain models
        idx_offset = 0
        brainmodels = []
        bm_ts = np.empty((timepoints, 0))

        for structure, labels in CIFTI_STRUCT_WITH_LABELS.items():
            if labels is None:  # surface model
                model_type = "CIFTI_MODEL_TYPE_SURFACE"
                # use the corresponding annotation
                hemi = structure.split('_')[-1]
                # currently only supports L/R cortex
                surf = nb.load(bold_surfs[hemi == "RIGHT"])
                surf_verts = len(surf.darrays[0].data)
                if annotation_files[0].endswith('.annot'):
                    annot = nb.freesurfer.read_annot(annotation_files[hemi == "RIGHT"])
                    # remove medial wall
                    medial = np.nonzero(annot[0] != annot[2].index(b'unknown'))[0]
                else:
                    annot = nb.load(annotation_files[hemi == "RIGHT"])
                    medial = np.nonzero(annot.darrays[0].data)[0]
                # extract values across volumes
                ts = np.array([tsarr.data[medial] for tsarr in surf.darrays])

                vert_idx = ci.Cifti2VertexIndices(medial)
                bm = ci.Cifti2BrainModel(index_offset=idx_offset,
                                         index_count=len(vert_idx),
                                         model_type=model_type,
                                         brain_structure=structure,
                                         vertex_indices=vert_idx,
                                         n_surface_vertices=surf_verts)
                bm_ts = np.column_stack((bm_ts, ts))
                idx_offset += len(vert_idx)
                brainmodels.append(bm)
            else:
                model_type = "CIFTI_MODEL_TYPE_VOXELS"
                vox = []
                ts = None
                for label in labels:
                    ijk = np.nonzero(label_data == label)
                    ts = (bold_data[ijk] if ts is None
                          else np.concatenate((ts, bold_data[ijk])))
                    vox += [[ijk[0][ix], ijk[1][ix], ijk[2][ix]]
                            for ix, row in enumerate(ts)]

                bm_ts = np.column_stack((bm_ts, ts.T))

                vox = ci.Cifti2VoxelIndicesIJK(vox)
                bm = ci.Cifti2BrainModel(index_offset=idx_offset,
                                         index_count=len(vox),
                                         model_type=model_type,
                                         brain_structure=structure,
                                         voxel_indices_ijk=vox)
                idx_offset += len(vox)
                brainmodels.append(bm)

        # add volume information
        brainmodels.append(
            ci.Cifti2Volume(
                bold_img.shape[:3],
                ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(-3, bold_img.affine)
            )
        )

        # generate Matrix information
        series_map = ci.Cifti2MatrixIndicesMap(
            (0, ),
            'CIFTI_INDEX_TYPE_SERIES',
            number_of_series_points=timepoints,
            series_exponent=0,
            series_start=0.,
            series_step=tr,
            series_unit='SECOND'
        )
        geometry_map = ci.Cifti2MatrixIndicesMap(
            (1, ),
            'CIFTI_INDEX_TYPE_BRAIN_MODELS',
            maps=brainmodels
        )
        # provide some metadata to CIFTI matrix
        meta = {
            "target_surface": targets[0],
            "target_volume": targets[1],
        }
        # generate and save CIFTI image
        matrix = ci.Cifti2Matrix()
        matrix.append(series_map)
        matrix.append(geometry_map)
        matrix.metadata = ci.Cifti2MetaData(meta)
        hdr = ci.Cifti2Header(matrix)
        img = ci.Cifti2Image(bm_ts, hdr)
        img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_DENSE_SERIES')

        _, out_base, _ = split_filename(bold_file)
        out_file = "{}.dtseries.nii".format(out_base)
        ci.save(img, out_file)
        return os.path.join(os.getcwd(), out_file)


class _CiftiNameSourceInputSpec(BaseInterfaceInputSpec):
    variant = traits.Str(mandatory=True,
                         desc=('unique label of spaces used in combination to'
                               ' generate CIFTI file'))


class _CiftiNameSourceOutputSpec(TraitedSpec):
    out_name = traits.Str(desc='(partial) filename formatted according to template')


class CiftiNameSource(SimpleInterface):
    """Construct new filename based on unique label of spaces used to generate a CIFTI file."""

    input_spec = _CiftiNameSourceInputSpec
    output_spec = _CiftiNameSourceOutputSpec

    def _run_interface(self, runtime):
        suffix = 'bold.dtseries'
        if 'hcp' in self.inputs.variant:
            suffix = 'space-hcp_bold.dtseries'
        self._results['out_name'] = suffix
        return runtime
