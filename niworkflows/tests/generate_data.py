from pathlib import Path

import numpy as np


def _create_dtseries_cifti(timepoints, models):
    """Create a dense timeseries CIFTI-2 file"""
    import nibabel.cifti2 as ci

    def create_series_map():
        return ci.Cifti2MatrixIndicesMap(
            (0,),
            'CIFTI_INDEX_TYPE_SERIES',
            number_of_series_points=timepoints,
            series_exponent=0,
            series_start=0,
            series_step=1,
            series_unit='SECOND',
        )

    def create_geometry_map():
        index_offset = 0
        brain_models = []
        timeseries = np.zeros((timepoints, 0))

        for name, data in models:
            if 'CORTEX' in name:
                model_type = 'CIFTI_MODEL_TYPE_SURFACE'
                attr = 'vertex_indices'
                indices = ci.Cifti2VertexIndices(np.arange(len(data)))
            else:
                model_type = 'CIFTI_MODEL_TYPE_VOXELS'
                attr = 'voxel_indices_ijk'
                indices = ci.Cifti2VoxelIndicesIJK(np.arange(len(data)))
            bm = ci.Cifti2BrainModel(
                index_offset=index_offset,
                index_count=len(data),
                model_type=model_type,
                brain_structure=name,
            )
            setattr(bm, attr, indices)
            if model_type == 'CIFTI_MODEL_TYPE_SURFACE':
                # define total vertices for surface models
                bm.surface_number_of_vertices = 32492
            index_offset += len(data)
            brain_models.append(bm)
            timeseries = np.column_stack((timeseries, data.T))

        brain_models.append(
            ci.Cifti2Volume(
                (4, 4, 4),
                ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(-3, np.eye(4)),
            )
        )

        return ci.Cifti2MatrixIndicesMap(
            (1,),
            'CIFTI_INDEX_TYPE_BRAIN_MODELS',
            maps=brain_models,
        ), timeseries

    matrix = ci.Cifti2Matrix()
    series_map = create_series_map()
    geometry_map, ts = create_geometry_map()
    matrix.append(series_map)
    matrix.append(geometry_map)
    hdr = ci.Cifti2Header(matrix)
    img = ci.Cifti2Image(dataobj=ts, header=hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_DENSE_SERIES')

    out_file = Path('test.dtseries.nii').absolute()
    ci.save(img, out_file)
    return out_file
