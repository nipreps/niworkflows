# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Handling surfaces."""
import os
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import nibabel as nb

from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    TraitedSpec,
    DynamicTraitedSpec,
    SimpleInterface,
    CommandLine,
    CommandLineInputSpec,
    File,
    traits,
    isdefined,
    InputMultiPath,
    OutputMultiPath,
    Undefined,
)


SECONDARY_ANAT_STRUC = {
    "smoothwm": "GrayWhite",
    "pial": "Pial",
    "midthickness": "GrayMid",
}


class _NormalizeSurfInputSpec(BaseInterfaceInputSpec):
    in_file = File(mandatory=True, exists=True, desc="Freesurfer-generated GIFTI file")
    transform_file = File(exists=True, desc="FSL or LTA affine transform file")


class _NormalizeSurfOutputSpec(TraitedSpec):
    out_file = File(desc="output file with re-centered GIFTI coordinates")


class NormalizeSurf(SimpleInterface):
    """
    Normalize a FreeSurfer-generated GIFTI image.

    FreeSurfer includes an offset to the center of the brain volume that is not
    respected by all software packages.
    Normalization involves adding this offset to the coordinates of all
    vertices, and zeroing out that offset, to ensure consistent behavior
    across software packages.
    In particular, this normalization is consistent with the Human Connectome
    Project pipeline (see `AlgorithmSurfaceApplyAffine`_ and
    `FreeSurfer2CaretConvertAndRegisterNonlinear`_), although the the HCP
    may not zero out the offset.

    GIFTI files with ``midthickness``/``graymid`` in the name are also updated
    to include the following metadata entries::

        {
            AnatomicalStructureSecondary: MidThickness,
            GeometricType: Anatomical
        }

    This interface is intended to be applied uniformly to GIFTI surface files
    generated from the ``?h.white``/``?h.smoothwm`` and ``?h.pial`` surfaces,
    as well as externally-generated ``?h.midthickness``/``?h.graymid`` files.
    In principle, this should apply safely to any other surface, although it is
    less relevant to surfaces that don't describe an anatomical structure.

    .. _AlgorithmSurfaceApplyAffine: https://github.com/Washington-University/workbench\
/blob/1b79e56/src/Algorithms/AlgorithmSurfaceApplyAffine.cxx#L73-L91

    .. _FreeSurfer2CaretConvertAndRegisterNonlinear: https://github.com/Washington-University/\
Pipelines/blob/ae69b9a/PostFreeSurfer/scripts/FreeSurfer2CaretConvertAndRegisterNonlinear.sh\
#L147-154

    """

    input_spec = _NormalizeSurfInputSpec
    output_spec = _NormalizeSurfOutputSpec

    def _run_interface(self, runtime):
        transform_file = self.inputs.transform_file
        if not isdefined(transform_file):
            transform_file = None
        self._results["out_file"] = normalize_surfs(
            self.inputs.in_file, transform_file, newpath=runtime.cwd
        )
        return runtime


class _Path2BIDSInputSpec(BaseInterfaceInputSpec):
    in_file = File(mandatory=True, desc="input GIFTI file")


class _Path2BIDSOutputSpec(DynamicTraitedSpec):
    extension = traits.Str()


class Path2BIDS(SimpleInterface):
    """
    Extract BIDS entities from paths using a pattern.

    Default pattern is given for Gifti surfaces.

    >>> Path2BIDS(in_file='_fix_surfs0/rh.pial.surf.gii').run().outputs
    <BLANKLINE>
    extension = .surf.gii
    hemi = R
    suffix = pial
    <BLANKLINE>

    >>> Path2BIDS(in_file='_fix_surfs0/rh.pial.gii').run().outputs
    <BLANKLINE>
    extension = .gii
    hemi = R
    suffix = pial
    <BLANKLINE>

    >>> Path2BIDS(in_file='_fix_surfs0/rh.smoothwm_converted.gii').run().outputs
    <BLANKLINE>
    extension = .gii
    hemi = R
    suffix = smoothwm
    <BLANKLINE>

    >>> Path2BIDS(in_file='_fix_surfs0/rh.smoothwm_converted.func.gii').run().outputs
    <BLANKLINE>
    extension = .func.gii
    hemi = R
    suffix = smoothwm
    <BLANKLINE>

    """

    input_spec = _Path2BIDSInputSpec
    output_spec = _Path2BIDSOutputSpec
    _pattern = re.compile(
        r"(?P<hemi>[lr])h.(?P<suffix>(wm|smoothwm|pial|midthickness|"
        r"inflated|vinflated|sphere|flat))[\w\d_-]*(?P<extprefix>\.\w+)?"
    )
    _excluded = ("extprefix",)

    def __init__(self, pattern=None, **inputs):
        """Initialize the interface."""
        super().__init__(**inputs)
        if pattern:
            self._pattern = re.compile(pattern)

    def _outputs(self):
        outputs = self.output_spec()
        outputs.trait_set(
            trait_change_notify=False,
            **{
                entity: Undefined
                for entity in self._pattern.groupindex.keys()
                if entity not in self._excluded
            },
        )
        return outputs

    def _run_interface(self, runtime):
        in_file = Path(self.inputs.in_file)
        extension = "".join(in_file.suffixes[-((in_file.suffixes[-1] == ".gz") + 1):])
        info = self._pattern.match(in_file.name[: -len(extension)]).groupdict()
        self._results["extension"] = f"{info.pop('extprefix', None) or ''}{extension}"
        self._results.update(info)
        if "hemi" in self._results:
            self._results["hemi"] = self._results["hemi"].upper()
        return runtime


class _GiftiNameSourceInputSpec(BaseInterfaceInputSpec):
    in_file = File(mandatory=True, exists=True, desc="input GIFTI file")
    pattern = traits.Str(
        mandatory=True, desc='input file name pattern (must capture named group "LR")'
    )
    template = traits.Str(mandatory=True, desc="output file name template")
    template_kwargs = traits.Dict(desc="additional template keyword value pairs")


class _GiftiNameSourceOutputSpec(TraitedSpec):
    out_name = traits.Str(desc="(partial) filename formatted according to template")


class GiftiNameSource(SimpleInterface):
    r"""
    Construct a new filename for a GIFTI file.

    Construct a new filename based on an input filename, a matching pattern,
    and a related template, with optionally additional keywords.

    This interface is intended for use with GIFTI files, to generate names
    conforming to Section 9.0 of the `GIFTI Standard`_.

    Patterns are expected to have named groups, including one named "LR" that
    matches "l" or "r".
    These groups must correspond to named format elements in the template.

    .. testsetup::

    >>> open('lh.pial.gii', 'w').close()
    >>> open('rh.fsaverage.gii', 'w').close()

    .. doctest::

    >>> surf_namer = GiftiNameSource()
    >>> surf_namer.inputs.pattern = r'(?P<LR>[lr])h.(?P<surf>\w+).gii'
    >>> surf_namer.inputs.template = r'{surf}.{LR}.surf'
    >>> surf_namer.inputs.in_file = 'lh.pial.gii'
    >>> res = surf_namer.run()
    >>> res.outputs.out_name
    'pial.L.surf'

    >>> func_namer = GiftiNameSource()
    >>> func_namer.inputs.pattern = r'(?P<LR>[lr])h.(?P<space>\w+).gii'
    >>> func_namer.inputs.template = r'space-{space}.{LR}.func'
    >>> func_namer.inputs.in_file = 'rh.fsaverage.gii'
    >>> res = func_namer.run()
    >>> res.outputs.out_name
    'space-fsaverage.R.func'

    >>> namer = GiftiNameSource()
    >>> namer.inputs.pattern = r'(?P<LR>[lr])h.(?P<space>\w+).gii'
    >>> namer.inputs.template = r'space-{space}_density-{density}_hemi-{LR}.func'
    >>> namer.inputs.in_file = 'rh.fsaverage.gii'
    >>> namer.inputs.template_kwargs = {'density': '10k'}
    >>> res = namer.run()
    >>> res.outputs.out_name
    'space-fsaverage_density-10k_hemi-R.func'

    .. testcleanup::

    >>> import os
    >>> os.unlink('lh.pial.gii')
    >>> os.unlink('rh.fsaverage.gii')

    .. _GIFTI Standard: https://www.nitrc.org/frs/download.php/2871/GIFTI_Surface_Format.pdf
    """
    input_spec = _GiftiNameSourceInputSpec
    output_spec = _GiftiNameSourceOutputSpec

    def _run_interface(self, runtime):
        in_format = re.compile(self.inputs.pattern)
        in_file = os.path.basename(self.inputs.in_file)
        info = in_format.match(in_file).groupdict()
        info["LR"] = info["LR"].upper()
        if self.inputs.template_kwargs:
            info.update(self.inputs.template_kwargs)
        filefmt = self.inputs.template
        self._results["out_name"] = filefmt.format(**info)
        return runtime


class _GiftiSetAnatomicalStructureInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        mandatory=True, exists=True, desc='GIFTI file beginning with "lh." or "rh."'
    )


class _GiftiSetAnatomicalStructureOutputSpec(TraitedSpec):
    out_file = File(desc="output file with updated AnatomicalStructurePrimary entry")


class GiftiSetAnatomicalStructure(SimpleInterface):
    """
    Set AnatomicalStructurePrimary attribute of GIFTI image based on filename.

    For files that begin with ``lh.`` or ``rh.``, update the metadata to
    include::

        {
            AnatomicalStructurePrimary: (CortexLeft | CortexRight),
        }

    If ``AnatomicalStructurePrimary`` is already set, this function has no
    effect.

    """

    input_spec = _GiftiSetAnatomicalStructureInputSpec
    output_spec = _GiftiSetAnatomicalStructureOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        if any(nvpair.name == "AnatomicalStruturePrimary" for nvpair in img.meta.data):
            out_file = self.inputs.in_file
        else:
            fname = os.path.basename(self.inputs.in_file)
            if fname[:3] in ("lh.", "rh."):
                asp = "CortexLeft" if fname[0] == "l" else "CortexRight"
            else:
                raise ValueError(
                    "AnatomicalStructurePrimary cannot be derived from filename"
                )
            img.meta.data.insert(
                0, nb.gifti.GiftiNVPairs("AnatomicalStructurePrimary", asp)
            )
            out_file = os.path.join(runtime.cwd, fname)
            img.to_filename(out_file)
        self._results["out_file"] = out_file
        return runtime


class _GiftiToCSVInputSpec(BaseInterfaceInputSpec):
    in_file = File(mandatory=True, exists=True, desc="GIFTI file")
    itk_lps = traits.Bool(False, usedefault=True, desc="flip XY axes")


class _GiftiToCSVOutputSpec(TraitedSpec):
    out_file = File(desc="output csv file")


class GiftiToCSV(SimpleInterface):
    """Converts GIfTI files to CSV to make them ammenable to use with
    ``antsApplyTransformsToPoints``."""

    input_spec = _GiftiToCSVInputSpec
    output_spec = _GiftiToCSVOutputSpec

    def _run_interface(self, runtime):
        gii = nb.load(self.inputs.in_file)
        data = gii.darrays[0].data

        if self.inputs.itk_lps:  # ITK: flip X and Y around 0
            data[:, :2] *= -1

        # antsApplyTransformsToPoints requires 5 cols with headers
        csvdata = np.hstack((data, np.zeros((data.shape[0], 3))))

        out_file = fname_presuffix(
            self.inputs.in_file, newpath=runtime.cwd, use_ext=False, suffix="points.csv"
        )
        np.savetxt(
            out_file,
            csvdata,
            delimiter=",",
            header="x,y,z,t,label,comment",
            fmt=["%.5f"] * 4 + ["%d"] * 2,
        )
        self._results["out_file"] = out_file
        return runtime


class _CSVToGiftiInputSpec(BaseInterfaceInputSpec):
    in_file = File(mandatory=True, exists=True, desc="CSV file")
    gii_file = File(mandatory=True, exists=True, desc="reference GIfTI file")
    itk_lps = traits.Bool(False, usedefault=True, desc="flip XY axes")


class _CSVToGiftiOutputSpec(TraitedSpec):
    out_file = File(desc="output GIfTI file")


class CSVToGifti(SimpleInterface):
    """Converts CSV files back to GIfTI, after moving vertices with
    ``antsApplyTransformToPoints``."""

    input_spec = _CSVToGiftiInputSpec
    output_spec = _CSVToGiftiOutputSpec

    def _run_interface(self, runtime):
        gii = nb.load(self.inputs.gii_file)
        data = np.loadtxt(
            self.inputs.in_file, delimiter=",", skiprows=1, usecols=(0, 1, 2)
        )

        if self.inputs.itk_lps:  # ITK: flip X and Y around 0
            data[:, :2] *= -1

        gii.darrays[0].data = data[:, :3].astype(gii.darrays[0].data.dtype)
        out_file = fname_presuffix(
            self.inputs.gii_file, newpath=runtime.cwd, suffix=".transformed"
        )
        gii.to_filename(out_file)
        self._results["out_file"] = out_file
        return runtime


class _SurfacesToPointCloudInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(
        File(exists=True), mandatory=True, desc="input GIfTI files"
    )
    out_file = File("pointcloud.ply", usedefault=True, desc="output file name")


class _SurfacesToPointCloudOutputSpec(TraitedSpec):
    out_file = File(desc="output pointcloud in PLY format")


class SurfacesToPointCloud(SimpleInterface):
    """Converts multiple surfaces into a pointcloud with corresponding normals
    to then apply Poisson reconstruction"""

    input_spec = _SurfacesToPointCloudInputSpec
    output_spec = _SurfacesToPointCloudOutputSpec

    def _run_interface(self, runtime):
        from pathlib import Path

        giis = [nb.load(g) for g in self.inputs.in_files]
        vertices = np.vstack([g.darrays[0].data for g in giis])
        norms = np.vstack(
            [vertex_normals(g.darrays[0].data, g.darrays[1].data) for g in giis]
        )
        out_file = Path(self.inputs.out_file).resolve()
        pointcloud2ply(vertices, norms, out_file=out_file)
        self._results["out_file"] = str(out_file)
        return runtime


class _PoissonReconInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr="--in %s",
        desc="input PLY pointcloud (vertices + normals)",
    )
    out_file = File(
        argstr="--out %s",
        keep_extension=True,
        name_source=["in_file"],
        name_template="%s_avg",
        desc="output PLY triangular mesh",
    )


class _PoissonReconOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output PLY triangular mesh")


class PoissonRecon(CommandLine):
    """Runs Poisson Reconstruction on a cloud of points + normals
    given in PLY format.
    See https://github.com/mkazhdan/PoissonRecon
    """

    input_spec = _PoissonReconInputSpec
    output_spec = _PoissonReconOutputSpec
    _cmd = "PoissonRecon"


class _PLYtoGiftiInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="input PLY file")
    surf_key = traits.Str(mandatory=True, desc="reference GIfTI file")


class _PLYtoGiftiOutputSpec(TraitedSpec):
    out_file = File(desc="output GIfTI file")


class PLYtoGifti(SimpleInterface):
    """Convert surfaces from PLY to GIfTI"""

    input_spec = _PLYtoGiftiInputSpec
    output_spec = _PLYtoGiftiOutputSpec

    def _run_interface(self, runtime):
        from pathlib import Path

        meta = {
            "GeometricType": "Anatomical",
            "VolGeomWidth": "256",
            "VolGeomHeight": "256",
            "VolGeomDepth": "256",
            "VolGeomXsize": "1.0",
            "VolGeomYsize": "1.0",
            "VolGeomZsize": "1.0",
            "VolGeomX_R": "-1.0",
            "VolGeomX_A": "0.0",
            "VolGeomX_S": "0.0",
            "VolGeomY_R": "0.0",
            "VolGeomY_A": "0.0",
            "VolGeomY_S": "-1.0",
            "VolGeomZ_R": "0.0",
            "VolGeomZ_A": "1.0",
            "VolGeomZ_S": "0.0",
            "VolGeomC_R": "0.0",
            "VolGeomC_A": "0.0",
            "VolGeomC_S": "0.0",
        }
        meta["AnatomicalStructurePrimary"] = "Cortex%s" % (
            "Left" if self.inputs.surf_key.startswith("lh") else "Right"
        )
        meta["AnatomicalStructureSecondary"] = SECONDARY_ANAT_STRUC[
            self.inputs.surf_key.split(".")[-1]
        ]
        meta["Name"] = "%s_average.gii" % self.inputs.surf_key

        out_file = Path(runtime.cwd) / meta["Name"]
        out_file = ply2gii(self.inputs.in_file, meta, out_file=out_file)
        self._results["out_file"] = str(out_file)
        return runtime


class _UnzipJoinedSurfacesInputSpec(BaseInterfaceInputSpec):
    in_files = traits.List(
        InputMultiPath(File(exists=True), mandatory=True, desc="input GIfTI files")
    )


class _UnzipJoinedSurfacesOutputSpec(TraitedSpec):
    out_files = traits.List(
        OutputMultiPath(File(exists=True), desc="output pointcloud in PLY format")
    )
    surf_keys = traits.List(traits.Str, desc="surface identifier keys")


class UnzipJoinedSurfaces(SimpleInterface):
    """Unpack surfaces by identifier keys"""

    input_spec = _UnzipJoinedSurfacesInputSpec
    output_spec = _UnzipJoinedSurfacesOutputSpec

    def _run_interface(self, runtime):
        from pathlib import Path

        groups = defaultdict(list)
        in_files = [it for items in self.inputs.in_files for it in items]

        for f in in_files:
            bname = Path(f).name
            groups[bname.split("_")[0]].append(f)

        self._results["out_files"] = [sorted(els) for els in groups.values()]
        self._results["surf_keys"] = list(groups.keys())

        return runtime


def normalize_surfs(in_file, transform_file, newpath=None):
    """
    Re-center GIFTI coordinates to fit align to native T1w space.

    For midthickness surfaces, add MidThickness metadata

    Coordinate update based on:
    https://github.com/Washington-University/workbench/blob/1b79e56/src/Algorithms/AlgorithmSurfaceApplyAffine.cxx#L73-L91
    and
    https://github.com/Washington-University/Pipelines/blob/ae69b9a/PostFreeSurfer/scripts/FreeSurfer2CaretConvertAndRegisterNonlinear.sh#L147
    """

    img = nb.load(in_file)
    transform = load_transform(transform_file)
    pointset = img.get_arrays_from_intent("NIFTI_INTENT_POINTSET")[0]
    coords = pointset.data.T
    c_ras_keys = ("VolGeomC_R", "VolGeomC_A", "VolGeomC_S")
    ras = np.array([[float(pointset.metadata[key])] for key in c_ras_keys])
    ones = np.ones((1, coords.shape[1]), dtype=coords.dtype)
    # Apply C_RAS translation to coordinates, then transform
    pointset.data = transform.dot(np.vstack((coords + ras, ones)))[:3].T.astype(
        coords.dtype
    )

    secondary = nb.gifti.GiftiNVPairs("AnatomicalStructureSecondary", "MidThickness")
    geom_type = nb.gifti.GiftiNVPairs("GeometricType", "Anatomical")
    has_ass = has_geo = False
    for nvpair in pointset.meta.data:
        # Remove C_RAS translation from metadata to avoid double-dipping in FreeSurfer
        if nvpair.name in c_ras_keys:
            nvpair.value = "0.000000"
        # Check for missing metadata
        elif nvpair.name == secondary.name:
            has_ass = True
        elif nvpair.name == geom_type.name:
            has_geo = True
    fname = os.path.basename(in_file)
    # Update metadata for MidThickness/graymid surfaces
    if "midthickness" in fname.lower() or "graymid" in fname.lower():
        if not has_ass:
            pointset.meta.data.insert(1, secondary)
        if not has_geo:
            pointset.meta.data.insert(2, geom_type)

    if newpath is not None:
        newpath = os.getcwd()
    out_file = os.path.join(newpath, fname)
    img.to_filename(out_file)
    return out_file


def load_transform(fname):
    """Load affine transform from file

    Parameters
    ----------
    fname : str or None
        Filename of an LTA or FSL-style MAT transform file.
        If ``None``, return an identity transform

    Returns
    -------
    affine : (4, 4) numpy.ndarray
    """
    if fname is None:
        return np.eye(4)

    if fname.endswith(".mat"):
        return np.loadtxt(fname)
    elif fname.endswith(".lta"):
        with open(fname, "rb") as fobj:
            for line in fobj:
                if line.startswith(b"1 4 4"):
                    break
            lines = fobj.readlines()[:4]
        return np.genfromtxt(lines)

    raise ValueError("Unknown transform type; pass FSL (.mat) or LTA (.lta)")


def vertex_normals(vertices, faces):
    """Calculates the normals of a triangular mesh"""

    def normalize_v3(arr):
        """ Normalize a numpy array of 3 component vectors shape=(n,3) """
        lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
        arr /= lens[:, np.newaxis]

    tris = vertices[faces]
    facenorms = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    normalize_v3(facenorms)

    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    norm[faces[:, 0]] += facenorms
    norm[faces[:, 1]] += facenorms
    norm[faces[:, 2]] += facenorms
    normalize_v3(norm)
    return norm


def pointcloud2ply(vertices, normals, out_file=None):
    """Converts the file to PLY format"""
    from pathlib import Path
    import pandas as pd
    from pyntcloud import PyntCloud

    df = pd.DataFrame(np.hstack((vertices, normals)))
    df.columns = ["x", "y", "z", "nx", "ny", "nz"]
    cloud = PyntCloud(df)

    if out_file is None:
        out_file = Path("pointcloud.ply").resolve()

    cloud.to_file(str(out_file))
    return out_file


def ply2gii(in_file, metadata, out_file=None):
    """Convert from ply to GIfTI"""
    from pathlib import Path
    from numpy import eye
    from nibabel.gifti import (
        GiftiMetaData,
        GiftiCoordSystem,
        GiftiImage,
        GiftiDataArray,
    )
    from pyntcloud import PyntCloud

    in_file = Path(in_file)
    surf = PyntCloud.from_file(str(in_file))

    # Update centroid metadata
    metadata.update(
        zip(
            ("SurfaceCenterX", "SurfaceCenterY", "SurfaceCenterZ"),
            ["%.4f" % c for c in surf.centroid],
        )
    )

    # Prepare data arrays
    da = (
        GiftiDataArray(
            data=surf.xyz.astype("float32"),
            datatype="NIFTI_TYPE_FLOAT32",
            intent="NIFTI_INTENT_POINTSET",
            meta=GiftiMetaData.from_dict(metadata),
            coordsys=GiftiCoordSystem(xform=eye(4), xformspace=3),
        ),
        GiftiDataArray(
            data=surf.mesh.values,
            datatype="NIFTI_TYPE_INT32",
            intent="NIFTI_INTENT_TRIANGLE",
            coordsys=None,
        ),
    )
    surfgii = GiftiImage(darrays=da)

    if out_file is None:
        out_file = fname_presuffix(
            in_file.name, suffix=".gii", use_ext=False, newpath=str(Path.cwd())
        )

    surfgii.to_filename(str(out_file))
    return out_file


def get_gii_meta(in_file):
    from nibabel import load

    if isinstance(in_file, list):
        in_file = in_file[0]
    gii = load(in_file)
    return gii.darrays[0].meta.metadata
