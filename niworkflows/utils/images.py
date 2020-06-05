"""Utilities to manipulate images."""
import nibabel as nb
import numpy as np


def unsafe_write_nifti_header_and_data(fname, header, data):
    """Write header and data without any consistency checks or data munging

    This is almost always a bad idea, and you should not use this function
    without a battery of tests for your specific use case.

    If you're not using this for NIfTI files specifically, you're playing
    with Fortran-ordered fire.
    """
    # ImageOpener handles zips transparently
    with nb.openers.ImageOpener(fname, mode="wb") as fobj:
        header.write_to(fobj)
        # This function serializes one block at a time to reduce memory usage a bit
        # It assumes Fortran-ordered data.
        nb.volumeutils.array_to_file(data, fobj, offset=header.get_data_offset())


def set_consumables(header, dataobj):
    header.set_slope_inter(dataobj.slope, dataobj.inter)
    header.set_data_offset(dataobj.offset)


def overwrite_header(img, fname):
    """Rewrite file with only changes to the header

    The data block is copied without scaling, avoiding copies in memory.
    The header is checked against the target file to ensure that no changes
    to the size, offset or interpretation of the data block will result.

    This function will not respect calls to:

    * img.header.set_slope_inter()
    * img.header.set_data_shape()
    * img.header.set_data_offset()

    These will all be determined by img.dataobj, which must be an
    ArrayProxy.

    If the qform or sform are updated, the
    ``img.header.get_best_affine()`` method must match ``img.affine``,
    or your changes may be lost.

    The intended use of this method is for making small header fixups
    that do not change the data or affine, e.g.:

    >>> import nibabel as nb
    >>> img = nb.load(nifti_fname, mmap=False)
    >>> img.header.set_qform(*img.header.get_sform(coded=True))
    >>> img.header['descrip'] = b'Modified with some extremely finicky tooling'
    >>> overwrite_header(img, nifti_fname)

    This is a destructive operation, and the image object should be considered unusable
    after calling this function.

    This should only be called with an image loaded with ``mmap=False``,
    or else you risk getting a ``BusError``.

    """
    # Synchronize header and set fields that nibabel transfer from header to dataobj
    img.update_header()
    header = img.header
    dataobj = img.dataobj

    if getattr(img.dataobj, "_mmap", False):
        raise ValueError("Image loaded with `mmap=True`. Aborting unsafe operation.")

    set_consumables(header, dataobj)

    ondisk = nb.load(fname, mmap=False)

    errmsg = "Cannot overwrite header (reason: {}).".format
    if not isinstance(ondisk.header, img.header_class):
        raise ValueError(errmsg("inconsistent header objects"))

    if (
        ondisk.get_data_dtype() != img.get_data_dtype()
        or img.header.get_data_shape() != ondisk.shape
    ):
        raise ValueError(errmsg("data blocks are not the same size"))

    if img.header["vox_offset"] != ondisk.dataobj.offset:
        raise ValueError(errmsg("change in offset from start of file"))

    if not np.allclose(
        img.header["scl_slope"], ondisk.dataobj.slope, equal_nan=True
    ) or not np.allclose(img.header["scl_inter"], ondisk.dataobj.inter, equal_nan=True):
        raise ValueError(errmsg("change in scale factors"))

    data = np.asarray(dataobj.get_unscaled())
    img._dataobj = data  # Allow old dataobj to be garbage collected
    del ondisk, img, dataobj  # Drop everything we don't need, to be safe
    unsafe_write_nifti_header_and_data(fname, header, data)


def update_header_fields(fname, **kwargs):
    """ Adjust header fields """
    # No-op
    if not kwargs:
        return
    img = nb.load(fname, mmap=False)
    for field, value in kwargs.items():
        img.header[field] = value
    overwrite_header(img, fname)


def dseg_label(in_seg, label, newpath=None):
    """Extract a particular label from a discrete segmentation."""
    from pathlib import Path
    import nibabel as nb
    import numpy as np
    from nipype.utils.filemanip import fname_presuffix

    newpath = Path(newpath or ".")

    nii = nb.load(in_seg)
    data = np.int16(nii.dataobj) == label

    out_file = fname_presuffix(in_seg, suffix="_mask", newpath=str(newpath.absolute()))
    new = nii.__class__(data, nii.affine, nii.header)
    new.set_data_dtype(np.uint8)
    new.to_filename(out_file)
    return out_file


def resample_by_spacing(in_file, zooms, order=3, clip=True):
    """Regrid the input image to match the new zooms."""
    from pathlib import Path
    import numpy as np
    import nibabel as nb
    from scipy.ndimage import map_coordinates

    if isinstance(in_file, (str, Path)):
        in_file = nb.load(in_file)

    # Prepare output x-forms
    sform, scode = in_file.get_sform(coded=True)
    qform, qcode = in_file.get_qform(coded=True)

    hdr = in_file.header.copy()
    dtype = hdr.get_data_dtype()
    data = np.asanyarray(in_file.dataobj)
    zooms = np.array(zooms)

    # Calculate the factors to normalize voxel size to the specific zooms
    pre_zooms = np.array(in_file.header.get_zooms()[:3])

    # Calculate an affine aligned with cardinal axes, for simplicity
    card = nb.affines.from_matvec(np.diag(pre_zooms))
    extent = card[:3, :3].dot(np.array(in_file.shape[:3]))
    card[:3, 3] = -0.5 * extent

    # Cover the FoV with the new grid
    new_size = np.ceil(extent / zooms).astype(int)
    offset = (extent - np.diag(zooms).dot(new_size)) * 0.5
    new_card = nb.affines.from_matvec(np.diag(zooms), card[:3, 3] + offset)

    # Calculate the new indexes
    new_grid = np.array(
        np.meshgrid(
            np.arange(new_size[0]),
            np.arange(new_size[1]),
            np.arange(new_size[2]),
            indexing="ij",
        )
    ).reshape((3, -1))

    # Calculate the locations of the new samples, w.r.t. the original grid
    ijk = np.linalg.inv(card).dot(
        new_card.dot(np.vstack((new_grid, np.ones((1, new_grid.shape[1])))))
    )

    # Resample data in the new grid
    resampled = map_coordinates(
        data,
        ijk[:3, :],
        output=dtype,
        order=order,
        mode="constant",
        cval=0,
        prefilter=True,
    ).reshape(new_size)
    if clip:
        resampled = np.clip(resampled, a_min=data.min(), a_max=data.max())

    # Set new zooms
    hdr.set_zooms(zooms)

    # Get the original image's affine
    affine = in_file.affine.copy()
    # Determine rotations w.r.t. cardinal axis and eccentricity
    rot = affine.dot(np.linalg.inv(card))
    # Apply to the new cardinal, so that the resampling is consistent
    new_affine = rot.dot(new_card)

    if qcode != 0:
        hdr.set_qform(new_affine.dot(np.linalg.inv(affine).dot(qform)), code=int(qcode))
    if scode != 0:
        hdr.set_sform(new_affine.dot(np.linalg.inv(affine).dot(sform)), code=int(scode))
    if (scode, qcode) == (0, 0):
        hdr.set_qform(new_affine, code=1)
        hdr.set_sform(new_affine, code=1)

    # Create a new x-form affine, aligned with cardinal axes, 1mm3 and centered.
    return nb.Nifti1Image(resampled, new_affine, hdr)
