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
    with nb.openers.ImageOpener(fname, mode='wb') as fobj:
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

    if getattr(img.dataobj, '_mmap', False):
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

    if img.header['vox_offset'] != ondisk.dataobj.offset:
        raise ValueError(errmsg("change in offset from start of file"))

    if (
        not np.allclose(img.header['scl_slope'], ondisk.dataobj.slope, equal_nan=True)
        or not np.allclose(img.header['scl_inter'], ondisk.dataobj.inter, equal_nan=True)
    ):
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
