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
    slope, inter = header.get_slope_inter()
    offset = header.get_data_offset()
    header.set_slope_inter(dataobj.slope, dataobj.inter)
    header.set_data_offset(dataobj.offset)
    return slope, inter, offset


def restore_consumables(header, slope, inter, offset):
    header.set_slope_inter(slope, inter)
    header.set_data_offset(offset)


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
    >>> img = nb.load(fname)
    >>> img.header.set_qform(*img.header.get_sform(coded=True))
    >>> img.header['descrip'] = b'Modified with some extremely finicky tooling'
    >>> overwrite_header(img, fname)

    """
    # Synchronize header and set fields that nibabel transfer from header to dataobj
    img.update_header()
    slope, inter, offset = set_consumables(img.header, img.dataobj)
    existing_img = nb.load(fname)

    try:
        assert isinstance(existing_img.header, img.header_class)
        assert (slope, inter, offset) == set_consumables(existing_img.header, existing_img.dataobj)
        # Check that the data block should be the same size
        assert existing_img.get_data_dtype() == img.get_data_dtype()
        assert existing_img.header.get_data_shape() == img.header.get_data_shape()
        # At the same offset from the start of the file
        assert existing_img.header['vox_offset'] == img.header['vox_offset']
        # With the same scale factors
        assert np.allclose(existing_img.header['scl_slope'], img.header['scl_slope'],
                           equal_nan=True)
        assert np.allclose(existing_img.header['scl_inter'], img.header['scl_inter'],
                           equal_nan=True)
    except AssertionError as e:
        raise ValueError("Cannot write header without compromising data") from e
    else:
        dataobj = img.dataobj
        data = np.asarray(dataobj.get_unscaled() if nb.is_proxy(dataobj) else dataobj)
        unsafe_write_nifti_header_and_data(fname, img.header, data)
    finally:
        restore_consumables(img.header, slope, inter, offset)


def update_header_fields(fname, **kwargs):
    """ Adjust header fields """
    # No-op
    if not kwargs:
        return
    img = nb.load(fname)
    for field, value in kwargs.items():
        img.header[field] = value
    overwrite_header(img, fname)
