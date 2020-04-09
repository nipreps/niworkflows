"""Visualization component for Jupyter Notebooks."""
from pathlib import Path
import numpy as np
import nibabel as nb
from .utils import compose_view, plot_registration, cuts_from_bbox


def display(
    fixed_image,
    moving_image,
    contour=None,
    cuts=None,
    fixed_label="F",
    moving_label="M",
):
    """Plot the flickering panels to show a registration process."""
    from IPython.display import SVG, display as _disp

    if isinstance(fixed_image, (str, Path)):
        fixed_image = nb.load(str(fixed_image))
    if isinstance(moving_image, (str, Path)):
        moving_image = nb.load(str(moving_image))

    if cuts is None:
        n_cuts = 7
        if contour is not None:
            if isinstance(contour, (str, Path)):
                contour = nb.load(str(contour))
            cuts = cuts_from_bbox(contour, cuts=n_cuts)
        else:
            hdr = fixed_image.header.copy()
            hdr.set_data_dtype("uint8")
            mask_nii = nb.Nifti1Image(
                np.ones(fixed_image.shape, dtype="uint8"), fixed_image.affine, hdr
            )
            cuts = cuts_from_bbox(mask_nii, cuts=n_cuts)

    # Call composer
    _disp(
        SVG(
            compose_view(
                plot_registration(
                    fixed_image,
                    "fixed-image",
                    estimate_brightness=True,
                    cuts=cuts,
                    label=fixed_label,
                    contour=contour,
                    compress=False,
                ),
                plot_registration(
                    moving_image,
                    "moving-image",
                    estimate_brightness=True,
                    cuts=cuts,
                    label=moving_label,
                    contour=contour,
                    compress=False,
                ),
            )
        )
    )
