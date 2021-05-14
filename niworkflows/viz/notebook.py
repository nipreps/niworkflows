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
