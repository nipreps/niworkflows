# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
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
"""Extracting signals from NIfTI and CIFTI2 files."""
import numpy as np
import nibabel as nb


def _cifti_timeseries(dataset):
    """Extract timeseries from CIFTI2 dataset."""
    dataset = nb.load(dataset) if isinstance(dataset, str) else dataset

    if dataset.nifti_header.get_intent()[0] != "ConnDenseSeries":
        raise ValueError("Not a dense timeseries")

    matrix = dataset.header.matrix
    labels = {
        "CIFTI_STRUCTURE_CORTEX_LEFT": "CtxL",
        "CIFTI_STRUCTURE_CORTEX_RIGHT": "CtxR",
        "CIFTI_STRUCTURE_CEREBELLUM_LEFT": "CbL",
        "CIFTI_STRUCTURE_CEREBELLUM_RIGHT": "CbR",
    }
    seg = {label: [] for label in list(labels.values()) + ["Other"]}
    for bm in matrix.get_index_map(1).brain_models:
        label = (
            "Other" if bm.brain_structure not in labels else
            labels[bm.brain_structure]
        )
        seg[label] += list(range(
            bm.index_offset, bm.index_offset + bm.index_count
        ))

    return dataset.get_fdata(dtype="float32").T, seg


def _nifti_timeseries(
    dataset,
    segmentation=None,
    labels=("Ctx GM", "dGM", "WM+CSF", "Cb", "Crown"),
    remap_rois=False,
    lut=None,
):
    """Extract timeseries from NIfTI1/2 datasets."""
    dataset = nb.load(dataset) if isinstance(dataset, str) else dataset
    data = dataset.get_fdata(dtype="float32").reshape((-1, dataset.shape[-1]))

    if segmentation is None:
        return data, None

    # Open NIfTI and extract numpy array
    segmentation = nb.load(segmentation) if isinstance(segmentation, str) else segmentation
    segmentation = np.asanyarray(segmentation.dataobj, dtype=int).reshape(-1)

    remap_rois = remap_rois or (len(np.unique(segmentation[segmentation > 0])) > len(labels))

    # Map segmentation
    if remap_rois or lut is not None:
        if lut is None:
            lut = np.zeros((256,), dtype="uint8")
            lut[100:201] = 1  # Ctx GM
            lut[30:99] = 2    # dGM
            lut[1:11] = 3     # WM+CSF
            lut[255] = 4      # Cerebellum
        # Apply lookup table
        segmentation = lut[segmentation]

    fgmask = segmentation > 0
    segmentation = segmentation[fgmask]
    seg_dict = {}
    for i in np.unique(segmentation):
        seg_dict[labels[i - 1]] = np.argwhere(segmentation == i).squeeze()

    return data[fgmask], seg_dict
