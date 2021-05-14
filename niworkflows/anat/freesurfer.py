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
"""FreeSurfer-related workflows."""

from os import getenv
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import io as nio
from nipype.interfaces import freesurfer as fs

from ..interfaces.freesurfer import (
    MakeMidthickness,
    PatchedRobustRegister as RobustRegister,
)
from ..interfaces.surf import NormalizeSurf


def init_gifti_surface_wf(
    name="gifti_surface_wf", subjects_dir=getenv("SUBJECTS_DIR", None)
):
    """
    Build a Nipype workflow to prepare GIFTI surfaces from FreeSurfer.

    This workflow prepares GIFTI surfaces from a FreeSurfer subjects directory
    If midthickness (or graymid) surfaces do not exist, they are generated and
    saved to the subject directory as ``lh/rh.midthickness``.
    These, along with the gray/white matter boundary (``lh/rh.smoothwm``), pial
    sufaces (``lh/rh.pial``) and inflated surfaces (``lh/rh.inflated``) are
    converted to GIFTI files.
    Additionally, the vertex coordinates are :py:class:`recentered
    <smriprep.interfaces.NormalizeSurf>` to align with native T1w space.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from niworkflows.anat.freesurfer import init_gifti_surface_wf
            wf = init_gifti_surface_wf(subjects_dir="/tmp")

    Parameters
    ----------
    subjects_dir : str
        FreeSurfer's ``$SUBJECTS_DIR`` environment variable.
    name : str
        Name for the workflow hierarchy of Nipype

    Inputs
    ------
    in_t1w : str
        original (pre-``recon-all``), reference T1w image.
    subject_id : str
        FreeSurfer subject ID

    Outputs
    -------
    surfaces : list
        GIFTI surfaces for gray/white matter boundary, pial surface,
        midthickness (or graymid) surface, and inflated surfaces.
    surf_norm : list
        Normalized (re-centered) GIFTI surfaces aligned in native T1w
        space, corresponding to the ``surfaces`` output.
    fsnative_to_t1w_xfm : str
        LTA formatted affine transform file.

    """
    if subjects_dir is None:
        raise RuntimeError("``$SUBJECTS_DIR`` must be set")

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(["in_t1w", "subject_id"]), name="inputnode"
    )
    outputnode = pe.Node(
        niu.IdentityInterface(["surfaces", "surf_norm", "fsnative_to_t1w_xfm"]),
        name="outputnode",
    )

    fssource = pe.Node(
        nio.FreeSurferSource(subjects_dir=subjects_dir),
        name="fssource",
        run_without_submitting=True,
    )

    fsnative_2_t1_xfm = pe.Node(
        RobustRegister(auto_sens=True, est_int_scale=True), name="fsnative_2_t1_xfm"
    )

    midthickness = pe.MapNode(
        MakeMidthickness(thickness=True, distance=0.5, out_name="midthickness"),
        iterfield="in_file",
        name="midthickness",
    )

    save_midthickness = pe.Node(
        nio.DataSink(parameterization=False, base_directory=subjects_dir),
        name="save_midthickness",
        run_without_submitting=True,
    )

    surface_list = pe.Node(
        niu.Merge(4, ravel_inputs=True),
        name="surface_list",
        run_without_submitting=True,
    )
    fs_2_gii = pe.MapNode(
        fs.MRIsConvert(out_datatype="gii"), iterfield="in_file", name="fs_2_gii"
    )
    fix_surfs = pe.MapNode(NormalizeSurf(), iterfield="in_file", name="fix_surfs")

    # fmt: off
    workflow.connect([
        (inputnode, fssource, [("subject_id", "subject_id")]),
        (inputnode, save_midthickness, [("subject_id", "container")]),
        # Generate fsnative-to-T1w transform
        (inputnode, fsnative_2_t1_xfm, [("in_t1w", "target_file")]),
        (fssource, fsnative_2_t1_xfm, [("orig", "source_file")]),
        # Generate midthickness surfaces and save to FreeSurfer derivatives
        (fssource, midthickness, [("smoothwm", "in_file"), ("graymid", "graymid")]),
        (midthickness, save_midthickness, [("out_file", "surf.@graymid")]),
        # Produce valid GIFTI surface files (dense mesh)
        (fssource, surface_list, [
            ("smoothwm", "in1"), ("pial", "in2"), ("inflated", "in3"),
        ]),
        (save_midthickness, surface_list, [("out_file", "in4")]),
        (surface_list, fs_2_gii, [("out", "in_file")]),
        (fs_2_gii, fix_surfs, [("converted", "in_file")]),
        (fsnative_2_t1_xfm, fix_surfs, [("out_reg_file", "transform_file")]),
        (fsnative_2_t1_xfm, outputnode, [("out_reg_file", "fsnative_to_t1w_xfm")]),
        (fix_surfs, outputnode, [("out_file", "surf_norm")]),
        (fs_2_gii, outputnode, [("converted", "surfaces")]),
    ])
    # fmt: on

    return workflow
