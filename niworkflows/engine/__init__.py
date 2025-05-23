# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
The fmriprep reporting engine for visual assessment
"""

from .splicer import splice_workflow, tag
from .workflows import LiterateWorkflow as Workflow
