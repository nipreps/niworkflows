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
"""
Supercharging Nipype's workflow engine.

Add special features to the Nipype's vanilla workflows
"""
from nipype.pipeline import engine as pe


class LiterateWorkflow(pe.Workflow):
    """Controls the setup and execution of a pipeline of processes."""

    def __init__(self, name, base_dir=None):
        """
        Create a workflow object.

        Parameters
        ----------
        name : alphanumeric :obj:`str`
            unique identifier for the workflow
        base_dir : :obj:`str`, optional
            path to workflow storage

        """
        super(LiterateWorkflow, self).__init__(name, base_dir)
        self.__desc__ = None
        self.__postdesc__ = None

    def visit_desc(self):
        """Build a citation boilerplate by visiting all workflows."""
        desc = []

        if self.__desc__:
            desc += [self.__desc__]

        for node in pe.utils.topological_sort(self._graph)[0]:
            if isinstance(node, LiterateWorkflow):
                add_desc = node.visit_desc()
                if add_desc not in desc:
                    desc.append(add_desc)

        if self.__postdesc__:
            desc += [self.__postdesc__]

        return "".join(desc)
