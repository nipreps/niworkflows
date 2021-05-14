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
"""Interfaces for handling spaces."""
from nipype.interfaces.base import (
    traits,
    TraitedSpec,
    BaseInterfaceInputSpec,
    SimpleInterface,
)


class _SpaceDataSourceInputSpec(BaseInterfaceInputSpec):
    in_tuple = traits.Tuple(
        (traits.Str, traits.Dict), mandatory=True, desc="a space declaration"
    )


class _SpaceDataSourceOutputSpec(TraitedSpec):
    space = traits.Str(desc="the space identifier, after dropping the cohort modifier.")
    cohort = traits.Str(desc="a cohort specifier")
    resolution = traits.Str(desc="a resolution specifier")
    density = traits.Str(desc="a density specifier")
    uid = traits.Str(desc="a unique identifier combining space specifications")


class SpaceDataSource(SimpleInterface):
    """
    Generate a Nipype interface from a Space specification.

    Example
    -------
    >>> SpaceDataSource(
    ...     in_tuple=('MNIPediatricAsym:cohort-2', {'res': 2, 'den': '91k'})).run().outputs
    <BLANKLINE>
    cohort = 2
    density = 91k
    resolution = 2
    space = MNIPediatricAsym
    uid = MNIPediatricAsym_cohort-2_res-2
    <BLANKLINE>

    >>> SpaceDataSource(
    ...     in_tuple=('MNIPediatricAsym:cohort-2', {'res': 'native', 'den': '91k'})).run().outputs
    <BLANKLINE>
    cohort = 2
    density = 91k
    resolution = native
    space = MNIPediatricAsym
    uid = MNIPediatricAsym_cohort-2_res-native
    <BLANKLINE>

    """

    input_spec = _SpaceDataSourceInputSpec
    output_spec = _SpaceDataSourceOutputSpec

    def _run_interface(self, runtime):
        from ..utils.spaces import format_reference, reference2dict

        self._results = reference2dict(self.inputs.in_tuple)
        self._results["uid"] = format_reference(self.inputs.in_tuple)
        return runtime
