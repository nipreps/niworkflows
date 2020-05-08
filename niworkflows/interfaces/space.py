# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
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
