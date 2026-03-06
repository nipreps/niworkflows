# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Report-capable registration interfaces not yet ported to NiReports."""

from nipype.interfaces.ants import registration, resampling
from nipype.interfaces.mixins import reporting

from ... import NIWORKFLOWS_LOG
from ..._deprecated import DeprecationError, moved_to_nireports_message
from ..fixes import (
    FixHeaderApplyTransforms as ApplyTransforms,
)
from ..fixes import (
    FixHeaderRegistration as Registration,
)
from ..norm import (
    SpatialNormalization,
    _SpatialNormalizationInputSpec,
    _SpatialNormalizationOutputSpec,
)
from . import base as nrb


class _SpatialNormalizationInputSpecRPT(
    nrb._SVGReportCapableInputSpec, _SpatialNormalizationInputSpec
):
    pass


class _SpatialNormalizationOutputSpecRPT(
    reporting.ReportCapableOutputSpec, _SpatialNormalizationOutputSpec
):
    pass


class SpatialNormalizationRPT(nrb.RegistrationRC, SpatialNormalization):
    input_spec = _SpatialNormalizationInputSpecRPT
    output_spec = _SpatialNormalizationOutputSpecRPT

    def _post_run_hook(self, runtime):
        # We need to dig into the internal ants.Registration interface
        self._fixed_image = self._get_ants_args()['fixed_image']
        if isinstance(self._fixed_image, (list, tuple)):
            self._fixed_image = self._fixed_image[0]  # get first item if list

        if self._get_ants_args().get('fixed_image_mask') is not None:
            self._fixed_image_mask = self._get_ants_args().get('fixed_image_mask')
        self._moving_image = self.aggregate_outputs(runtime=runtime).warped_image
        NIWORKFLOWS_LOG.info(
            'Report - setting fixed (%s) and moving (%s) images',
            self._fixed_image,
            self._moving_image,
        )

        return super()._post_run_hook(runtime)


class _ANTSRegistrationInputSpecRPT(
    nrb._SVGReportCapableInputSpec, registration.RegistrationInputSpec
):
    pass


class _ANTSRegistrationOutputSpecRPT(
    reporting.ReportCapableOutputSpec, registration.RegistrationOutputSpec
):
    pass


class ANTSRegistrationRPT(nrb.RegistrationRC, Registration):
    input_spec = _ANTSRegistrationInputSpecRPT
    output_spec = _ANTSRegistrationOutputSpecRPT

    def _post_run_hook(self, runtime):
        self._fixed_image = self.inputs.fixed_image[0]
        self._moving_image = self.aggregate_outputs(runtime=runtime).warped_image
        NIWORKFLOWS_LOG.info(
            'Report - setting fixed (%s) and moving (%s) images',
            self._fixed_image,
            self._moving_image,
        )

        return super()._post_run_hook(runtime)


class _ANTSApplyTransformsInputSpecRPT(
    nrb._SVGReportCapableInputSpec, resampling.ApplyTransformsInputSpec
):
    pass


class _ANTSApplyTransformsOutputSpecRPT(
    reporting.ReportCapableOutputSpec, resampling.ApplyTransformsOutputSpec
):
    pass


class ANTSApplyTransformsRPT(nrb.RegistrationRC, ApplyTransforms):
    input_spec = _ANTSApplyTransformsInputSpecRPT
    output_spec = _ANTSApplyTransformsOutputSpecRPT

    def _post_run_hook(self, runtime):
        self._fixed_image = self.inputs.reference_image
        self._moving_image = self.aggregate_outputs(runtime=runtime).output_image
        NIWORKFLOWS_LOG.info(
            'Report - setting fixed (%s) and moving (%s) images',
            self._fixed_image,
            self._moving_image,
        )

        return super()._post_run_hook(runtime)


_MOVED_REPORTLETS = {
    'ApplyTOPUPRPT': 'nireports.interfaces.reporting.registration.ApplyTOPUPRPT',
    'ApplyXFMRPT': 'nireports.interfaces.reporting.registration.ApplyXFMRPT',
    'BBRegisterRPT': 'nireports.interfaces.reporting.registration.BBRegisterRPT',
    'FLIRTRPT': 'nireports.interfaces.reporting.registration.FLIRTRPT',
    'FUGUERPT': 'nireports.interfaces.reporting.registration.FUGUERPT',
    'MRICoregRPT': 'nireports.interfaces.reporting.registration.MRICoregRPT',
    'ResampleBeforeAfterRPT': 'nireports.interfaces.reporting.registration.ResampleBeforeAfterRPT',
    'SimpleBeforeAfterRPT': 'nireports.interfaces.reporting.registration.SimpleBeforeAfterRPT',
}


def __getattr__(name):
    if name in _MOVED_REPORTLETS:
        raise DeprecationError(
            moved_to_nireports_message(
                f'niworkflows.interfaces.reportlets.registration.{name}',
                (_MOVED_REPORTLETS[name],),
            )
        )
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = [
    'ANTSApplyTransformsRPT',
    'ANTSRegistrationRPT',
    'SpatialNormalizationRPT',
]
