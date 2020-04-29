# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from .confounds import ExpandModel, SpikeRegressors
from .masks import BETRPT as BET
from .segmentation import FASTRPT as FAST
from .registration import (
    FLIRTRPT as FLIRT,
    ApplyXFMRPT as ApplyXFM,
    RobustMNINormalizationRPT as RobustMNINormalization,
    ANTSRegistrationRPT as Registration,
    ANTSApplyTransformsRPT as ApplyTransforms,
    SimpleBeforeAfterRPT as SimpleBeforeAfter,
)
from .utils import CopyXForm, CopyHeader, NormalizeMotionParams, SanitizeImage
from .plotting import FMRISummary, CompCorVariancePlot, ConfoundsCorrelationPlot
