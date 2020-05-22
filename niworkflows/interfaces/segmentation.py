# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""ReportCapableInterfaces for segmentation tools."""
import os

from nipype.interfaces.base import File, isdefined
from nipype.interfaces import fsl, freesurfer
from nipype.interfaces.mixins import reporting
from . import report_base as nrc
from .. import NIWORKFLOWS_LOG


class _FASTInputSpecRPT(nrc._SVGReportCapableInputSpec, fsl.preprocess.FASTInputSpec):
    pass


class _FASTOutputSpecRPT(
    reporting.ReportCapableOutputSpec, fsl.preprocess.FASTOutputSpec
):
    pass


class FASTRPT(nrc.SegmentationRC, fsl.FAST):
    input_spec = _FASTInputSpecRPT
    output_spec = _FASTOutputSpecRPT

    def _run_interface(self, runtime):
        if self.generate_report:
            self.inputs.segments = True

        return super(FASTRPT, self)._run_interface(runtime)

    def _post_run_hook(self, runtime):
        """ generates a report showing nine slices, three per axis, of an
        arbitrary volume of `in_files`, with the resulting segmentation
        overlaid """
        self._anat_file = self.inputs.in_files[0]
        outputs = self.aggregate_outputs(runtime=runtime)
        self._mask_file = outputs.tissue_class_map
        # We are skipping the CSF class because with combination with others
        # it only shows the skullstriping mask
        self._seg_files = outputs.tissue_class_files[1:]
        self._masked = False

        NIWORKFLOWS_LOG.info(
            "Generating report for FAST (in_files %s, "
            "segmentation %s, individual tissue classes %s).",
            self.inputs.in_files,
            outputs.tissue_class_map,
            outputs.tissue_class_files,
        )

        return super(FASTRPT, self)._post_run_hook(runtime)


class _ReconAllInputSpecRPT(
    nrc._SVGReportCapableInputSpec, freesurfer.preprocess.ReconAllInputSpec
):
    pass


class _ReconAllOutputSpecRPT(
    reporting.ReportCapableOutputSpec, freesurfer.preprocess.ReconAllOutputSpec
):
    pass


class ReconAllRPT(nrc.SurfaceSegmentationRC, freesurfer.preprocess.ReconAll):
    input_spec = _ReconAllInputSpecRPT
    output_spec = _ReconAllOutputSpecRPT

    def _post_run_hook(self, runtime):
        """ generates a report showing nine slices, three per axis, of an
        arbitrary volume of `in_files`, with the resulting segmentation
        overlaid """
        outputs = self.aggregate_outputs(runtime=runtime)
        self._anat_file = os.path.join(
            outputs.subjects_dir, outputs.subject_id, "mri", "brain.mgz"
        )
        self._contour = os.path.join(
            outputs.subjects_dir, outputs.subject_id, "mri", "ribbon.mgz"
        )
        self._masked = False

        NIWORKFLOWS_LOG.info(
            "Generating report for ReconAll (subject %s)", outputs.subject_id
        )

        return super(ReconAllRPT, self)._post_run_hook(runtime)


class _MELODICInputSpecRPT(nrc._SVGReportCapableInputSpec, fsl.model.MELODICInputSpec):
    out_report = File(
        "melodic_reportlet.svg",
        usedefault=True,
        desc="Filename for the visual" " report generated " "by Nipype.",
    )
    report_mask = File(
        desc="Mask used to draw the outline on the reportlet. "
        "If not set the mask will be derived from the data."
    )


class _MELODICOutputSpecRPT(
    reporting.ReportCapableOutputSpec, fsl.model.MELODICOutputSpec
):
    pass


class MELODICRPT(fsl.MELODIC):
    input_spec = _MELODICInputSpecRPT
    output_spec = _MELODICOutputSpecRPT
    _out_report = None

    def __init__(self, generate_report=False, **kwargs):
        super(MELODICRPT, self).__init__(**kwargs)
        self.generate_report = generate_report

    def _post_run_hook(self, runtime):
        # Run _post_run_hook of super class
        runtime = super(MELODICRPT, self)._post_run_hook(runtime)
        # leave early if there's nothing to do
        if not self.generate_report:
            return runtime

        NIWORKFLOWS_LOG.info("Generating report for MELODIC.")
        _melodic_dir = runtime.cwd
        if isdefined(self.inputs.out_dir):
            _melodic_dir = self.inputs.out_dir
        self._melodic_dir = os.path.abspath(_melodic_dir)

        self._out_report = self.inputs.out_report
        if not os.path.isabs(self._out_report):
            self._out_report = os.path.abspath(
                os.path.join(runtime.cwd, self._out_report)
            )

        mix = os.path.join(self._melodic_dir, "melodic_mix")
        if not os.path.exists(mix):
            NIWORKFLOWS_LOG.warning(
                "MELODIC outputs not found, assuming it didn't converge."
            )
            self._out_report = self._out_report.replace(".svg", ".html")
            snippet = "<h4>MELODIC did not converge, no output</h4>"
            with open(self._out_report, "w") as fobj:
                fobj.write(snippet)
            return runtime

        self._generate_report()
        return runtime

    def _list_outputs(self):
        try:
            outputs = super(MELODICRPT, self)._list_outputs()
        except NotImplementedError:
            outputs = {}
        if self._out_report is not None:
            outputs["out_report"] = self._out_report
        return outputs

    def _generate_report(self):
        from niworkflows.viz.utils import plot_melodic_components

        plot_melodic_components(
            melodic_dir=self._melodic_dir,
            in_file=self.inputs.in_files[0],
            tr=self.inputs.tr_sec,
            out_file=self._out_report,
            compress=self.inputs.compress_report,
            report_mask=self.inputs.report_mask,
        )


class _ICA_AROMAInputSpecRPT(
    nrc._SVGReportCapableInputSpec, fsl.aroma.ICA_AROMAInputSpec
):
    out_report = File(
        "ica_aroma_reportlet.svg",
        usedefault=True,
        desc="Filename for the visual" " report generated " "by Nipype.",
    )
    report_mask = File(
        desc="Mask used to draw the outline on the reportlet. "
        "If not set the mask will be derived from the data."
    )


class _ICA_AROMAOutputSpecRPT(
    reporting.ReportCapableOutputSpec, fsl.aroma.ICA_AROMAOutputSpec
):
    pass


class ICA_AROMARPT(reporting.ReportCapableInterface, fsl.ICA_AROMA):
    input_spec = _ICA_AROMAInputSpecRPT
    output_spec = _ICA_AROMAOutputSpecRPT

    def _generate_report(self):
        from niworkflows.viz.utils import plot_melodic_components

        plot_melodic_components(
            melodic_dir=self.inputs.melodic_dir,
            in_file=self.inputs.in_file,
            out_file=self.inputs.out_report,
            compress=self.inputs.compress_report,
            report_mask=self.inputs.report_mask,
            noise_components_file=self._noise_components_file,
        )

    def _post_run_hook(self, runtime):
        outputs = self.aggregate_outputs(runtime=runtime)
        self._noise_components_file = os.path.join(
            outputs.out_dir, "classified_motion_ICs.txt"
        )

        NIWORKFLOWS_LOG.info("Generating report for ICA AROMA")

        return super(ICA_AROMARPT, self)._post_run_hook(runtime)
