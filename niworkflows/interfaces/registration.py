# -*- coding: utf-8 -*-
"""
ReportCapableInterfaces for registration tools

"""
from nipype.interfaces import ants, fsl
from niworkflows.common import report as nrc


class FLIRTInputSpecRPT(nrc.RegistrationRCInputSpec,
                        fsl.preprocess.FLIRTInputSpec):
    pass

class FLIRTOutputSpecRPT(nrc.ReportCapableOutputSpec,
                         fsl.preprocess.FLIRTOutputSpec):
    pass

class FLIRTRPT(nrc.RegistrationRC, fsl.FLIRT):
    input_spec = FLIRTInputSpecRPT
    output_spec = FLIRTOutputSpecRPT

    # def _generate_report(self):
    #     ref = self.inputs.reference
    #     ref_image_name = '{}.svg'.format(ref)
    #     out = self.inputs.out_file
    #     out_image_name = '{}.svg'.format(out)

    #     plotting.plot_img(ref, output_file=ref_image_name)
    #     plotting.plot_img(out, output_file=out_image_name)

    #     with open(ref_image_name, 'r') as file_obj:
    #         ref_image = file_obj.readlines()
    #         ref_image = ''.join(ref_image[4:])

    #     with open(out_image_name, 'r') as file_obj:
    #         out_image = file_obj.readlines()
    #         out_image = ''.join(out_image[4:])

    #     nrc.save_html(
    #         template='overlay_3d_nrc.tpl',
    #         report_file_name=self.html_report,
    #         unique_string='flirt' + str(uuid.uuid4()),
    #         base_image=ref_image,
    #         overlay_image=out_image,
    #         inputs=self.inputs,
    #         outputs=self.aggregate_outputs(),
    #         title="FLIRT: Overlay of registered image on top of reference file"
    #     )

class ApplyXFMInputSpecRPT(nrc.RegistrationRCInputSpec,
                           fsl.preprocess.ApplyXfmInputSpec):
    pass

class ApplyXFMRPT(FLIRTRPT):
    ''' ApplyXFM is a wrapper around FLIRT. ApplyXFMRPT is a wrapper around FLIRTRPT.'''
    input_spec = ApplyXFMInputSpecRPT

class ANTSRegistrationInputSpecRPT(nrc.RegistrationRCInputSpec,
                                   ants.registration.RegistrationInputSpec):
    pass

class ANTSRegistrationOutputSpecRPT(nrc.ReportCapableOutputSpec,
                                    ants.registration.RegistrationOutputSpec):
    pass

class ANTSRegistrationRPT(nrc.RegistrationRC, ants.Registration):
    input_spec = ANTSRegistrationInputSpecRPT
    output_spec = ANTSRegistrationOutputSpecRPT
