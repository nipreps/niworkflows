# -*- coding: utf-8 -*-
"""
ReportCapableInterfaces for segmentation tools


"""
from __future__ import absolute_import, division, print_function


import uuid
import nibabel as nb
from nilearn import plotting, image as nlimage
from nipype.interfaces import fsl
from niworkflows.common import report as nrc
from niworkflows.viz.utils import as_svg, save_html


class BETInputSpecRPT(nrc.ReportCapableInputSpec,
                      fsl.preprocess.BETInputSpec):
    pass

class BETOutputSpecRPT(nrc.ReportCapableOutputSpec,
                       fsl.preprocess.BETOutputSpec):
    pass

class BETRPT(report.ReportCapableInterface, fsl.BET):
    input_spec = BETInputSpecRPT
    output_spec = BETOutputSpecRPT

    N_SLICES = 3 # number of slices to display per dimension

    def _run_interface(self, runtime):
        if self.inputs.generate_report:
            self.inputs.mask = True

        return super(BETRPT, self)._run_interface(runtime)

    def _generate_report(self):
        ''' generates a report showing three orthogonal slices of an arbitrary
        volume of in_file, with the resulting binary brain mask overlaid '''

        def _xyz_svgs(plot_func, cut_coord_basis, cut_coords_num, plot_params):
            ''' plot_func: function that returns an image like nilearn's plotting functions
                cut_coord_basis: nii image for which to calculate cut coords
                plot_params: dict of common parameters to plot_func
            returns a string of html containing svgs'''
            svgs = []
            for display_mode in 'x', 'y', 'z':
                plot_params['cut_coords'] = plotting.find_cut_slices(nb.load(cut_coord_basis),
                                                                     direction=display_mode,
                                                                     n_cuts=cut_coords_num)
                plot_params['display_mode'] = display_mode
                image = plot_func(**plot_params)
                svgs.append(as_svg(image))
                image.close()
            return '<br />'.join(svgs)

        def _plot_overlay_over_anat(**plot_params):
            ''' plot_params: dict of params for plot_func '''
            image = plotting.plot_anat(**plot_params)
            image.add_contours(self.aggregate_outputs().mask_file, filled=False, colors='r',
                               levels=[0.5], alpha=1)
            return image

        def _3d_in_file():
            ''' if self.inputs.in_file is 3d, return it.
            if 4d, pick an arbitrary volume and return that '''
            in_file = nlimage.concat_imgs(self.inputs.in_file) # result is always "4d"
            return nlimage.index_img(in_file, 0)

        background_params = {'anat_img': _3d_in_file(),
                             'cmap': 'gray'}
        base_svgs = _xyz_svgs(plotting.plot_anat, self.aggregate_outputs().mask_file,
                              self.N_SLICES, background_params)
        overlay_svgs = _xyz_svgs(_plot_overlay_over_anat,
                                 self.aggregate_outputs().mask_file,
                                 self.N_SLICES, background_params)

        save_html(template='overlay_3d_report.tpl',
                  report_file_name=self.html_report,
                  unique_string='bet' + str(uuid.uuid4()),
                  base_image=base_svgs,
                  overlay_image=overlay_svgs,
                  inputs=self.inputs,
                  outputs=self.aggregate_outputs(),
                  title='BET: Outline (calculated from brain mask) over the input '
                        '(anatomical)')
