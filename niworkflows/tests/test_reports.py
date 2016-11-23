# -*- coding: utf-8 -*-
""" all tests """

from __future__ import absolute_import, division, print_function

import os
import unittest

import nibabel as nb
from nilearn import image
from nipype.utils.tmpdirs import InTemporaryDirectory

from niworkflows.data.getters import get_mni_template_ras

from niworkflows.interfaces.registration import FLIRTRPT
from niworkflows.interfaces.segmentation import BETRPT

MNI_DIR = get_mni_template_ras()

class TestFLIRTRPT(unittest.TestCase):
    def setUp(self):
        self.out_file = "test_flirt.nii.gz"

    def test_known_file_out(self, flirt=FLIRTRPT):
        with InTemporaryDirectory():
            reference = os.path.join(MNI_DIR, 'MNI152_T1_1mm.nii.gz')
            moving = os.path.join(MNI_DIR, 'MNI152_T2_1mm.nii.gz')
            flirt_rpt = flirt(generate_report=True, in_file=template,
                              reference=template)
            res = flirt_rpt.run()
            out_report = res.outputs.out_report
            self.assertTrue(os.path.isfile(out_report), 'HTML report exists at {}'
                            .format(out_report))

#     #def test_applyxfm_wrapper(self):
#     #    self.test_known_file_out(ApplyXFMRPT)


class TestBETRPT(unittest.TestCase):
    ''' tests it using mni as in_file '''

    def test_generate_report(self):
        ''' test of BET's report under basic (output binary mask) conditions '''
        self._smoke(BETRPT(in_file=os.path.join(MNI_DIR, 'MNI152_T1_2mm.nii.gz'),
                           generate_report=True))

    def test_generate_report_from_4d(self):
        ''' if the in_file was 4d, it should be able to produce the same report
        anyway (using arbitrary volume) '''
        # makeshift 4d in_file
        mni_file = os.path.join(MNI_DIR, 'MNI152_T1_2mm.nii.gz')
        mni_4d = image.concat_imgs([mni_file, mni_file, mni_file])
        mni_4d_file = os.path.join(os.getcwd(), 'mni_4d.nii.gz')
        nb.save(mni_4d, mni_4d_file)

        self._smoke(BETRPT(in_file=mni_4d_file, generate_report=True))

    def _smoke(self, bet_interface):
        with InTemporaryDirectory():
            out_report = bet_interface.run().outputs.out_report
            self.assertTrue(os.path.isfile(out_report), 'HTML report exists at {}'
                            .format(out_report))
