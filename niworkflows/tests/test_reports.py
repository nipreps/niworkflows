# -*- coding: utf-8 -*-
""" all tests """

from __future__ import absolute_import, division, print_function

import os
import unittest
from shutil import copy

import nibabel as nb
from nilearn import image
from nipype.utils.tmpdirs import InTemporaryDirectory
from nipype.interfaces import fsl

from niworkflows.data.getters import get_mni_template_ras, get_ds003_downsampled

from niworkflows.interfaces.registration import FLIRTRPT
from niworkflows.interfaces.segmentation import BETRPT, FASTRPT

MNI_DIR = get_mni_template_ras()
MNI_2MM = os.path.join(MNI_DIR, 'MNI152_T1_2mm.nii.gz')
DS003_DIR = get_ds003_downsampled()
@unittest.skip
class TestFLIRTRPT(unittest.TestCase):
    def setUp(self):
        self.out_file = "test_flirt.nii.gz"

    def test_known_file_out(self, flirt=FLIRTRPT):
        with InTemporaryDirectory():
            reference = os.path.join(MNI_DIR, 'MNI152_T1_1mm.nii.gz')
            moving = os.path.join(DS003_DIR, 'sub-01/anat/sub-01_T1w.nii.gz')
            flirt_rpt = flirt(generate_report=True, in_file=moving,
                              reference=reference)
            res = flirt_rpt.run()
            out_report = res.outputs.out_report

            if os.getenv('SAVE_CIRCLE_ARTIFACTS', False) == "1":
                copy(out_report, os.path.join('/scratch', 'testFLIRT.svg'))

            self.assertTrue(os.path.isfile(out_report), 'HTML report exists at {}'
                            .format(out_report))

#     #def test_applyxfm_wrapper(self):
#     #    self.test_known_file_out(ApplyXFMRPT)
@unittest.skip
class TestBETRPT(unittest.TestCase):
    ''' tests it using mni as in_file '''

    def test_generate_report(self):
        ''' test of BET's report under basic (output binary mask) conditions '''
        _smoke_test_report(BETRPT(in_file=MNI_2MM, generate_report=True, mask=True))

    def test_generate_report_from_4d(self):
        ''' if the in_file was 4d, it should be able to produce the same report
        anyway (using arbitrary volume) '''
        # makeshift 4d in_file
        mni_file = MNI_2MM
        mni_4d = image.concat_imgs([mni_file, mni_file, mni_file])
        mni_4d_file = os.path.join(os.getcwd(), 'mni_4d.nii.gz')
        nb.save(mni_4d, mni_4d_file)

        _smoke_test_report(BETRPT(in_file=mni_4d_file, generate_report=True, mask=True))

def _smoke_test_report(report_interface):
    with InTemporaryDirectory() as temp_dir:
        report_interface.run()
        out_report = report_interface.aggregate_outputs().out_report
        unittest.TestCase.assertTrue(os.path.isfile(out_report), 'HTML report exists at {}'
                        .format(out_report))

class TestFASTRPT(unittest.TestCase):
    ''' tests use mni as in_file '''

    def test_generate_report(self):
        ''' test of FAST's report under basic conditions '''

        _smoke_test_report(FASTRPT(in_files=MNI_2MM, generate_report=True, no_bias=True,
                                   probability_maps=True))

