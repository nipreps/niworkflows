# -*- coding: utf-8 -*-
""" all tests """

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import unittest
from shutil import copy

import nibabel as nb
from nilearn import image
from nipype.utils.tmpdirs import InTemporaryDirectory

from niworkflows.data.getters import (get_mni_template_ras, get_ds003_downsampled,
                                      get_ants_oasis_template_ras)

from niworkflows.interfaces.registration import FLIRTRPT, RobustMNINormalizationRPT
from niworkflows.interfaces.segmentation import FASTRPT
from niworkflows.interfaces.masks import BETRPT, BrainExtractionRPT

MNI_DIR = get_mni_template_ras()
MNI_2MM = os.path.join(MNI_DIR, 'MNI152_T1_2mm.nii.gz')
DS003_DIR = get_ds003_downsampled()

def stage_artifacts(filename, new_filename):
    """ filename: the name of the file to be saved as an artifact.
        new_filename: what to call the artifact (which will be saved in the
       `/scratch` folder) """
    if os.getenv('SAVE_CIRCLE_ARTIFACTS', False) == "1":
        copy(filename, os.path.join('/scratch', new_filename))

class TestFLIRTRPT(unittest.TestCase):
    def test_FLIRTRPT(self):
        with InTemporaryDirectory():
            reference = os.path.join(MNI_DIR, 'MNI152_T1_1mm.nii.gz')
            moving = os.path.join(DS003_DIR, 'sub-01/anat/sub-01_T1w.nii.gz')
            flirt_rpt = FLIRTRPT(generate_report=True, in_file=moving,
                                 reference=reference)
            res = flirt_rpt.run()
            out_report = res.outputs.out_report

            stage_artifacts(out_report, 'testFLIRT.svg')

            self.assertTrue(os.path.isfile(out_report), 'HTML report exists at {}'
                            .format(out_report))

    def test_RobustMNINormalizationRPT(self):
        with InTemporaryDirectory():
            moving = os.path.join(DS003_DIR, 'sub-01/anat/sub-01_T1w.nii.gz')
            ants_rpt = RobustMNINormalizationRPT(
                generate_report=True, moving_image=moving, testing=True)
            res = ants_rpt.run()
            out_report = res.outputs.out_report

            if os.getenv('SAVE_CIRCLE_ARTIFACTS', False) == "1":
                copy(out_report, os.path.join('/scratch', 'testRobustMNINormalizationRPT.svg'))

            self.assertTrue(os.path.isfile(out_report), 'HTML report exists at {}'
                            .format(out_report))


#     #def test_applyxfm_wrapper(self):
#     #    self.test_known_file_out(ApplyXFMRPT)

class TestBETRPT(unittest.TestCase):
    ''' tests it using mni as in_file '''

    def test_generate_report(self):
        ''' test of BET's report under basic (output binary mask) conditions '''
        _smoke_test_report(BETRPT(in_file=MNI_2MM, generate_report=True, mask=True),
                           'testBET.html')

    def test_generate_report_from_4d(self):
        ''' if the in_file was 4d, it should be able to produce the same report
        anyway (using arbitrary volume) '''
        # makeshift 4d in_file
        mni_file = MNI_2MM
        mni_4d = image.concat_imgs([mni_file, mni_file, mni_file])
        mni_4d_file = os.path.join(os.getcwd(), 'mni_4d.nii.gz')
        nb.save(mni_4d, mni_4d_file)

        _smoke_test_report(BETRPT(in_file=mni_4d_file, generate_report=True, mask=True),
                           'testBET4d.html')

class TestBrainExtractionRPT(unittest.TestCase):
    ''' tests the report capable version of ANTS's BrainExtraction interface, using mni as input'''

    def test_generate_report(self):
        ''' test of BrainExtractionRPT under basic conditions:
                - dimension=3
                - use_floatingpoint_precision=1,
                - brain_template, brain_probability_mask, extraction_registration_mask from get_ants_oasis_template_ras()
        '''
        def _template_name(filename):
            return os.path.join(get_ants_oasis_template_ras(), filename)

        _smoke_test_report(BrainExtractionRPT(generate_report=True,
                                              dimension=3,
                                              use_floatingpoint_precision=1,
                                              anatomical_image=MNI_2MM,
                                              brain_template=_template_name('T_template0.nii.gz'),
                                              brain_probability_mask=_template_name('T_template0_BrainCerebellumProbabilityMask.nii.gz'),
                                              extraction_registration_mask=_template_name('T_template0_BrainCerebellumRegistrationMask.nii.gz'),
                                              out_prefix='testBrainExtractionRPT',
                                              debug=True), # run faster for testing purposes
                           'testANTSBrainExtraction.html')

def _smoke_test_report(report_interface, artifact_name):
    with InTemporaryDirectory():
        report_interface.run()
        out_report = report_interface.inputs.out_report
        stage_artifacts(out_report, artifact_name)
        unittest.TestCase.assertTrue(os.path.isfile(out_report), 'HTML report exists at {}'
                                     .format(out_report))

class TestFASTRPT(unittest.TestCase):
    ''' tests use mni as in_file '''

    def test_generate_report(self):
        ''' test of FAST's report under basic conditions '''

        bet_interface = BETRPT(in_file=MNI_2MM, mask=True)
        bet_interface.run()
        skullstripped = bet_interface.aggregate_outputs().out_file

        report_interface = FASTRPT(in_files=skullstripped, generate_report=True, no_bias=True,
                                   probability_maps=True, segments=True, out_basename='test')

        _smoke_test_report(report_interface, 'testFAST.html')
