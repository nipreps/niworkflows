# -*- coding: utf-8 -*-
""" all tests """

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import unittest
import pkg_resources as pkgr
from multiprocessing import cpu_count
from shutil import copy

import nibabel as nb
from nilearn import image
from niworkflows.nipype.utils.tmpdirs import InTemporaryDirectory

from niworkflows.data.getters import (get_mni_template_ras, get_ds003_downsampled,
                                      get_ants_oasis_template_ras)

from niworkflows.interfaces.segmentation import FASTRPT, ReconAllRPT
from niworkflows.interfaces.masks import BETRPT, BrainExtractionRPT, \
    SimpleShowMaskRPT

MNI_DIR = get_mni_template_ras()
MNI_2MM = os.path.join(MNI_DIR, 'MNI152_T1_2mm.nii.gz')
DS003_DIR = get_ds003_downsampled()

# Tests are linear, so don't worry about leaving space for a control thread
nthreads = min(8, cpu_count())

def stage_artifacts(filename, new_filename):
    """ filename: the name of the file to be saved as an artifact.
        new_filename: what to call the artifact (which will be saved in the
       `/scratch` folder) """
    save_artifacts = os.getenv('SAVE_CIRCLE_ARTIFACTS', False)
    if save_artifacts:
        copy(filename, os.path.join(save_artifacts, new_filename))


def _smoke_test_report(report_interface, artifact_name):
    with InTemporaryDirectory():
        report_interface.run()
        out_report = report_interface.inputs.out_report
        stage_artifacts(out_report, artifact_name)
        assert os.path.isfile(out_report), 'Report does not exist'


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


def _template_name(filename):
    return os.path.join(get_ants_oasis_template_ras(), filename)


class TestSimpleShowMaskRPT(unittest.TestCase):

    def test_generate_report(self):
        ''' test of SimpleShowMaskRPT's report '''
        _smoke_test_report(SimpleShowMaskRPT(background_file=_template_name('T_template0.nii.gz'),
                                             mask_file=_template_name('T_template0_BrainCerebellumRegistrationMask.nii.gz')),
                           'testSimpleShowMaskRPT.html')


class TestBrainExtractionRPT(unittest.TestCase):
    ''' tests the report capable version of ANTS's BrainExtraction interface, using mni as input'''

    def test_generate_report(self):
        ''' test of BrainExtractionRPT under basic conditions:
                - dimension=3
                - use_floatingpoint_precision=1,
                - brain_template, brain_probability_mask, extraction_registration_mask from get_ants_oasis_template_ras()
        '''

        _smoke_test_report(
            BrainExtractionRPT(
                generate_report=True,
                dimension=3,
                use_floatingpoint_precision=1,
                anatomical_image=MNI_2MM,
                brain_template=_template_name('T_template0.nii.gz'),
                brain_probability_mask=_template_name('T_template0_BrainCerebellumProbabilityMask.nii.gz'),
                extraction_registration_mask=_template_name('T_template0_BrainCerebellumRegistrationMask.nii.gz'),
                out_prefix='testBrainExtractionRPT',
                debug=True, # run faster for testing purposes
                num_threads=cpu_count()
            ),
            'testANTSBrainExtraction.html'
        )

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

    def test_generate_report_no_segments(self):
        ''' test of FAST's report under no segments conditions '''

        bet_interface = BETRPT(in_file=MNI_2MM, mask=True)
        bet_interface.run()
        skullstripped = bet_interface.aggregate_outputs().out_file

        report_interface = FASTRPT(in_files=skullstripped, generate_report=True, no_bias=True,
                                   probability_maps=True, out_basename='test')

        _smoke_test_report(report_interface, 'testFAST_no_segments.html')


class TestCompression(unittest.TestCase):

    def test_compression(self):
        ''' test if compression makes files smaller '''
        uncompressed_int = BETRPT(in_file=MNI_2MM, generate_report=True,
                                  mask=True, compress_report=False)
        uncompressed_int.run()
        uncompressed_report = uncompressed_int.inputs.out_report

        compressed_int = BETRPT(in_file=MNI_2MM, generate_report=True,
                                mask=True, compress_report=True)
        compressed_int.run()
        compressed_report = compressed_int.inputs.out_report

        size = int(os.stat(uncompressed_report).st_size)
        size_compress = int(os.stat(compressed_report).st_size)

        assert size >= size_compress, ('The uncompressed report is smaller (%d)'
                                       'than the compressed report (%d)' % (size, size_compress))


class TestReconAllRPT(unittest.TestCase):
    def test_generate_report(self):
        report_interface = ReconAllRPT(subject_id='fsaverage', directive='all',
                                       subjects_dir=os.getenv('SUBJECTS_DIR'),
                                       generate_report=True,
                                       out_report='test.svg')
        report_interface.mock_run = True
        _smoke_test_report(report_interface, 'testReconAll.svg')
