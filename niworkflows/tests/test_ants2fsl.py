import os
import unittest

from nipype.utils.tmpdirs import InTemporaryDirectory

from niworkflows.func.utils import Ants2FSL

class TestAnts2FSL(unittest.TestCase):

    def test_Ants2FSL(self):
        with InTemporaryDirectory():
            cwd = os.getcwd()
            in_filename = os.path.join(cwd, 'in_file.csv')
            fp = open(in_filename, 'w+')
            fp.write("this line is ignored\n")
            fp.write(
                "0,-0.99918075422702,1.00028207678993,-7.41063731046199e-06,"
                "3.93289649449106e-05,1.24969530535555e-05,1.00021114616063,"
                "0.000233157514656132,-0.000195740079275366,-0.00033024846828514,"
                "0.999495881936253,-0.0168301940330171,-0.0549722774931478,"
                "0.202056708561567\n"
            )
            fp.close()
            a2f = Ants2FSL()
            a2f.inputs.in_file = in_filename
            a2f.run()
