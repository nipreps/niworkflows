#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import os.path as op
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
import nibabel as nb
import numpy as np
import scipy.ndimage as nd

def main():
    parser = ArgumentParser(description='Normalize ICBM152 atlases',
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument('input_file', nargs=3, help='input file')
    args = parser.parse_args()

    imgs = [nb.load(f) for f in args.input_file]
    w = np.zeros_like(imgs[0].get_data())

    tpms = [] 
    for im in imgs:
        data = im.get_data()
        data -= data.min()
        data[data < 0] = 0
        data /= data.max()
        w += data
        tpms.append(data)

    hdr = imgs[0].get_header().copy()
    hdr.set_data_dtype(float)
    for tpm, in_file in zip(tpms, args.input_file):
        print(tpm.min())
        out_file = in_file.split('.')[0] + '_norm.nii.gz'
        # tpms[i][w>0] = tpms[i][w>0]/w[w>0]
        nb.Nifti1Image(tpm.astype(float), imgs[0].get_affine(), hdr).to_filename(out_file)
        

if __name__ == '__main__':
    main()

