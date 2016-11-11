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

HARVARD_OXFORD_SUBC_WM = [1, 12]
HARVARD_OXFORD_SUBC_GM = [2, 13]
HARVARD_OXFORD_SUBC_BSTEM = 8


def main():
    parser = ArgumentParser(description='Normalize ICBM152 atlases',
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument('reference')
    parser.add_argument('output_file', help='output file')
    parser.add_argument('--template_path', default=os.getenv('FSL_HARVARD_OXFORD_TEMPLATE'))
    parser.add_argument('--brainmask')
    args = parser.parse_args()
    
    # Load subcortical segmentation
    subc_nii = nb.as_closest_canonical(
        nb.load(op.join(args.template_path, 'HarvardOxford-sub-maxprob-thr25-1mm.nii.gz')))
    subc = subc_nii.get_data()

    # Load cortical segmentation
    cort_nii = nb.as_closest_canonical(
        nb.load(op.join(args.template_path, 'HarvardOxford-cortl-maxprob-thr0-1mm.nii.gz')))
    cort = cort_nii.get_data()
    cort[cort > 0 ] += 100

    for label in HARVARD_OXFORD_SUBC_GM:
        subc[subc == label] = cort[subc == label]

    subc[(subc < 50) & (subc > 2)] += 30
    subc[subc == 33] = 3
    subc[subc == 44] = 4
    subc[subc == (HARVARD_OXFORD_SUBC_WM[1] + 30)] = 2
    subc[subc == (HARVARD_OXFORD_SUBC_BSTEM + 30)] = 5


    aff = subc_nii.get_affine()
    hdr = subc_nii.get_header()
    hdr.set_data_dtype(np.uint16)
    nb.Nifti1Image(subc.astype(np.uint16), aff, hdr).to_filename(args.output_file)

    ref = nb.as_closest_canonical(nb.load(args.brainmask)).get_data()
    padding = ((np.array(ref.shape) - subc.shape)).astype(int)
    if np.any(padding > 0):
        from subprocess import check_call
        check_call(['mri_convert', '-rl', args.reference, '-rt', 'nearest', args.output_file, args.output_file])
        sub_nii = nb.load(args.output_file)
        subc = sub_nii.get_data()
        aff = sub_nii.get_affine()
        hdr = sub_nii.get_header()

    if args.brainmask is not None:
        bmask = nb.as_closest_canonical(nb.load(args.brainmask)).get_data()
        cereb = np.zeros_like(bmask)
        cereb[subc == 0] = bmask[subc == 0]
        struc = nd.morphology.iterate_structure(nd.generate_binary_structure(3, 2), 2)
        cereb = nd.binary_opening(cereb, struc).astype(np.uint8)
        subc[cereb == 1] = 255        

    hdr.set_data_dtype(np.uint16)
    nb.Nifti1Image(subc.astype(np.uint16), aff, hdr).to_filename(args.output_file)


if __name__ == '__main__':
    main()

