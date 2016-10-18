import os

import nibabel as nib
import numpy as np

from niworkflows.data import get_mni_template
from niworkflows.viz.plot_utils import plot_segmentation

def test_plot_segmentation():
    anat_file = os.path.join(get_mni_template(), 'MNI152_T1_1mm.nii.gz')
    image = nib.load(anat_file)
    data = image.get_data()
    for x in np.nditer(data, op_flags=['readwrite']):
        if x > 1000:
            x = 1
        else:
            x = 0
    new_image = nib.Nifti1Image(data, image.get_affine())
    new_image_filename = 'new_image.nii'
    plot_filename = 'plot.png'
    nib.save(new_image, new_image_filename)
    plot_segmentation(anat_file, new_image_filename, plot_filename,
                      cut_coords=None)
    assert os.path.isfile(new_image_filename) == True
    os.remove(new_image_filename)
    os.remove(plot_filename)
