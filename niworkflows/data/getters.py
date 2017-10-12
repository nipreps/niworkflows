#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Data grabbers
"""
from __future__ import print_function, division, absolute_import, unicode_literals

from ..data.utils import _get_dataset_dir, _fetch_file

OSF_PROJECT_URL = ('https://files.osf.io/v1/resources/fvuh8/providers/osfstorage/')
OSF_RESOURCES = {
    'brainweb': ('57f32b96b83f6901f194c3ca', '384263fbeadc8e2cca92ced98f224c4b'),
    'ds003_downsampled': ('57f328f6b83f6901ef94cf70', '5a558961c1eb5e5f162696d8afa956e8'),
    'mni_template': ('57f32ab29ad5a101fb77fd89', 'debfa882b8c301cd6d75dd769e73f727'),
    'mni_template_RAS': ('57f32a799ad5a101f977eb77', 'a4669f0e7acceae148bb39450b2b21b4'),
    'ants_oasis_template': ('57f32ae89ad5a101f977eb79', '34d39070b541c416333cc8b6c2fe993c'),
    'ants_oasis_template_ras': ('584123a29ad5a1020913609d', 'afa21f99c66ae1672320d8aa0408229a'),
    'ants_nki_template_ras': ('59cd90f46c613b02b3d79782', 'e5debaee65b8f2c8971577db1327e314'),
    'mni_epi': ('57fa09cdb83f6901d93623a0', '9df727e1f742ec55213480434b4c4811'),
    'mni152_nlin_sym_las': ('57fa7fc89ad5a101e635eeef', '9c4c0cad2a2e99d6799f01abf4107f5a'),
    'mni152_nlin_sym_ras': ('57fa7fd09ad5a101df35eed0', '65d64ad5a980da86e7d07d95b3ed2ccb'),
    'mni_icbm152_linear': ('580705eb594d9001ed622649', '72be639e92532def7caad75cb4058e83'),
    'mni_icbm152_nlin_asym_09c': ('580705089ad5a101f17944a9', '002f9bf24dc5c32de50c03f01fa539ec')
}

BIDS_EXAMPLES = {
    'BIDS-examples-1-1.0.0-rc3u5': (
        'https://github.com/chrisfilo/BIDS-examples-1/archive/1.0.0-rc3u5.tar.gz',
        '035fe54445c56eff5bd845ef3795fd56'),
    'BIDS-examples-1-enh-ds054': (
        'http://github.com/chrisfilo/BIDS-examples-1/archive/enh/ds054.zip',
        '56cee272860624924bc23efbe868acb7'),
}

# Map names of templates to OSF_RESOURCES keys
TEMPLATE_MAP = {
    'MNI152NLin2009cAsym': 'mni_icbm152_nlin_asym_09c',
}

def get_dataset(dataset_name, data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the BIDS-fied brainweb 1mm normal


    :param str data_dir: path of the data directory. Used to force data storage
        in a non-standard location.
    :param str url: download URL of the dataset. Overwrite the default URL.

    """
    file_id, md5 = OSF_RESOURCES[dataset_name]
    if url is None:
        url = '{}/{}'.format(OSF_PROJECT_URL, file_id)

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, verbose=verbose)

    if _fetch_file(url, data_dir, filetype='tar', resume=resume, verbose=verbose,
                   md5sum=md5):
        return data_dir
    else:
        return None

def get_brainweb_1mm_normal(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the BIDS-fied brainweb 1mm normal


    :param str data_dir: path of the data directory. Used to force data storage
        in a non-standard location.
    :param str url: download URL of the dataset. Overwrite the default URL.

    """
    return get_dataset('brainweb', data_dir, url, resume, verbose)

def get_ds003_downsampled(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the BIDS-fied ds003_downsampled


    :param str data_dir: path of the data directory. Used to force data storage
        in a non-standard location.
    :param str url: download URL of the dataset. Overwrite the default URL.

    """
    return get_dataset('ds003_downsampled', data_dir, url, resume, verbose)

def get_mni_template(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the necessary files from the mni template


    :param str data_dir: path of the data directory. Used to force data storage
        in a non-standard location.
    :param str url: download URL of the dataset. Overwrite the default URL.

    """
    return get_dataset('mni_template', data_dir, url, resume, verbose)

def get_mni_template_ras(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the necessary files from the mni template
    :param str data_dir: path of the data directory. Used to force data storage
        in a non-standard location.
    :param str url: download URL of the dataset. Overwrite the default URL.
    """
    return get_dataset('mni_template_RAS', data_dir, url, resume, verbose)

def get_mni_epi(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the necessary files from the mni template
    :param str data_dir: path of the data directory. Used to force data storage
        in a non-standard location.
    :param str url: download URL of the dataset. Overwrite the default URL.
    """
    return get_dataset('mni_epi', data_dir, url, resume, verbose)

def get_ants_oasis_template(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the necessary files from the ANTs template of the OASIS dataset.
    :param str data_dir: path of the data directory. Used to force data storage
        in a non-standard location.
    :param str url: download URL of the dataset. Overwrite the default URL.
    """
    return get_dataset('ants_oasis_template', data_dir, url, resume, verbose)

def get_ants_oasis_template_ras(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the necessary files from the ANTs template of the OASIS dataset.
    :param str data_dir: path of the data directory. Used to force data storage
        in a non-standard location.
    :param str url: download URL of the dataset. Overwrite the default URL.
    """
    return get_dataset('ants_oasis_template_ras', data_dir, url, resume, verbose)

def get_ants_nki_template_ras(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the necessary files from the ANTs template of the NKI dataset.
    :param str data_dir: path of the data directory. Used to force data storage
        in a non-standard location.
    :param str url: download URL of the dataset. Overwrite the default URL.
    """
    return get_dataset('ants_nki_template_ras', data_dir, url, resume, verbose)

def get_mni152_nlin_sym_las(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the necessary files from the mni template
    :param str data_dir: path of the data directory. Used to force data storage
        in a non-standard location.
    :param str url: download URL of the dataset. Overwrite the default URL.
    """
    return get_dataset('mni152_nlin_sym_las', data_dir, url, resume, verbose)

def get_mni152_nlin_sym_ras(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the necessary files from the mni template
    :param str data_dir: path of the data directory. Used to force data storage
        in a non-standard location.
    :param str url: download URL of the dataset. Overwrite the default URL.
    """
    return get_dataset('mni152_nlin_sym_ras', data_dir, url, resume, verbose)

def get_mni_icbm152_nlin_asym_09c(data_dir=None, url=None, resume=True, verbose=1):
    return get_dataset('mni_icbm152_nlin_asym_09c', data_dir, url, resume, verbose)

def get_mni_icbm152_linear(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the necessary files from the mni template
    :param str data_dir: path of the data directory. Used to force data storage
        in a non-standard location.
    :param str url: download URL of the dataset. Overwrite the default URL.
    """
    return get_dataset('mni_icbm152_linear', data_dir, url, resume, verbose)

def get_bids_examples(data_dir=None, url=None, resume=True, verbose=1, variant=None):
    """
    Download BIDS-examples-1
    """

    if variant is None or variant not in BIDS_EXAMPLES:
        variant = 'BIDS-examples-1-1.0.0-rc3u5'

    if url is None:
        url = BIDS_EXAMPLES[variant][0]
    md5 = BIDS_EXAMPLES[variant][1]
    data_dir = _get_dataset_dir(variant, data_dir=data_dir, verbose=verbose)

    if _fetch_file(url, data_dir, filetype=None, resume=resume, verbose=verbose,
                   md5sum=md5):
        return data_dir
    else:
        return None
