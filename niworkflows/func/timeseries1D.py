#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Operations to apply analytic tools designed for 4D NIfTI time series to
2D TSV files.
"""
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, SimpleInterface, File)
from .filter import _unfold_image


class TSVToImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='TSV file')
    t_rep = traits.Float(mandatory=True, desc='Repetition time')


class TSVToImageOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Image derived from input TSV')
    header = traits.CList(traits.Str, desc='Header stripped from TSV')


class TSVToImage(SimpleInterface):
    """Convert a TSV into an image"""
    input_spec = TSVToImageInputSpec
    output_spec = TSVToImageOutputSpec

    def _run_interface(self, runtime):
        out_file = fname_presuffix(self.inputs.in_file,
                                   suffix='_TSVToImage.nii.gz',
                                   newpath=runtime.cwd,
                                   use_ext=False)
        (self._results['out_file'],
         self._results['header']) = tsv2img(tsv=self.inputs.in_file,
                                            img=out_file,
                                            t_rep=self.inputs.t_rep)
        return runtime


class ImageToTSVInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='NIfTI file')
    mask = traits.Either(
        None, File(exists=True), default=None, usedefault=True,
        desc='NIfTI mask for TSV extraction')
    header = traits.CList(traits.Str, desc='Header for TSV columns')


class ImageToTSVOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='TSV derived from input image')


class ImageToTSV(SimpleInterface):
    """Convert an image into a TSV."""
    input_spec = ImageToTSVInputSpec
    output_spec = ImageToTSVOutputSpec

    def _run_interface(self, runtime):
        out_file = fname_presuffix(self.inputs.in_file,
                                   suffix='_imageToTSV.tsv',
                                   newpath=runtime.cwd,
                                   use_ext=False)
        self._results['out_file'] = img2tsv(img=self.inputs.in_file,
                                            tsv=out_file,
                                            mask=self.inputs.mask,
                                            header=self.inputs.header)
        return runtime


class TSVTo1DInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='TSV to transpose')
    transpose = traits.Bool(usedefault=True, desc='Indicates whether the '
                            'data should be transposed before saving as 1D')


class TSVTo1DOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='transposed TSV')
    header = traits.CList(traits.Str, desc='header stripped from TSV')


class TSVTo1D(SimpleInterface):
    """Convert a TSV file to a 1D file that can be used with AFNI."""
    input_spec = TSVTransposeInputSpec
    output_spec = TSVTransposeOutputSpec

    def _run_interface(self, runtime):
        out_file = fname_presuffix(self.inputs.in_file,
                                   suffix='_TSVto1D',
                                   newpath=runtime.cwd)
        (self._results['out_file'],
         self._results['header']) = tsv_to_1d(tsv=self.inputs.in_file,
                                              afni1d=out_file,
                                              transpose=self.inputs.transpose)
        return runtime


class TSVFrom1DInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='AFNI 1D file')
    in_format = traits.Enum(
        'columns', 'rows', usedefault=True,
        desc='format of the 1D file (observations in `rows` or `columns`)')
    header = traits.Either(None, traits.List, default=None, usedefault=True,
                           desc='header to prepend to the output TSV')


class TSVFrom1DOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='TSV derived from input 1D file')


class TSVFrom1D(SimpleInterface):
    """Convert an AFNI 1D file to (potentially) BIDS-compatible TSV format."""
    input_spec = TSVFrom1DInputSpec
    output_spec = TSVFrom1DOutputSpec

    def _run_interface(self, runtime):
        out_file = fname_presuffix(self.inputs.in_file,
                                   suffix='_TSVfrom1D',
                                   newpath=runtime.cwd)

        self._results['out_file'] = tsv_from_1D(afni1d=self.inputs.in_file,
                                                tsv=out_file,
                                                format=self.inputs.in_format,
                                                header=self.inputs.header)

        return runtime


def tsv2img(tsv, img, t_rep):
    """Create a dummy NIfTI image out of a tsv file, thus enabling it to be
    processed through nodes that ordinarily only operate on valid image files.

    Note that dummy NIfTIs must explicitly have repetition time set via input
    to this function, and the affine of a dummy NIfTI is identity.
    
    Parameters
    ----------
    tsv: str
        Path to the TSV file to be converted into an image.
    img: str
        Path where the converted file should be saved.
    t_rep: float
        Repetition time or sampling interval of the dataset.

    Returns
    -------
    str
        Path to the image generated from the TSV file.
    list
        Column names from the TSV header, formatted as a list, so that they
        can potentially be added later if the image is converted back.
    """
    tsv_data = pd.read_csv(tsv, header=None, skiprows=[0], sep='\t').values.T
    tsv_data.shape = [tsv_data.shape[0],1,1,tsv_data.shape[1]]
    header = list(pd.read_csv(tsv, sep='\t', nrows=0, header=0).columns)
    img = nib.Nifti1Image(dataobj=tsv_data, affine=np.eye(4))
    tsv_img.header['pixdim'][4] = t_rep
    nib.save(tsv_img, img)
    return img, header


def img2tsv(img, tsv, mask=None, header=None):
    """Dump data from an image file into tabular form.

    Parameters
    ----------
    img
        The time series image to be dumped into TSV form.
    tsv
        The path where the TSV should be saved.
    mask
        Spatial mask indicating voxels of the image to include in the TSV.
    header
        Header to be added to the output TSV file.

    Returns
    -------
    str
        Path to the TSV generated from image data.
    """
    img = nib.load(img)
    img_data = _unfold_image(img, mask=mask)

    if header is not None:
        tsv_data = DataFrame(data=img_data.T, columns=header)
        tsv_data.to_csv(tsv, sep='\t', index=False,
                        na_rep='n/a', header=True)
    else:
        tsv_data = DataFrame(data=img_data.T)
        tsv_data.to_csv(tsv, sep='\t', index=False,
                        na_rep='n/a', header=False)
    return tsv


def tsv_to_1d(tsv, afni1d, transpose=False):
    """Convert a tsv file into an AFNI 1D file, stripping the header and
    separately returning it.

    Parameters
    ----------
    tsv: str
        Path to a BIDS-formatted TSV file.
    afni1d: str
        Path where the output 1D file should be saved.
    transpose: bool
        Indicates whether the observations (e.g., in the time dimension) of
        the 1D file should run across rows or columns. To process a 1D file
        with 1D tools, this should be False, but to process a 1D file with 3D
        tools, this should be True.

    Returns
    -------
    str
        Path to the saved 1D file.
    list
        Column names from the TSV header, formatted as a list.
    """
    tsv_data = pd.read_csv(tsv, sep='\t', header=None, skiprows=[0])
    tsv_header = list(pd.read_csv(tsv, sep='\t', nrows=0, header=0).columns)
    if transpose:
        tsv_data = tsv_data.T
    tsv_data.to_csv(afni1d, sep='\t', index=False, na_rep='n/a', header=False)
    return afni1d, tsv_header


def tsv_from_1D(afni1d, tsv, obs='columns', header=None):
    """Convert an AFNI 1D file into a tsv file. Note that header information
    will not be included and must be provided separately.

    Parameters
    ----------
    afni1d: str
        Input AFNI 1D file.
    tsv: str
        Path where the output TSV should be written.
    obs: 'rows' or 'columns'
        Indicates whether the observations (e.g., in the time dimension) of
        the 1D file run across rows or columns. For a 1D file processed with
        1D tools, this will be `rows`, but for a 1D file processed with 3D
        tools, this will be `columns`.
    header: list
        List of column names for the output TSV.

    Returns
    -------
    str
        Path to the saved TSV.
    """
    # We use a lookbehind to ensure that any leading spaces in the
    # AFNI 1D file aren't improperly imported into the tsv as NaNs.
    tsv_data = pd.read_csv(afni1d, sep='(?<!^) ',
                           header=None, engine='python', comment='#')
    if obs == 'columns':
        tsv_data = tsv_data.T
    if header is not None:
        tsv_data.columns = header
    tsv_data.to_tsv(tsv, header=False)

    return tsv
