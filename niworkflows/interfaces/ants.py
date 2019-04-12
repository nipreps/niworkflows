#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Nipype interfaces for ANTs commands
"""

import os
from glob import glob
from nipype.interfaces import base
from nipype.interfaces.ants.base import ANTSCommandInputSpec, ANTSCommand
from nipype.interfaces.base import traits, isdefined


class ImageMathInputSpec(ANTSCommandInputSpec):
    dimension = traits.Int(3, usedefault=True, position=1, argstr='%d',
                           desc='dimension of output image')
    output_image = base.File(position=2, argstr='%s', name_source=['op1'],
                             name_template='%s_maths', desc='output image file',
                             keep_extension=True)
    operation = base.Str(mandatory=True, position=3, argstr='%s',
                         desc='operations and intputs')
    op1 = base.File(exists=True, mandatory=True, position=-2, argstr='%s',
                    desc='first operator')
    op2 = traits.Either(base.File(exists=True), base.Str, position=-1,
                        argstr='%s', desc='second operator')


class ImageMathOuputSpec(base.TraitedSpec):
    output_image = base.File(exists=True, desc='output image file')


class ImageMath(ANTSCommand):
    """
    Operations over images

    Example:
    --------

    """

    _cmd = 'ImageMath'
    input_spec = ImageMathInputSpec
    output_spec = ImageMathOuputSpec


class ResampleImageBySpacingInputSpec(ANTSCommandInputSpec):
    dimension = traits.Int(3, usedefault=True, position=1, argstr='%d',
                           desc='dimension of output image')
    input_image = base.File(exists=True, mandatory=True, position=2, argstr='%s',
                            desc='input image file')
    output_image = base.File(position=3, argstr='%s', name_source=['input_image'],
                             name_template='%s_resampled', desc='output image file',
                             keep_extension=True)
    out_spacing = traits.Either(
        traits.List(traits.Float, minlen=2, maxlen=3),
        traits.Tuple(traits.Float, traits.Float, traits.Float),
        traits.Tuple(traits.Float, traits.Float),
        position=4, argstr='%s', mandatory=True, desc='output spacing'
    )
    apply_smoothing = traits.Bool(False, argstr='%d', position=5,
                                  desc='smooth before resampling')
    addvox = traits.Int(argstr='%d', position=6, requires=['apply_smoothing'],
                        desc='addvox pads each dimension by addvox')
    nn_interp = traits.Bool(argstr='%d', desc='nn interpolation',
                            position=-1, requires=['addvox'])


class ResampleImageBySpacingOutputSpec(base.TraitedSpec):
    output_image = traits.File(exists=True, desc='resampled file')


class ResampleImageBySpacing(ANTSCommand):
    """
    Resamples an image with a given spacing

    Example:
    --------

    >>> res = ResampleImageBySpacing(dimension=3)
    >>> res.inputs.input_image = 'input.nii.gz'
    >>> res.inputs.output_image = 'output.nii.gz'
    >>> res.inputs.out_spacing = (4, 4, 4)
    'ResampleImageBySpacing input.nii.gz output.nii.gz 4 4 4'

    >>> res = ResampleImageBySpacing(dimension=3)
    >>> res.inputs.input_image = 'input.nii.gz'
    >>> res.inputs.output_image = 'output.nii.gz'
    >>> res.inputs.out_spacing = (4, 4, 4)
    >>> res.inputs.apply_smoothing = True
    'ResampleImageBySpacing input.nii.gz output.nii.gz 4 4 4 1'

    >>> res = ResampleImageBySpacing(dimension=3)
    >>> res.inputs.input_image = 'input.nii.gz'
    >>> res.inputs.output_image = 'output.nii.gz'
    >>> res.inputs.out_spacing = (4, 4, 4)
    >>> res.inputs.apply_smoothing = True
    >>> res.inputs.addvox = 2
    >>> res.inputs.nn_interp = False
    'ResampleImageBySpacing input.nii.gz output.nii.gz 4 4 4 1 2 0'

    """
    _cmd = 'ResampleImageBySpacing'
    input_spec = ResampleImageBySpacingInputSpec
    output_spec = ResampleImageBySpacingOutputSpec

    def _format_arg(self, name, trait_spec, value):
        if name == 'out_spacing':
            if len(value) != self.inputs.dimension:
                raise ValueError('out_spacing dimensions should match dimension')

            value = ' '.join(['%d' % d for d in value])

        return super(ResampleImageBySpacing, self)._format_arg(
            name, trait_spec, value)


class ThresholdImageInputSpec(ANTSCommandInputSpec):
    dimension = traits.Int(3, usedefault=True, position=1, argstr='%d',
                           desc='dimension of output image')
    input_image = base.File(exists=True, mandatory=True, position=2, argstr='%s',
                            desc='input image file')
    output_image = base.File(position=3, argstr='%s', name_source=['input_image'],
                             name_template='%s_resampled', desc='output image file',
                             keep_extension=True)

    mode = traits.Enum('Otsu', 'Kmeans', argstr='%s', position=4,
                       requires=['num_thresholds'], xor=['th_low', 'th_high'],
                       desc='whether to run Otsu / Kmeans thresholding')
    num_thresholds = traits.Int(position=5, argstr='%d',
                                desc='number of thresholds')
    input_mask = base.File(exists=True, requires=['num_thresholds'], argstr='%s',
                           desc='input mask for Otsu, Kmeans')

    th_low = traits.Float(position=4, argstr='%f', xor=['mode'],
                          desc='lower threshold')
    th_high = traits.Float(position=5, argstr='%f', xor=['mode'],
                           desc='upper threshold')
    inside_value = traits.Float(1, position=6, argstr='%f', requires=['th_low'],
                                desc='inside value')
    outside_value = traits.Float(0, position=7, argstr='%f', requires=['th_low'],
                                 desc='outside value')


class ThresholdImageOutputSpec(base.TraitedSpec):
    output_image = traits.File(exists=True, desc='resampled file')


class ThresholdImage(ANTSCommand):
    """
    Apply thresholds on images

    Example:
    --------

    >>> res = ThresholdImage(dimension=3)
    >>> res.inputs.input_image = 'input.nii.gz'
    >>> res.inputs.output_image = 'output.nii.gz'
    >>> res.inputs.th_low = 0.5
    >>> res.inputs.th_high = 1.0
    >>> res.inputs.inside_val = 1.0
    >>> res.inputs.outside_val = 0.0
    'ThresholdImage input.nii.gz output.nii.gz 0.50000 1.00000 1.00000 0.00000'

    >>> res = ThresholdImage(dimension=3)
    >>> res.inputs.input_image = 'input.nii.gz'
    >>> res.inputs.output_image = 'output.nii.gz'
    >>> res.inputs.mode = 'Kmeans'
    >>> res.inputs.num_thresholds = 4
    'ThresholdImage input.nii.gz output.nii.gz Kmeans 4'

    """
    _cmd = 'ThresholdImage'
    input_spec = ThresholdImageInputSpec
    output_spec = ThresholdImageOutputSpec


class AIInputSpec(ANTSCommandInputSpec):
    dimension = traits.Int(3, usedefault=True, argstr='-d %d',
                           desc='dimension of output image')
    verbose = traits.Bool(False, usedefault=True, argstr='-v %d',
                          desc='enable verbosity')

    fixed_image = traits.File(
        exists=True, mandatory=True,
        desc='Image to which the moving_image should be transformed')
    moving_image = traits.File(
        exists=True, mandatory=True,
        desc='Image that will be transformed to fixed_image')

    fixed_image_mask = traits.File(
        exists=True, argstr='-x %s', desc='fixed mage mask')
    moving_image_mask = traits.File(
        exists=True, requires=['fixed_image_mask'],
        desc='moving mage mask')

    metric_trait = (
        traits.Enum("Mattes", "GC", "MI"),
        traits.Int(32),
        traits.Enum('Regular', 'Random', 'None'),
        traits.Range(value=0.2, low=0.0, high=1.0)
    )
    metric = traits.Tuple(*metric_trait, argstr='-m %s', mandatory=True,
                          desc='the metric(s) to use.')

    transform = traits.Tuple(
        traits.Enum('Affine', 'Rigid', 'Similarity'),
        traits.Range(value=0.1, low=0.0, exclude_low=True),
        argstr='-t %s[%f]', usedefault=True,
        desc='Several transform options are available')

    principal_axes = traits.Bool(False, usedefault=True, argstr='-p %d', xor=['blobs'],
                                 desc='align using principal axes')
    search_factor = traits.Tuple(
        traits.Float(20), traits.Range(value=0.12, low=0.0, high=1.0),
        usedefault=True, argstr='-s [%f,%f]', desc='search factor')

    search_grid = traits.Either(
        traits.Tuple(traits.Float, traits.Tuple(traits.Float, traits.Float, traits.Float)),
        traits.Tuple(traits.Float, traits.Tuple(traits.Float, traits.Float)),
        argstr='-g %s', desc='Translation search grid in mm')

    convergence = traits.Tuple(
        traits.Range(low=1, high=10000, value=10),
        traits.Float(1e-6),
        traits.Range(low=1, high=100, value=10),
        usedefault=True, argstr='-c [%d,%f,%d]', desc='convergence')

    output_transform = traits.File(
        'initialization.mat', usedefault=True, argstr='-o %s',
        desc='output file name')


class AIOuputSpec(base.TraitedSpec):
    output_transform = traits.File(exists=True, desc='output file name')


class AI(ANTSCommand):
    """
    The replacement for ``AffineInitializer``.

    Example:
    --------

    """

    _cmd = 'antsAI'
    input_spec = AIInputSpec
    output_spec = AIOuputSpec

    def _run_interface(self, runtime, correct_return_codes=(0, )):
        runtime = super(AI, self)._run_interface(
            runtime, correct_return_codes)

        setattr(self, '_output', {
            'output_transform': os.path.join(
                runtime.cwd,
                os.path.basename(self.inputs.output_transform))
        })
        return runtime

    def _format_arg(self, opt, spec, val):
        if opt == 'metric':
            val = '%s[{fixed_image},{moving_image},%d,%s,%f]' % val
            val = val.format(
                fixed_image=self.inputs.fixed_image,
                moving_image=self.inputs.moving_image)
            return spec.argstr % val

        if opt == 'search_grid':
            val1 = 'x'.join(['%f' % v for v in val[1]])
            fmtval = '[%s]' % ','.join([str(val[0]), val1])
            return spec.argstr % fmtval

        if opt == 'fixed_image_mask':
            if isdefined(self.inputs.moving_image_mask):
                return spec.argstr % ('[%s,%s]' % (
                    val, self.inputs.moving_image_mask))

        return super(AI, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        return getattr(self, '_output')


class AntsJointFusionInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(
        3,
        2,
        4,
        argstr='-d %d',
        desc='This option forces the image to be treated '
        'as a specified-dimensional image. If not '
        'specified, the program tries to infer the '
        'dimensionality from the input image.')
    target_image = traits.List(
        base.InputMultiPath(base.File(exists=True)),
        argstr='-t %s',
        mandatory=True,
        desc='The target image (or '
        'multimodal target images) assumed to be '
        'aligned to a common image domain.')
    atlas_image = traits.List(
        base.InputMultiPath(base.File(exists=True)),
        argstr="-g %s...",
        mandatory=True,
        desc='The atlas image (or '
        'multimodal atlas images) assumed to be '
        'aligned to a common image domain.')
    atlas_segmentation_image = base.InputMultiPath(
        base.File(exists=True),
        argstr="-l %s...",
        mandatory=True,
        desc='The atlas segmentation '
        'images. For performing label fusion the number '
        'of specified segmentations should be identical '
        'to the number of atlas image sets.')
    alpha = traits.Float(
        default_value=0.1,
        usedefault=True,
        argstr='-a %s',
        desc=(
            'Regularization '
            'term added to matrix Mx for calculating the inverse. Default = 0.1'
        ))
    beta = traits.Float(
        default_value=2.0,
        usedefault=True,
        argstr='-b %s',
        desc=('Exponent for mapping '
              'intensity difference to the joint error. Default = 2.0'))
    retain_label_posterior_images = traits.Bool(
        False,
        argstr='-r',
        usedefault=True,
        requires=['atlas_segmentation_image'],
        desc=('Retain label posterior probability images. Requires '
              'atlas segmentations to be specified. Default = false'))
    retain_atlas_voting_images = traits.Bool(
        False,
        argstr='-f',
        usedefault=True,
        desc=('Retain atlas voting images. Default = false'))
    constrain_nonnegative = traits.Bool(
        False,
        argstr='-c',
        usedefault=True,
        desc=('Constrain solution to non-negative weights.'))
    patch_radius = traits.ListInt(
        minlen=3,
        maxlen=3,
        argstr='-p %s',
        desc=('Patch radius for similarity measures.'
              'Default: 2x2x2'))
    patch_metric = traits.Enum(
        'PC',
        'MSQ',
        argstr='-m %s',
        desc=('Metric to be used in determining the most similar '
              'neighborhood patch. Options include Pearson\'s '
              'correlation (PC) and mean squares (MSQ). Default = '
              'PC (Pearson correlation).'))
    search_radius = traits.List(
        [3, 3, 3],
        minlen=1,
        maxlen=3,
        argstr='-s %s',
        usedefault=True,
        desc=('Search radius for similarity measures. Default = 3x3x3. '
              'One can also specify an image where the value at the '
              'voxel specifies the isotropic search radius at that voxel.'))
    exclusion_image_label = traits.List(
        traits.Str(),
        argstr='-e %s',
        requires=['exclusion_image'],
        desc=('Specify a label for the exclusion region.'))
    exclusion_image = traits.List(
        base.File(exists=True),
        desc=('Specify an exclusion region for the given label.'))
    mask_image = base.File(
        argstr='-x %s',
        exists=True,
        desc='If a mask image '
        'is specified, fusion is only performed in the mask region.')
    out_label_fusion = base.File(
        argstr="%s", hash_files=False, desc='The output label fusion image.')
    out_intensity_fusion_name_format = traits.Str(
        argstr="",
        desc='Optional intensity fusion '
        'image file name format. '
        '(e.g. "antsJointFusionIntensity_%d.nii.gz")')
    out_label_post_prob_name_format = traits.Str(
        'antsJointFusionPosterior_%d.nii.gz',
        requires=['out_label_fusion', 'out_intensity_fusion_name_format'],
        desc='Optional label posterior probability '
        'image file name format.')
    out_atlas_voting_weight_name_format = traits.Str(
        'antsJointFusionVotingWeight_%d.nii.gz',
        requires=[
            'out_label_fusion', 'out_intensity_fusion_name_format',
            'out_label_post_prob_name_format'
        ],
        desc='Optional atlas voting weight image '
        'file name format.')
    verbose = traits.Bool(False, argstr="-v", desc=('Verbose output.'))


class AntsJointFusionOutputSpec(base.TraitedSpec):
    out_label_fusion = base.File(exists=True)
    out_intensity_fusion = base.OutputMultiPath(
        base.File(exists=True))
    out_label_post_prob = base.OutputMultiPath(
        base.File(exists=True))
    out_atlas_voting_weight = base.OutputMultiPath(
        base.File(exists=True))


class AntsJointFusion(ANTSCommand):
    """
    """
    input_spec = AntsJointFusionInputSpec
    output_spec = AntsJointFusionOutputSpec
    _cmd = 'antsJointFusion'

    def _format_arg(self, opt, spec, val):
        if opt == 'exclusion_image_label':
            retval = []
            for ii in range(len(self.inputs.exclusion_image_label)):
                retval.append(
                    '-e {0}[{1}]'.format(self.inputs.exclusion_image_label[ii],
                                         self.inputs.exclusion_image[ii]))
            retval = ' '.join(retval)
        elif opt == 'patch_radius':
            retval = '-p {0}'.format(self._format_xarray(val))
        elif opt == 'search_radius':
            retval = '-s {0}'.format(self._format_xarray(val))
        elif opt == 'out_label_fusion':
            if isdefined(self.inputs.out_intensity_fusion_name_format):
                if isdefined(self.inputs.out_label_post_prob_name_format):
                    if isdefined(
                            self.inputs.out_atlas_voting_weight_name_format):
                        retval = '-o [{0}, {1}, {2}, {3}]'.format(
                            self.inputs.out_label_fusion,
                            self.inputs.out_intensity_fusion_name_format,
                            self.inputs.out_label_post_prob_name_format,
                            self.inputs.out_atlas_voting_weight_name_format)
                    else:
                        retval = '-o [{0}, {1}, {2}]'.format(
                            self.inputs.out_label_fusion,
                            self.inputs.out_intensity_fusion_name_format,
                            self.inputs.out_label_post_prob_name_format)
                else:
                    retval = '-o [{0}, {1}]'.format(
                        self.inputs.out_label_fusion,
                        self.inputs.out_intensity_fusion_name_format)
            else:
                retval = '-o {0}'.format(self.inputs.out_label_fusion)
        elif opt == 'out_intensity_fusion_name_format':
            retval = ''
            if not isdefined(self.inputs.out_label_fusion):
                retval = '-o {0}'.format(
                    self.inputs.out_intensity_fusion_name_format)
        elif opt == 'atlas_image':
            atlas_image_cmd = " ".join([
                '-g [{0}]'.format(", ".join("'%s'" % fn for fn in ai))
                for ai in self.inputs.atlas_image
            ])
            retval = atlas_image_cmd
        elif opt == 'target_image':
            target_image_cmd = " ".join([
                '-t [{0}]'.format(", ".join("'%s'" % fn for fn in ai))
                for ai in self.inputs.target_image
            ])
            retval = target_image_cmd
        elif opt == 'atlas_segmentation_image':
            if len(val) != len(self.inputs.atlas_image):
                raise ValueError(
                    "Number of specified segmentations should be identical to the number "
                    "of atlas image sets {0}!={1}".format(
                        len(val), len(self.inputs.atlas_image)))

            atlas_segmentation_image_cmd = " ".join([
                '-l {0}'.format(fn)
                for fn in self.inputs.atlas_segmentation_image
            ])
            retval = atlas_segmentation_image_cmd
        else:

            return super(AntsJointFusion, self)._format_arg(opt, spec, val)
        return retval

    def _list_outputs(self):
        outputs = self._outputs().get()
        if isdefined(self.inputs.out_label_fusion):
            outputs['out_label_fusion'] = os.path.abspath(
                self.inputs.out_label_fusion)
        if isdefined(self.inputs.out_intensity_fusion_name_format):
            outputs['out_intensity_fusion'] = glob(os.path.abspath(
                self.inputs.out_intensity_fusion_name_format.replace(
                    '%d', '*'))
            )
        if isdefined(self.inputs.out_label_post_prob_name_format):
            outputs['out_label_post_prob'] = glob(os.path.abspath(
                self.inputs.out_label_post_prob_name_format.replace(
                    '%d', '*'))
            )
        if isdefined(self.inputs.out_atlas_voting_weight_name_format):
            outputs['out_atlas_voting_weight'] = glob(os.path.abspath(
                self.inputs.out_atlas_voting_weight_name_format.replace(
                    '%d', '*'))
            )
        return outputs
