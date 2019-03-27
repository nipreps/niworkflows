#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Select terms for a confound model, and compute any requisite expansions.
"""

import re
import numpy as np
import pandas as pd
from functools import reduce
from collections import deque, OrderedDict
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, isdefined,
    SimpleInterface
)


class ExpandModelInputInterface(BaseInterfaceInputSpec):
    confounds_file = File(exists=True, mandatory=True,
                          desc='TSV containing confound time series for '
                          'expansion according to the specified formula.')
    model_formula = traits.Str(
        '(dd1(rps + wm + csf + gsr))^^2 + others', usedefault=True,
        desc='Formula for generating model expansions. By default, the '
        '32-parameter expansion will be generated. Note that any expressions '
        'to be expanded *must* be in parentheses, even if they include only '
        'a single variable (e.g., (x)^2, not x^2).\n\n'
        'Examples:\n'
        '* rps + wm + csf + gsr : 9-parameter model. rps denotes realignment\n'
        '  parameters, wm denotes mean white matter signal, csf denotes mean\n'
        '  cerebrospinal fluid signal, and gsr denotes mean global signal.\n'
        '* (dd1(rps + wm + csf + gsr))^^2 : 36-parameter expansion.\n'
        '  rps + wm + csf + gsr denotes that realignment parameters and mean\n'
        '  WM, CSF, and global signals should be included. dd1 denotes that\n'
        '  these signals should be augmented with their first temporal\n'
        '  derivatives. ^^2 denotes that the original signals and temporal\n'
        '  derivatives should be augmented with quadratic expansions.\n'
        '* (dd1(rps))^^2 : 24-parameter expansion. rps denotes that\n'
        '  realignment parameters should be included. dd1 and ^^2 denote\n'
        '  temporal derivative and quadratic expansions as above.\n'
        '* (dd1(rps + wm + csf + gsr))^^2 + others : generate all expansion\n'
        '  terms necessary for a 36-parameter model as above, and\n'
        '  concatenate those expansion terms to all other regressor columns\n'
        '  in the confounds file.\n')
    output_file = File(desc='Output path')


class ExpandModelOutputInterface(TraitedSpec):
    confounds_file = File(exists=True, desc='Output confounds file')


class ExpandModel(SimpleInterface):
    """Expand a confound model according to a specified formula.
    """
    input_spec = ExpandModelInputInterface
    output_spec = ExpandModelOutputInterface

    def _run_interface(self, runtime):
        if isdefined(self.inputs.output_file):
            out_file = self.inputs.output_file
        else:
            out_file = fname_presuffix(
                self.inputs.confounds_file,
                suffix='_expansion.tsv',
                newpath=runtime.cwd,
                use_ext=False)

        confounds_data = pd.read_csv(self.inputs.confounds_file, sep='\t')
        _, confounds_data = parse_formula(
            model_formula=self.inputs.model_formula,
            parent_data=confounds_data,
            unscramble=True
        )
        confounds_data.to_csv(out_file, sep='\t', index=False,
                              na_rep='n/a')
        self._results['confounds_file'] = out_file
        return runtime


class SpikeRegressorsInputInterface(BaseInterfaceInputSpec):
    confounds_file = File(
        exists=True, mandatory=True,
        desc='TSV containing criterion time series (e.g., framewise '
        'displacement, DVARS) to be used for creating spike regressors.')
    fd_thresh = traits.Float(
        0.5, usedefault=True,
        desc='Minimum framewise displacement threshold for flagging a frame '
        'as a spike.')
    dvars_thresh = traits.Float(
        1.5, usedefault=True,
        desc='Minimum standardised DVARS threshold for flagging a frame as '
        'a spike.')
    header_prefix = traits.Str(
        'motion_outlier', usedefault=True,
        desc='Prefix for spikes in the output TSV header')
    lags = traits.List(traits.Int, value=[0], usedefault=True,
                       desc='Relative indices of lagging frames to flag for '
                       'each flagged frame')
    minimum_contiguous = traits.Either(
        None, traits.Int, usedefault=True,
        desc='Minimum number of contiguous volumes required to avoid '
        'flagging as a spike')
    concatenate = traits.Bool(
        True, usedefault=True,
        desc='Indicates whether to concatenate spikes to existing confounds '
        'or return spikes only')
    output_format = traits.Enum('spikes', 'mask', usedefault=True,
                                desc='Format of output (spikes or mask)')
    output_file = File(desc='Output path')


class SpikeRegressorsOutputInterface(TraitedSpec):
    confounds_file = File(exists=True, desc='Output confounds file')


class SpikeRegressors(SimpleInterface):
    """Generate spike regressors.
    """
    input_spec = SpikeRegressorsInputInterface
    output_spec = SpikeRegressorsOutputInterface

    def _run_interface(self, runtime):
        if isdefined(self.inputs.output_file):
            out_file = self.inputs.output_file
        else:
            out_file = fname_presuffix(
                self.inputs.confounds_file,
                suffix='_desc-motion_outliers.tsv',
                newpath=runtime.cwd,
                use_ext=False)

        spike_criteria = {
            'framewise_displacement': ('>', self.inputs.fd_thresh),
            'std_dvars': ('>', self.inputs.dvars_thresh)
        }

        confounds_data = pd.read_csv(self.inputs.confounds_file, sep='\t')
        confounds_data = spike_regressors(
            data=confounds_data,
            criteria=spike_criteria,
            header_prefix=self.inputs.header_prefix,
            lags=self.inputs.lags,
            minimum_contiguous=self.inputs.minimum_contiguous,
            concatenate=self.inputs.concatenate,
            output=self.inputs.output_format
        )
        confounds_data.to_csv(out_file, sep='\t', index=False,
                              na_rep='n/a')
        self._results['confounds_file'] = out_file
        return runtime


def spike_regressors(data, criteria=None, header_prefix='motion_outlier',
                     lags=None, minimum_contiguous=None, concatenate=True,
                     output='spikes'):
    """
    Add spike regressors to a confound/nuisance matrix.

    Parameters
    ----------
    data: pandas DataFrame object
        A tabulation of observations from which spike regressors should be
        estimated.
    criteria: dict{str: ('>' or '<', float)}
        Criteria for generating a spike regressor. If, for a given frame, the
        value of the variable corresponding to the key exceeds the threshold
        indicated by the value, then a spike regressor is created for that
        frame. By default, the strategy from Power 2014 is implemented: any
        frames with FD greater than 0.5 or standardised DV greater than 1.5
        are flagged for censoring.
    header_prefix: str
        The prefix used to indicate spike regressors in the output data table.
    lags: list(int)
        A list indicating the frames to be censored relative to each flag.
        For instance, [0] censors the flagged frame, while [0, 1] censors
        both the flagged frame and the following frame.
    minimum_contiguous: int or None
        The minimum number of contiguous frames that must be unflagged for
        spike regression. If any series of contiguous unflagged frames is
        shorter than the specified minimum, then all of those frames will
        additionally have spike regressors implemented.
    concatenate: bool
        Indicates whether the returned object should include only spikes
        (if false) or all input time series and spikes (if true, default).
    output: str
        Indicates whether the output should be formatted as spike regressors
        ('spikes', a separate column for each outlier) or as a temporal mask
        ('mask', a single output column indicating the locations of outliers).

    Returns
    -------
    data: pandas DataFrame object
        The input DataFrame with a column for each spike regressor.

    References
    ----------
    Power JD, Mitra A, Laumann TO, Snyder AZ, Schlaggar BL, Petersen SE (2014)
        Methods to detect, characterize, and remove motion artifact in resting
        state fMRI. NeuroImage.
    """
    mask = {}
    indices = range(data.shape[0])
    lags = lags or [0]
    criteria = criteria or {'framewise_displacement': ('>', 0.5),
                            'std_dvars': ('>', 1.5)}
    for metric, (criterion, threshold) in criteria.items():
        if criterion == '<':
            mask[metric] = set(np.where(data[metric] < threshold)[0])
        elif criterion == '>':
            mask[metric] = set(np.where(data[metric] > threshold)[0])
    mask = reduce((lambda x, y: x | y), mask.values())

    for lag in lags:
        mask = set([m + lag for m in mask]) | mask

    mask = mask.intersection(indices)
    if minimum_contiguous is not None:
        post_final = data.shape[0] + 1
        epoch_length = np.diff(sorted(mask |
                                      set([-1, post_final]))) - 1
        epoch_end = sorted(mask | set([post_final]))
        for end, length in zip(epoch_end, epoch_length):
            if length < minimum_contiguous:
                mask = mask | set(range(end - length, end))
        mask = mask.intersection(indices)

    if output == 'mask':
        spikes = np.zeros(data.shape[0])
        spikes[list(mask)] = 1
        spikes = pd.DataFrame(data=spikes, columns=[header_prefix])
    else:
        spikes = np.zeros((max(indices)+1, len(mask)))
        for i, m in enumerate(sorted(mask)):
            spikes[m, i] = 1
        header = ['{:s}{:02d}'.format(header_prefix, vol)
                  for vol in range(len(mask))]
        spikes = pd.DataFrame(data=spikes, columns=header)
    if concatenate:
        return pd.concat((data, spikes), axis=1)
    else:
        return spikes


def temporal_derivatives(order, variables, data):
    """
    Compute temporal derivative terms by the method of backwards differences.

    Parameters
    ----------
    order: range or list(int)
        A list of temporal derivative terms to include. For instance, [1, 2]
        indicates that the first and second derivative terms should be added.
        To retain the original terms, 0 *must* be included in the list.
    variables: list(str)
        List of variables for which temporal derivative terms should be
        computed.
    data: pandas DataFrame object
        Table of values of all observations of all variables.

    Returns
    -------
    variables_deriv: list
        A list of variables to include in the final data frame after adding
        the specified derivative terms.
    data_deriv: pandas DataFrame object
        Table of values of all observations of all variables, including any
        specified derivative terms.
    """
    variables_deriv = OrderedDict()
    data_deriv = OrderedDict()
    if 0 in order:
        data_deriv[0] = data[variables]
        variables_deriv[0] = variables
        order = set(order) - set([0])
    for o in order:
        variables_deriv[o] = ['{}_derivative{}'.format(v, o)
                              for v in variables]
        data_deriv[o] = np.tile(np.nan, data[variables].shape)
        data_deriv[o][o:, :] = np.diff(data[variables], n=o, axis=0)
    variables_deriv = reduce((lambda x, y: x + y), variables_deriv.values())
    data_deriv = pd.DataFrame(columns=variables_deriv,
                              data=np.concatenate([*data_deriv.values()],
                                                  axis=1))

    return (variables_deriv, data_deriv)


def exponential_terms(order, variables, data):
    """
    Compute exponential expansions.

    Parameters
    ----------
    order: range or list(int)
        A list of exponential terms to include. For instance, [1, 2]
        indicates that the first and second exponential terms should be added.
        To retain the original terms, 1 *must* be included in the list.
    variables: list(str)
        List of variables for which exponential terms should be computed.
    data: pandas DataFrame object
        Table of values of all observations of all variables.

    Returns
    -------
    variables_exp: list
        A list of variables to include in the final data frame after adding
        the specified exponential terms.
    data_exp: pandas DataFrame object
        Table of values of all observations of all variables, including any
        specified exponential terms.
    """
    variables_exp = OrderedDict()
    data_exp = OrderedDict()
    if 1 in order:
        data_exp[1] = data[variables]
        variables_exp[1] = variables
        order = set(order) - set([1])
    for o in order:
        variables_exp[o] = ['{}_power{}'.format(v, o) for v in variables]
        data_exp[o] = data[variables]**o
    variables_exp = reduce((lambda x, y: x + y), variables_exp.values())
    data_exp = pd.DataFrame(columns=variables_exp,
                            data=np.concatenate([*data_exp.values()], axis=1))
    return (variables_exp, data_exp)


def _order_as_range(order):
    """Convert a hyphenated string representing order for derivative or
    exponential terms into a range object that can be passed as input to the
    appropriate expansion function."""
    order = order.split('-')
    order = [int(o) for o in order]
    if len(order) > 1:
        order = range(order[0], (order[-1] + 1))
    return order


def _check_and_expand_exponential(expr, variables, data):
    """Check if the current operation specifies exponential expansion. ^^6
    specifies all powers up to the 6th, ^5-6 the 5th and 6th powers, ^6 the
    6th only."""
    if re.search(r'\^\^[0-9]+$', expr):
        order = re.compile(r'\^\^([0-9]+)$').findall(expr)
        order = range(1, int(*order) + 1)
        variables, data = exponential_terms(order, variables, data)
    elif re.search(r'\^[0-9]+[\-]?[0-9]*$', expr):
        order = re.compile(r'\^([0-9]+[\-]?[0-9]*)').findall(expr)
        order = _order_as_range(*order)
        variables, data = exponential_terms(order, variables, data)
    return variables, data


def _check_and_expand_derivative(expr, variables, data):
    """Check if the current operation specifies a temporal derivative. dd6x
    specifies all derivatives up to the 6th, d5-6x the 5th and 6th, d6x the
    6th only."""
    if re.search(r'^dd[0-9]+', expr):
        order = re.compile(r'^dd([0-9]+)').findall(expr)
        order = range(0, int(*order) + 1)
        (variables, data) = temporal_derivatives(order, variables, data)
    elif re.search(r'^d[0-9]+[\-]?[0-9]*', expr):
        order = re.compile(r'^d([0-9]+[\-]?[0-9]*)').findall(expr)
        order = _order_as_range(*order)
        (variables, data) = temporal_derivatives(order, variables, data)
    return variables, data


def _check_and_expand_subformula(expression, parent_data, variables, data):
    """Check if the current operation contains a suboperation, and parse it
    where appropriate."""
    grouping_depth = 0
    for i, char in enumerate(expression):
        if char == '(':
            if grouping_depth == 0:
                formula_delimiter = i + 1
            grouping_depth += 1
        elif char == ')':
            grouping_depth -= 1
            if grouping_depth == 0:
                expr = expression[formula_delimiter:i].strip()
                return parse_formula(expr, parent_data)
    return variables, data


def parse_expression(expression, parent_data):
    """
    Parse an expression in a model formula.

    Parameters
    ----------
    expression: str
        Formula expression: either a single variable or a variable group
        paired with an operation (exponentiation or differentiation).
    parent_data: pandas DataFrame
        The source data for the model expansion.

    Returns
    -------
    variables: list
        A list of variables in the provided formula expression.
    data: pandas DataFrame
        A tabulation of all terms in the provided formula expression.
    """
    variables = None
    data = None
    variables, data = _check_and_expand_subformula(expression,
                                                   parent_data,
                                                   variables,
                                                   data)
    variables, data = _check_and_expand_exponential(expression,
                                                    variables,
                                                    data)
    variables, data = _check_and_expand_derivative(expression,
                                                   variables,
                                                   data)
    if variables is None:
        expr = expression.strip()
        variables = [expr]
        data = parent_data[expr]
    return variables, data


def _get_matches_from_data(regex, variables):
    matches = re.compile(regex)
    matches = ' + '.join([v for v in variables if matches.match(v)])
    return matches


def _get_variables_from_formula(model_formula):
    symbols_to_clear = [' ', r'\(', r'\)', 'dd[0-9]+', r'd[0-9]+[\-]?[0-9]*',
                        r'\^\^[0-9]+', r'\^[0-9]+[\-]?[0-9]*']
    for symbol in symbols_to_clear:
        model_formula = re.sub(symbol, '', model_formula)
    variables = model_formula.split('+')
    return variables


def _expand_shorthand(model_formula, variables):
    """Expand shorthand terms in the model formula.
    """
    wm = 'white_matter'
    gsr = 'global_signal'
    rps = 'trans_x + trans_y + trans_z + rot_x + rot_y + rot_z'
    fd = 'framewise_displacement'
    acc = _get_matches_from_data('a_comp_cor_[0-9]+', variables)
    tcc = _get_matches_from_data('t_comp_cor_[0-9]+', variables)
    dv = _get_matches_from_data('^std_dvars$', variables)
    dvall = _get_matches_from_data('.*dvars', variables)
    nss = _get_matches_from_data('non_steady_state_outlier[0-9]+',
                                 variables)
    spikes = _get_matches_from_data('motion_outlier[0-9]+', variables)

    model_formula = re.sub('wm', wm, model_formula)
    model_formula = re.sub('gsr', gsr, model_formula)
    model_formula = re.sub('rps', rps, model_formula)
    model_formula = re.sub('fd', fd, model_formula)
    model_formula = re.sub('acc', acc, model_formula)
    model_formula = re.sub('tcc', tcc, model_formula)
    model_formula = re.sub('dv', dv, model_formula)
    model_formula = re.sub('dvall', dvall, model_formula)
    model_formula = re.sub('nss', nss, model_formula)
    model_formula = re.sub('spikes', spikes, model_formula)

    formula_variables = _get_variables_from_formula(model_formula)
    others = ' + '.join(set(variables) - set(formula_variables))
    model_formula = re.sub('others', others, model_formula)
    return model_formula


def _unscramble_regressor_columns(parent_data, data):
    """Reorder the columns of a confound matrix such that the columns are in
    the same order as the input data with any expansion columns inserted
    immediately after the originals.
    """
    matches = ['_power[0-9]+', '_derivative[0-9]+']
    var = OrderedDict((c, deque()) for c in parent_data.columns)
    for c in data.columns:
        col = c
        for m in matches:
            col = re.sub(m, '', col)
        if col == c:
            var[col].appendleft(c)
        else:
            var[col].append(c)
    unscrambled = reduce((lambda x, y: x + y), var.values())
    return data[[*unscrambled]]


def parse_formula(model_formula, parent_data, unscramble=False):
    """
    Recursively parse a model formula by breaking it into additive atoms
    and tracking grouping symbol depth.

    Parameters
    ----------
    model_formula: str
        Expression for the model formula, e.g.
        '(a + b)^^2 + dd1(c + (d + e)^3) + f'
        Note that any expressions to be expanded *must* be in parentheses,
        even if they include only a single variable (e.g., (x)^2, not x^2).
    parent_data: pandas DataFrame
        A tabulation of all values usable in the model formula. Each additive
        term in `model_formula` should correspond either to a variable in this
        data frame or to instructions for operating on a variable (for
        instance, computing temporal derivatives or exponential terms).

        Temporal derivative options:
        * d6(variable) for the 6th temporal derivative
        * dd6(variable) for all temporal derivatives up to the 6th
        * d4-6(variable) for the 4th through 6th temporal derivatives
        * 0 must be included in the temporal derivative range for the original
          term to be returned when temporal derivatives are computed.

        Exponential options:
        * (variable)^6 for the 6th power
        * (variable)^^6 for all powers up to the 6th
        * (variable)^4-6 for the 4th through 6th powers
        * 1 must be included in the powers range for the original term to be
          returned when exponential terms are computed.

        Temporal derivatives and exponential terms are computed for all terms
        in the grouping symbols that they adjoin.

    Returns
    -------
    variables: list(str)
        A list of variables included in the model parsed from the provided
        formula.
    data: pandas DataFrame
        All values in the complete model.
    """
    variables = {}
    data = {}
    expr_delimiter = 0
    grouping_depth = 0
    model_formula = _expand_shorthand(model_formula, parent_data.columns)
    for i, char in enumerate(model_formula):
        if char == '(':
            grouping_depth += 1
        elif char == ')':
            grouping_depth -= 1
        elif grouping_depth == 0 and char == '+':
            expression = model_formula[expr_delimiter:i].strip()
            variables[expression] = None
            data[expression] = None
            expr_delimiter = i + 1
    expression = model_formula[expr_delimiter:].strip()
    variables[expression] = None
    data[expression] = None
    for expression in list(variables):
        if expression[0] == '(' and expression[-1] == ')':
            (variables[expression],
             data[expression]) = parse_formula(expression[1:-1],
                                               parent_data)
        else:
            (variables[expression],
             data[expression]) = parse_expression(expression,
                                                  parent_data)
    variables = list(set(reduce((lambda x, y: x + y), variables.values())))
    data = pd.concat((data.values()), axis=1)

    if unscramble:
        data = _unscramble_regressor_columns(parent_data, data)

    return variables, data
