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
import traits.api as traits
from functools import reduce


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

    Outputs
    -------
    variables_deriv: list
        A list of variables to include in the final data frame after adding
        the specified derivative terms.
    data_deriv: pandas DataFrame object
        Table of values of all observations of all variables, including any
        specified derivative terms.
    """
    variables_deriv = {}
    data_deriv = {}
    if 0 in order:
        data_deriv[0] = data[variables]
        variables_deriv[0] = variables
        order = set(order) - set([0])
    for o in order:
        variables_deriv[o] = ['{}_derivative{}'.format(v, o)
                                 for v in variables]
        data_deriv[o] = np.tile(np.nan, data[variables].shape)
        data_deriv[o][o:,:] = np.diff(data[variables], n=o, axis=0)
    variables_deriv = reduce((lambda x, y: x + y), variables_deriv.values())
    data_deriv = pd.DataFrame(columns=variables_deriv,
                    data=np.concatenate([*data_deriv.values()], axis=1))

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

    Outputs
    -------
    variables_deriv: list
        A list of variables to include in the final data frame after adding
        the specified exponential terms.
    data_deriv: pandas DataFrame object
        Table of values of all observations of all variables, including any
        specified exponential terms.
    """
    variables_exp = {}
    data_exp = {}
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
    if re.search('\^\^[0-9]+$', expr):
        order = re.compile('\^\^([0-9]+)$').findall(expr)
        order = range(1, int(*order) + 1)
        variables, data = exponential_terms(order, variables, data)
    elif re.search('\^[0-9]+[\-]?[0-9]*$', expr):
        order = re.compile('\^([0-9]+[\-]?[0-9]*)').findall(expr)
        order = _order_as_range(*order)
        variables, data = exponential_terms(order, variables, data)
    return variables, data


def _check_and_expand_derivative(expr, variables, data):
    """Check if the current operation specifies a temporal derivative. dd6x
    specifies all derivatives up to the 6th, d5-6x the 5th and 6th, d6x the
    6th only."""
    if re.search('^dd[0-9]+x', expr):
        order = re.compile('^dd([0-9]+)x').findall(expr)
        order = range(0, int(*order) + 1)
        (variables, data) = temporal_derivatives(order, variables, data)
    elif re.search('^d[0-9]+[\-]?[0-9]*x', expr):
        order = re.compile('^d([0-9]+[\-]?[0-9]*)x').findall(expr)
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

    Outputs
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


def parse_formula(model_formula, parent_data):
    """
    Recursively parse a model formula by breaking it into additive atoms
    and tracking grouping symbol depth.

    Parameters
    ----------
    model_formula: str
        Expression for the model formula, e.g.
        '(a + b)^^2 + dd1(c + (d + e)^3) + f'
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

    Outputs
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
    data = data.T.drop_duplicates().T

    return variables, data
