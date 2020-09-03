# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities for the creation of nipype workflows.

Because these functions are meant to be inlined in nipype's ``connect`` invocations,
all the imports MUST be done in each function's context.

"""

__all__ = [
    "listify",
    "pop_file",
]


def pop_file(in_files):
    """
    Select the first file from a list of filenames.

    Used to grab the first echo's file when processing
    multi-echo data through workflows that only accept
    a single file.

    Examples
    --------
    >>> pop_file('some/file.nii.gz')
    'some/file.nii.gz'
    >>> pop_file(['some/file1.nii.gz', 'some/file2.nii.gz'])
    'some/file1.nii.gz'

    """
    if isinstance(in_files, (list, tuple)):
        return in_files[0]
    return in_files


def listify(value):
    """
    Convert to a list (inspired by bids.utils.listify).

    Examples
    --------
    >>> listify('some/file.nii.gz')
    ['some/file.nii.gz']
    >>> listify((0.1, 0.2))
    [0.1, 0.2]
    >>> listify(None) is None
    True

    """
    from pathlib import Path
    from nipype.interfaces.base import isdefined
    if not isdefined(value) or isinstance(value, type(None)):
        return value
    if isinstance(value, (str, bytes, Path)):
        return [str(value)]
    return list(value)
