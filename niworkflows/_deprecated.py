# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Helpers for hard deprecations."""

from collections.abc import Mapping, Sequence


class DeprecationError(ImportError):
    """Raised when a niworkflows API has moved to NiReports."""


def _format_replacements(replacements):
    if isinstance(replacements, str):
        return [replacements]
    if isinstance(replacements, Mapping):
        return [f'{name} -> {target}' for name, target in replacements.items()]
    if isinstance(replacements, Sequence):
        return list(replacements)

    raise TypeError('replacements must be a string, mapping, or sequence')


def moved_to_nireports_message(module, replacements):
    lines = [
        f'{module} has been removed from niworkflows and moved to nireports.',
        'Import from one of the following locations instead:',
    ]
    lines.extend(f'  - {replacement}' for replacement in _format_replacements(replacements))
    return '\n'.join(lines)


def raise_moved_to_nireports(module, replacements):
    raise DeprecationError(moved_to_nireports_message(module, replacements))
