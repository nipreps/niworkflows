# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interfaces being evaluated before upstreaming to nipype.interfaces.utility

"""
from __future__ import absolute_import, division, print_function, unicode_literals

from nipype.interfaces.io import add_traits
from nipype.interfaces.base import (
    Str, DynamicTraitedSpec, BaseInterface
)


class KeySelectInputSpec(DynamicTraitedSpec):
    key = Str(mandatory=True, desc='selective key')


class KeySelectOutputSpec(DynamicTraitedSpec):
    key = Str(desc='propagates selected key')


class KeySelect(BaseInterface):
    """
    An interface that operates similarly to an OrderedDict

    >>> ks = KeySelect(keys=['MNI152NLin6Asym', 'MNI152Lin', 'fsaverage'],
    ...                fields=['field1', 'field2', 'field3'])
    >>> ks.inputs.field1 = ['fsl', 'mni', 'freesurfer']
    >>> ks.inputs.field2 = ['volume', 'volume', 'surface']
    >>> ks.inputs.field3 = [True, False, False]
    >>> ks.inputs.key = 'MNI152Lin'
    >>> ks.run().outputs
    <BLANKLINE>
    field1 = mni
    field2 = volume
    field3 = False
    key = MNI152Lin
    <BLANKLINE>

    >>> ks.inputs.field1 = ['fsl', 'mni', 'freesurfer']
    >>> ks.inputs.field2 = ['volume', 'volume', 'surface']
    >>> ks.inputs.field3 = [True, False, False]
    >>> ks.inputs.key = 'fsaverage'
    >>> ks.run().outputs
    <BLANKLINE>
    field1 = freesurfer
    field2 = surface
    field3 = False
    key = fsaverage
    <BLANKLINE>

    >>> ks.inputs.field1 = ['fsl', 'mni', 'freesurfer']
    >>> ks.inputs.field2 = ['volume', 'volume', 'surface']
    >>> ks.inputs.field3 = [True, False]  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: Trying to set an invalid value

    >>> ks.inputs.key = 'MNINLin2009cAsym'
    Traceback (most recent call last):
    ValueError: Selected key "MNINLin2009cAsym" not found in the index

    """
    input_spec = KeySelectInputSpec
    output_spec = KeySelectOutputSpec

    def __init__(self, keys=None, fields=None, **inputs):
        if not keys:
            raise ValueError('The index of ordered keys is required to instantiate '
                             'this interface.')

        nitems = len(keys)
        if isinstance(keys, str) or nitems == 1:
            raise ValueError('The index of ordered keys is required to be an iterable '
                             'over two or more string objects.')

        if len(set(keys)) != nitems:
            raise ValueError('Found duplicated entries in the index of ordered keys')

        self._keys = keys
        self._nitems = nitems

        if not fields:
            raise ValueError('A list or multiplexed fields must be provided at '
                             'instantiation time.')
        if isinstance(fields, str):
            fields = [fields]

        _invalid = set(self.input_spec.class_editable_traits()).intersection(fields)
        if _invalid:
            raise ValueError('Some fields are invalid (%s).' % ', '.join(_invalid))

        self._fields = fields

        # Call constructor
        super(KeySelect, self).__init__(**inputs)
        add_traits(self.inputs, self._fields)

        self.inputs.on_trait_change(self._check_len)

        for in_field in set(self._fields).intersection(inputs.keys()):
            setattr(self.inputs, in_field, inputs[in_field])

    def _check_len(self, name, new):
        if name in self._fields and (isinstance(new, str) or len(new) != self._nitems):
            raise ValueError('Trying to set an invalid value (%s) for input "%s"' % (
                new, name))

        if name == "key" and new not in self._keys:
            raise ValueError('Selected key "%s" not found in the index' % new)

    def _run_interface(self, runtime):
        return runtime

    def _list_outputs(self):
        index = self._keys.index(self.inputs.key)

        outputs = {k: getattr(self.inputs, k)[index]
                   for k in self._fields}

        outputs['key'] = self.inputs.key
        return outputs

    def _outputs(self):
        base = super(KeySelect, self)._outputs()
        if self._fields:
            base = add_traits(base, self._fields)
        return base
