# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Interfaces under evaluation before upstreaming to nipype.interfaces.utility."""
from nipype.interfaces.io import add_traits
from nipype.interfaces.base import (
    InputMultiObject,
    Str,
    DynamicTraitedSpec,
    BaseInterface,
    isdefined,
)


class _KeySelectInputSpec(DynamicTraitedSpec):
    key = Str(mandatory=True, desc="selective key")
    keys = InputMultiObject(Str, mandatory=True, min=1, desc="index of keys")


class _KeySelectOutputSpec(DynamicTraitedSpec):
    key = Str(desc="propagates selected key")


class KeySelect(BaseInterface):
    """
    An interface that operates similarly to an OrderedDict.

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

    >>> ks = KeySelect(fields=['field1', 'field2', 'field3'])
    >>> ks.inputs.keys=['MNI152NLin6Asym', 'MNI152Lin', 'fsaverage']
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

    >>> ks = KeySelect(fields=['field1', 'field2', 'field3'])
    >>> ks.inputs.keys=['MNI152NLin6Asym']
    >>> ks.inputs.field1 = ['fsl']
    >>> ks.inputs.field2 = ['volume']
    >>> ks.inputs.field3 = [True]
    >>> ks.inputs.key = 'MNI152NLin6Asym'
    >>> ks.run().outputs
    <BLANKLINE>
    field1 = fsl
    field2 = volume
    field3 = True
    key = MNI152NLin6Asym
    <BLANKLINE>

    """

    input_spec = _KeySelectInputSpec
    output_spec = _KeySelectOutputSpec

    def __init__(self, keys=None, fields=None, **inputs):
        """
        Instantiate a KeySelect utility interface.

        Examples
        --------
        >>> ks = KeySelect(fields='field1')
        >>> ks.inputs
        <BLANKLINE>
        field1 = <undefined>
        key = <undefined>
        keys = <undefined>
        <BLANKLINE>

        >>> ks = KeySelect(fields='field1', field1=['a', 'b'])
        >>> ks.inputs
        <BLANKLINE>
        field1 = ['a', 'b']
        key = <undefined>
        keys = <undefined>
        <BLANKLINE>

        >>> ks = KeySelect()
        Traceback (most recent call last):
        ValueError: A list or multiplexed...

        >>> ks = KeySelect(fields='key')
        Traceback (most recent call last):
        ValueError: Some fields are invalid...

        """
        # Call constructor
        super(KeySelect, self).__init__(**inputs)

        # Handle and initiate fields
        if not fields:
            raise ValueError(
                "A list or multiplexed fields must be provided at "
                "instantiation time."
            )
        if isinstance(fields, str):
            fields = [fields]

        _invalid = set(self.input_spec.class_editable_traits()).intersection(fields)
        if _invalid:
            raise ValueError("Some fields are invalid (%s)." % ", ".join(_invalid))

        self._fields = fields

        # Attach events
        self.inputs.on_trait_change(self._check_len)
        if keys:
            self.inputs.keys = keys

        # Add fields in self._fields
        add_traits(self.inputs, self._fields)

        for in_field in set(self._fields).intersection(inputs.keys()):
            setattr(self.inputs, in_field, inputs[in_field])

    def _check_len(self, name, new):
        if name == "keys":
            nitems = len(new)
            if len(set(new)) != nitems:
                raise ValueError(
                    "Found duplicated entries in the index of ordered keys"
                )

        if not isdefined(self.inputs.keys):
            return

        if name == "key" and new not in self.inputs.keys:
            raise ValueError('Selected key "%s" not found in the index' % new)

        if name in self._fields:
            if isinstance(new, str) or len(new) < 1:
                raise ValueError(
                    'Trying to set an invalid value (%s) for input "%s"' % (new, name)
                )

            if len(new) != len(self.inputs.keys):
                raise ValueError(
                    'Length of value (%s) for input field "%s" does not match '
                    "the length of the indexing list." % (new, name)
                )

    def _run_interface(self, runtime):
        return runtime

    def _list_outputs(self):
        index = self.inputs.keys.index(self.inputs.key)

        outputs = {k: getattr(self.inputs, k)[index] for k in self._fields}

        outputs["key"] = self.inputs.key
        return outputs

    def _outputs(self):
        base = super(KeySelect, self)._outputs()
        base = add_traits(base, self._fields)
        return base
