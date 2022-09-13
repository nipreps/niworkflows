# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Interfaces under evaluation before upstreaming to nipype.interfaces.utility."""
import numpy as np
import re
import json
from collections import OrderedDict

from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.io import add_traits
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    DynamicTraitedSpec,
    File,
    InputMultiObject,
    isdefined,
    SimpleInterface,
    Str,
    TraitedSpec,
    traits,
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


class _AddTSVHeaderInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="input file")
    columns = traits.List(traits.Str, mandatory=True, desc="header for columns")


class _AddTSVHeaderOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output average file")


class AddTSVHeader(SimpleInterface):
    r"""Add a header row to a TSV file

    Examples
    --------
    An example TSV:

    >>> np.savetxt('data.tsv', np.arange(30).reshape((6, 5)), delimiter='\t')

    Add headers:

    >>> addheader = AddTSVHeader()
    >>> addheader.inputs.in_file = 'data.tsv'
    >>> addheader.inputs.columns = ['a', 'b', 'c', 'd', 'e']
    >>> res = addheader.run()
    >>> df = pd.read_csv(res.outputs.out_file, delim_whitespace=True,
    ...                  index_col=None)
    >>> df.columns.ravel().tolist()
    ['a', 'b', 'c', 'd', 'e']

    >>> np.all(df.values == np.arange(30).reshape((6, 5)))
    True

    """
    input_spec = _AddTSVHeaderInputSpec
    output_spec = _AddTSVHeaderOutputSpec

    def _run_interface(self, runtime):
        out_file = fname_presuffix(
            self.inputs.in_file,
            suffix="_motion.tsv",
            newpath=runtime.cwd,
            use_ext=False,
        )
        data = np.loadtxt(self.inputs.in_file)
        np.savetxt(
            out_file,
            data,
            delimiter="\t",
            header="\t".join(self.inputs.columns),
            comments="",
        )

        self._results["out_file"] = out_file
        return runtime


class _JoinTSVColumnsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="input file")
    join_file = File(exists=True, mandatory=True, desc="file to be adjoined")
    side = traits.Enum("right", "left", usedefault=True, desc="where to join")
    columns = traits.List(traits.Str, desc="header for columns")


class _JoinTSVColumnsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output TSV file")


class JoinTSVColumns(SimpleInterface):
    r"""Add a header row to a TSV file

    Examples
    --------
    An example TSV:

    >>> data = np.arange(30).reshape((6, 5))
    >>> np.savetxt('data.tsv', data[:, :3], delimiter='\t')
    >>> np.savetxt('add.tsv', data[:, 3:], delimiter='\t')

    Join without naming headers:

    >>> join = JoinTSVColumns()
    >>> join.inputs.in_file = 'data.tsv'
    >>> join.inputs.join_file = 'add.tsv'
    >>> res = join.run()
    >>> df = pd.read_csv(res.outputs.out_file, delim_whitespace=True,
    ...                  index_col=None, dtype=float, header=None)
    >>> df.columns.ravel().tolist() == list(range(5))
    True

    >>> np.all(df.values.astype(int) == data)
    True

    Adding column names:

    >>> join = JoinTSVColumns()
    >>> join.inputs.in_file = 'data.tsv'
    >>> join.inputs.join_file = 'add.tsv'
    >>> join.inputs.columns = ['a', 'b', 'c', 'd', 'e']
    >>> res = join.run()
    >>> res.outputs.out_file  # doctest: +ELLIPSIS
    '...data_joined.tsv'
    >>> df = pd.read_csv(res.outputs.out_file, delim_whitespace=True,
    ...                  index_col=None)
    >>> df.columns.ravel().tolist()
    ['a', 'b', 'c', 'd', 'e']

    >>> np.all(df.values == np.arange(30).reshape((6, 5)))
    True

    >>> join = JoinTSVColumns()
    >>> join.inputs.in_file = 'data.tsv'
    >>> join.inputs.join_file = 'add.tsv'
    >>> join.inputs.side = 'left'
    >>> join.inputs.columns = ['a', 'b', 'c', 'd', 'e']
    >>> res = join.run()
    >>> df = pd.read_csv(res.outputs.out_file, delim_whitespace=True,
    ...                  index_col=None)
    >>> df.columns.ravel().tolist()
    ['a', 'b', 'c', 'd', 'e']

    >>> np.all(df.values == np.hstack((data[:, 3:], data[:, :3])))
    True

    """
    input_spec = _JoinTSVColumnsInputSpec
    output_spec = _JoinTSVColumnsOutputSpec

    def _run_interface(self, runtime):
        out_file = fname_presuffix(
            self.inputs.in_file,
            suffix="_joined.tsv",
            newpath=runtime.cwd,
            use_ext=False,
        )

        header = ""
        if isdefined(self.inputs.columns) and self.inputs.columns:
            header = "\t".join(self.inputs.columns)

        with open(self.inputs.in_file) as ifh:
            data = ifh.read().splitlines(keepends=False)

        with open(self.inputs.join_file) as ifh:
            join = ifh.read().splitlines(keepends=False)

        if len(data) != len(join):
            raise ValueError("Number of columns in datasets do not match")

        merged = []
        for d, j in zip(data, join):
            line = "%s\t%s" % ((j, d) if self.inputs.side == "left" else (d, j))
            merged.append(line)

        if header:
            merged.insert(0, header)

        with open(out_file, "w") as ofh:
            ofh.write("\n".join(merged))

        self._results["out_file"] = out_file
        return runtime


class _DictMergeInputSpec(BaseInterfaceInputSpec):
    in_dicts = traits.List(
        traits.Either(traits.Dict, traits.Instance(OrderedDict)),
        desc="Dictionaries to be merged. In the event of a collision, values "
        "from dictionaries later in the list receive precedence.",
    )


class _DictMergeOutputSpec(TraitedSpec):
    out_dict = traits.Dict(desc="Merged dictionary")


class DictMerge(SimpleInterface):
    """Merge (ordered) dictionaries."""

    input_spec = _DictMergeInputSpec
    output_spec = _DictMergeOutputSpec

    def _run_interface(self, runtime):
        out_dict = {}
        for in_dict in self.inputs.in_dicts:
            out_dict.update(in_dict)
        self._results["out_dict"] = out_dict
        return runtime


class _TSV2JSONInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="Input TSV file")
    index_column = traits.Str(
        mandatory=True,
        desc="Name of the column in the TSV to be used "
        "as the top-level key in the JSON. All "
        "remaining columns will be assigned as "
        "nested keys.",
    )
    output = traits.Either(
        None,
        File,
        desc="Path where the output file is to be saved. "
        "If this is `None`, then a JSON-compatible "
        "dictionary is returned instead.",
    )
    additional_metadata = traits.Either(
        None,
        traits.Dict,
        traits.Instance(OrderedDict),
        usedefault=True,
        desc="Any additional metadata that "
        "should be applied to all "
        "entries in the JSON.",
    )
    drop_columns = traits.Either(
        None,
        traits.List(),
        usedefault=True,
        desc="List of columns in the TSV to be " "dropped from the JSON.",
    )
    enforce_case = traits.Bool(
        True,
        usedefault=True,
        desc="Enforce snake case for top-level keys " "and camel case for nested keys",
    )


class _TSV2JSONOutputSpec(TraitedSpec):
    output = traits.Either(
        traits.Dict,
        File(exists=True),
        traits.Instance(OrderedDict),
        desc="Output dictionary or JSON file",
    )


class TSV2JSON(SimpleInterface):
    """Convert metadata from TSV format to JSON format."""

    input_spec = _TSV2JSONInputSpec
    output_spec = _TSV2JSONOutputSpec

    def _run_interface(self, runtime):
        if not isdefined(self.inputs.output):
            output = fname_presuffix(
                self.inputs.in_file, suffix=".json", newpath=runtime.cwd, use_ext=False
            )
        else:
            output = self.inputs.output

        self._results["output"] = _tsv2json(
            in_tsv=self.inputs.in_file,
            out_json=output,
            index_column=self.inputs.index_column,
            additional_metadata=self.inputs.additional_metadata,
            drop_columns=self.inputs.drop_columns,
            enforce_case=self.inputs.enforce_case,
        )
        return runtime


def _tsv2json(
    in_tsv,
    out_json,
    index_column,
    additional_metadata=None,
    drop_columns=None,
    enforce_case=True,
):
    """
    Convert metadata from TSV format to JSON format.

    Parameters
    ----------
    in_tsv: str
        Path to the metadata in TSV format.
    out_json: str
        Path where the metadata should be saved in JSON format after
        conversion. If this is None, then a dictionary is returned instead.
    index_column: str
        Name of the column in the TSV to be used as an index (top-level key in
        the JSON).
    additional_metadata: dict
        Any additional metadata that should be applied to all entries in the
        JSON.
    drop_columns: list
        List of columns from the input TSV to be dropped from the JSON.
    enforce_case: bool
        Indicates whether BIDS case conventions should be followed. Currently,
        this means that index fields (column names in the associated data TSV)
        use snake case and other fields use camel case.

    Returns
    -------
    str
        Path to the metadata saved in JSON format.
    """
    import pandas as pd

    # Adapted from https://dev.to/rrampage/snake-case-to-camel-case-and- ...
    # back-using-regular-expressions-and-python-m9j
    re_to_camel = r"(.*?)_([a-zA-Z0-9])"
    re_to_snake = r"(^.+?|.*?)((?<![_A-Z])[A-Z]|(?<![_0-9])[0-9]+)"

    def snake(match):
        return "{}_{}".format(match.group(1).lower(), match.group(2).lower())

    def camel(match):
        return "{}{}".format(match.group(1), match.group(2).upper())

    # from fmriprep
    def less_breakable(a_string):
        """hardens the string to different envs (i.e. case insensitive, no
        whitespace, '#'"""
        return "".join(a_string.split()).strip("#")

    drop_columns = drop_columns or []
    additional_metadata = additional_metadata or {}
    try:
        tsv_data = pd.read_csv(in_tsv, "\t")
    except pd.errors.EmptyDataError:
        tsv_data = pd.DataFrame()
    for k, v in additional_metadata.items():
        tsv_data[k] = [v] * len(tsv_data.index)
    for col in drop_columns:
        tsv_data.drop(labels=col, axis="columns", inplace=True)
    if index_column in tsv_data:
        tsv_data.set_index(index_column, drop=True, inplace=True)
    if enforce_case:
        tsv_data.index = [
            re.sub(re_to_snake, snake, less_breakable(i), 0).lower()
            for i in tsv_data.index
        ]
        tsv_data.columns = [
            re.sub(re_to_camel, camel, less_breakable(i).title(), 0).replace(
                "Csf", "CSF"
            )
            for i in tsv_data.columns
        ]
    json_data = tsv_data.to_json(orient="index")
    json_data = json.JSONDecoder(object_pairs_hook=OrderedDict).decode(json_data)
    for i in json_data:
        json_data[i].update(additional_metadata)

    if out_json is None:
        return json_data

    with open(out_json, "w") as f:
        json.dump(json_data, f, indent=4)
    return out_json
