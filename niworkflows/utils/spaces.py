"""Utilities for tracking and filtering spaces."""
import argparse
import attr
from collections import defaultdict
from itertools import product
from templateflow import api as _tfapi

NONSTANDARD_REFERENCES = [
    'T1w',
    'T2w',
    'anat',
    'fsnative',
    'func',
    'run',
    'sbref',
    'session',
]
"""List of supported nonstandard reference spaces."""


FSAVERAGE_DENSITY = {
    'fsaverage3': '642',
    'fsaverage4': '2562',
    'fsaverage5': '10k',
    'fsaverage6': '41k',
    'fsaverage': '164k',
}
"""A map of legacy fsaverageX names to surface densities."""

FSAVERAGE_LEGACY = {v: k for k, v in FSAVERAGE_DENSITY.items()}
"""A map of surface densities to legacy fsaverageX names."""


@attr.s(slots=True, frozen=True)
class Space:
    """
    Represent a (non)standard space specification.

    Examples
    --------
    >>> Space('MNI152NLin2009cAsym')
    Space(name='MNI152NLin2009cAsym', spec={})

    >>> Space('MNI152NLin2009cAsym', {})
    Space(name='MNI152NLin2009cAsym', spec={})

    >>> Space('MNI152NLin2009cAsym', None)
    Space(name='MNI152NLin2009cAsym', spec={})

    >>> Space('MNI152NLin2009cAsym', {'res': 1})
    Space(name='MNI152NLin2009cAsym', spec={'res': 1})

    >>> Space('MNIPediatricAsym', {'cohort': '1'})
    Space(name='MNIPediatricAsym', spec={'cohort': '1'})

    >>> Space('func')
    Space(name='func', spec={})

    >>> # Checks spaces with cohorts:
    >>> Space('MNIPediatricAsym')
    Traceback (most recent call last):
      ...
    ValueError: standard space "MNIPediatricAsym" is not fully defined.
    ...

    >>> Space(name='MNI152Lin', spec={'cohort': 1})
    Traceback (most recent call last):
      ...
    ValueError: standard space "MNI152Lin" does not accept ...

    >>> Space('MNIPediatricAsym', {'cohort': '100'})
    Traceback (most recent call last):
      ...
    ValueError: standard space "MNIPediatricAsym" does not contain ...
    ...

    >>> Space('MNIPediatricAsym', 'blah')
    Traceback (most recent call last):
      ...
    TypeError: ...

    >>> Space('shouldraise')
    Traceback (most recent call last):
      ...
    ValueError: space identifier "shouldraise" is invalid.
    ...

    >>> # Check standard property
    >>> Space('func').standard
    False
    >>> Space('MNI152Lin').standard
    True
    >>> Space('MNIPediatricAsym', {'cohort': 1}).standard
    True

    >>> # Equality/inequality checks
    >>> Space('func') == Space('func')
    True
    >>> Space('func') != Space('MNI152Lin')
    True
    >>> Space('MNI152Lin', {'res': 1}) == Space('MNI152Lin', {'res': 1})
    True
    >>> Space('MNI152Lin', {'res': 1}) == Space('MNI152Lin', {'res': 2})
    False
    >>> sp1 = Space('MNIPediatricAsym', {'cohort': 1})
    >>> sp2 = Space('MNIPediatricAsym', {'cohort': 2})
    >>> sp1 == sp2
    False
    >>> sp1 = Space('MNIPediatricAsym', {'res': 1, 'cohort': 1})
    >>> sp2 = Space('MNIPediatricAsym', {'cohort': 1, 'res': 1})
    >>> sp1 == sp2
    True

    """

    _standard_spaces = tuple(_tfapi.templates())

    name = attr.ib(default=None, type=str)
    """Unique name designating this space."""
    spec = attr.ib(factory=dict, validator=attr.validators.optional(
        attr.validators.instance_of(dict)))
    """The dictionary of specs."""
    standard = attr.ib(default=False, repr=False, type=bool)
    """Whether this space is standard or not."""
    dim = attr.ib(default=3, repr=False, type=int)
    """Dimensionality of the sampling manifold."""

    def __attrs_post_init__(self):
        """Extract cohort out of spec."""
        if self.spec is None:
            object.__setattr__(self, "spec", {})

        if self.name.startswith('fsaverage'):
            name = self.name
            object.__setattr__(self, "name", "fsaverage")

            if 'den' not in self.spec or name != "fsaverage":
                spec = self.spec.copy()
                spec['den'] = FSAVERAGE_DENSITY[name]
                object.__setattr__(self, "spec", spec)

        if self.name.startswith('fs'):
            object.__setattr__(self, "dim", 2)

        if self.name in self._standard_spaces:
            object.__setattr__(self, "standard", True)

        _cohorts = ["%s" % t
                    for t in _tfapi.TF_LAYOUT.get_cohorts(template=self.name)]
        if "cohort" in self.spec:
            if not _cohorts:
                raise ValueError(
                    'standard space "%s" does not accept a cohort '
                    'specification.' % self.name)

            if str(self.spec["cohort"]) not in _cohorts:
                raise ValueError(
                    'standard space "%s" does not contain any cohort '
                    'named "%s".' % (self.name, self.spec["cohort"]))
        elif _cohorts:
            _cohorts = ', '.join(['"cohort-%s"' % c for c in _cohorts])
            raise ValueError(
                'standard space "%s" is not fully defined.\n'
                'Set a valid cohort selector from: %s.' % (self.name, _cohorts))

    @property
    def fullname(self):
        """
        Generate a full-name combining cohort.

        Examples
        --------
        >>> Space('MNI152Lin').fullname
        'MNI152Lin'

        >>> Space('MNIPediatricAsym', {'cohort': 1}).fullname
        'MNIPediatricAsym:cohort-1'

        """
        if "cohort" not in self.spec:
            return self.name
        return "%s:cohort-%s" % (self.name, self.spec["cohort"])

    @property
    def legacyname(self):
        """
        Generate a legacy name for fsaverageX spaces.

        Examples
        --------
        >>> Space(name='fsaverage')
        Space(name='fsaverage', spec={'den': '164k'})
        >>> Space(name='fsaverage').legacyname
        'fsaverage'
        >>> Space(name='fsaverage6')
        Space(name='fsaverage', spec={'den': '41k'})
        >>> Space(name='fsaverage6').legacyname
        'fsaverage6'
        >>> # Overwrites density of legacy "fsaverage" specifications
        >>> Space(name='fsaverage6', spec={'den': '10k'})
        Space(name='fsaverage', spec={'den': '41k'})
        >>> Space(name='fsaverage6', spec={'den': '10k'}).legacyname
        'fsaverage6'
        >>> # Return None if no legacy name
        >>> Space(name='fsaverage', spec={'den': '30k'}).legacyname is None
        True

        """
        if self.name == "fsaverage" and self.spec["den"] in FSAVERAGE_LEGACY:
            return FSAVERAGE_LEGACY[self.spec["den"]]

    @name.validator
    def _check_name(self, attribute, value):
        if value.startswith('fsaverage'):
            return
        valid = list(self._standard_spaces) + NONSTANDARD_REFERENCES
        if value not in valid:
            raise ValueError(
                'space identifier "%s" is invalid.\nValid '
                'identifiers are: %s' % (value, ', '.join(valid)))

    @classmethod
    def from_string(cls, value):
        """
        Parse a string to generate the corresponding list of Spaces.

        .. testsetup::

            >>> if PY_VERSION < (3, 6):
            ...     pytest.skip("This doctest does not work on python <3.6")

        Parameters
        ----------
        value: :obj:`str`
            A string containing a space specification following *fMRIPrep*'s
            language for ``--output-spaces``
            (e.g., ``MNIPediatricAsym:cohort-1:cohort-2:res-1:res-2``).

        Returns
        -------
        spaces : :obj:`list` of :obj:`Space`
            A list of corresponding spaces given the input string.

        Examples
        --------
        >>> Space.from_string("MNI152NLin2009cAsym")
        [Space(name='MNI152NLin2009cAsym', spec={})]

        >>> # Bad name
        >>> Space.from_string("shouldraise")
        Traceback (most recent call last):
          ...
        ValueError: space identifier "shouldraise" is invalid.
        ...

        >>> # Missing cohort
        >>> Space.from_string("MNIPediatricAsym")
        Traceback (most recent call last):
          ...
        ValueError: standard space "MNIPediatricAsym" is not fully defined.
        ...

        >>> Space.from_string("MNIPediatricAsym:cohort-1")
        [Space(name='MNIPediatricAsym', spec={'cohort': '1'})]

        >>> Space.from_string("MNIPediatricAsym:cohort-1:cohort-2")
        [Space(name='MNIPediatricAsym', spec={'cohort': '1'}),
         Space(name='MNIPediatricAsym', spec={'cohort': '2'})]

        >>> Space.from_string("fsaverage:den-10k:den-164k")
        [Space(name='fsaverage', spec={'den': '10k'}),
         Space(name='fsaverage', spec={'den': '164k'})]

        >>> Space.from_string("MNIPediatricAsym:cohort-5:cohort-6:res-2")
        [Space(name='MNIPediatricAsym', spec={'cohort': '5', 'res': '2'}),
         Space(name='MNIPediatricAsym', spec={'cohort': '6', 'res': '2'})]

        >>> Space.from_string("MNIPediatricAsym:cohort-5:cohort-6:res-2:res-iso1.6mm")
        [Space(name='MNIPediatricAsym', spec={'cohort': '5', 'res': '2'}),
         Space(name='MNIPediatricAsym', spec={'cohort': '5', 'res': 'iso1.6mm'}),
         Space(name='MNIPediatricAsym', spec={'cohort': '6', 'res': '2'}),
         Space(name='MNIPediatricAsym', spec={'cohort': '6', 'res': 'iso1.6mm'})]

        """
        _args = value.split(':')
        spec = defaultdict(list, {})
        for modifier in _args[1:]:
            mitems = modifier.split('-', 1)
            spec[mitems[0]].append(len(mitems) == 1 or mitems[1])

        allspecs = _expand_entities(spec)

        return [cls(_args[0], s) for s in allspecs]


class SpatialReferences:
    """
    Manage specifications of spatial references.

    Examples
    --------
    >>> sp = SpatialReferences([
    ...     'func',
    ...     'fsnative',
    ...     'MNI152NLin2009cAsym',
    ...     'anat',
    ...     'fsaverage5',
    ...     'fsaverage6',
    ...     ('MNIPediatricAsym', {'cohort': '2'}),
    ...     ('MNI152NLin2009cAsym', {'res': 2}),
    ...     ('MNI152NLin2009cAsym', {'res': 1}),
    ... ])
    >>> sp.get_nonstd_spaces()
    ['func', 'fsnative', 'anat']

    >>> sp.get_nonstd_spaces(dim=(3,))
    ['func', 'anat']

    >>> sp.get_nonstd_spaces(only_names=False, dim=(3,))
    [Space(name='func', spec={}),
     Space(name='anat', spec={})]

    >>> sp.get_std_spaces()
    ['MNI152NLin2009cAsym', 'fsaverage', 'MNIPediatricAsym:cohort-2']

    >>> sp.get_std_spaces(dim=(3,))
    ['MNI152NLin2009cAsym', 'MNIPediatricAsym:cohort-2']

    >>> sp.get_fs_spaces()
    ['fsnative', 'fsaverage5', 'fsaverage6']

    >>> sp.get_templates()
    [Space(name='fsaverage', spec={'den': '10k'}),
     Space(name='fsaverage', spec={'den': '41k'}),
     Space(name='MNI152NLin2009cAsym', spec={'res': 2}),
     Space(name='MNI152NLin2009cAsym', spec={'res': 1})]

    >>> sp.add(('MNIPediatricAsym', {'cohort': '2'}))
    >>> sp.get_std_spaces(dim=(3,))
    ['MNI152NLin2009cAsym', 'MNIPediatricAsym:cohort-2']

    >>> sp += [('MNIPediatricAsym', {'cohort': '2'})]
    Traceback (most recent call last):
      ...
    ValueError: space ...

    >>> sp += [('MNIPediatricAsym', {'cohort': '1'})]
    >>> sp.get_std_spaces(dim=(3,))
    ['MNI152NLin2009cAsym', 'MNIPediatricAsym:cohort-2', 'MNIPediatricAsym:cohort-1']

    >>> sp.insert(0, ('MNIPediatricAsym', {'cohort': '3'}))
    >>> sp.get_std_spaces(dim=(3,))
    ['MNIPediatricAsym:cohort-3',
     'MNI152NLin2009cAsym',
     'MNIPediatricAsym:cohort-2',
     'MNIPediatricAsym:cohort-1']

    >>> sp.insert(0, ('MNIPediatricAsym', {'cohort': '3'}))
    Traceback (most recent call last):
      ...
    ValueError: space ...

    """

    __slots__ = ('_spaces',)
    standard_spaces = tuple(_tfapi.templates())
    """List of supported standard reference spaces."""

    @staticmethod
    def check_space(space):
        """Build a :class:`Space` object."""
        try:
            if isinstance(space, Space):
                return space
        except IndexError:
            pass

        spec = {}
        if not isinstance(space, str):
            try:
                spec = space[1] or {}
            except IndexError:
                pass
            except TypeError:
                space = (None, )

            space = space[0]
        return Space(space, spec)

    def __init__(self, spaces=None):
        """
        Maintain the bookkeeping of spaces and templates.

        Internal spaces are normalizations required for pipeline execution which
        can vary based on user arguments.
        Output spaces are desired user outputs.
        """
        self._spaces = []
        if spaces is not None:
            if isinstance(spaces, str):
                spaces = [spaces]
            self.__iadd__(spaces)

    def __iadd__(self, b):
        """Append a list of transforms to the internal list."""
        if not isinstance(b, (list, tuple)):
            raise TypeError('Must be a list.')

        for space in b:
            self.append(space)
        return self

    def __contains__(self, item):
        """Implement the ``in`` builtin."""
        if not self._spaces:
            return False
        item = self.check_space(item)
        for s in self._spaces:
            if s == item:
                return True
        return False

    def __str__(self):
        """
        Representation of this object.

        Examples
        --------
        >>> print(SpatialReferences())
        Spatial References: <none>.

        >>> print(SpatialReferences(['MNI152NLin2009cAsym']))
        Spatial References:
            - Space(name='MNI152NLin2009cAsym', spec={})

        >>> print(SpatialReferences(['MNI152NLin2009cAsym', 'fsaverage5']))
        Spatial References:
            - Space(name='MNI152NLin2009cAsym', spec={})
            - Space(name='fsaverage', spec={'den': '10k'})

        """
        spaces = '\n    - '.join([''] + [str(s) for s in self.spaces]) \
            if self.spaces else ' <none>.'
        return 'Spatial References:%s' % spaces

    @property
    def spaces(self):
        """Get all specified spaces."""
        return self._spaces

    @spaces.setter
    def spaces(self, value):
        self._spaces = value

    def add(self, value):
        """Add one more space, without erroring if it exists."""
        if value not in self:
            self.spaces += [self.check_space(value)]

    def append(self, value):
        """Concatenate one more space."""
        if value not in self:
            self.spaces += [self.check_space(value)]
            return

        raise ValueError('space "%s" already in spaces.' % str(value))

    def insert(self, index, value, error=True):
        """Concatenate one more space."""
        if value not in self:
            self.spaces.insert(index, self.check_space(value))
        elif error is True:
            raise ValueError('space "%s" already in spaces.' % str(value))

    def get_std_spaces(self, only_names=True, dim=(2, 3)):
        """Return only standard spaces."""
        names = []
        std_spaces = []
        for s in self._spaces:
            if s.standard and s.dim in dim and s.fullname not in names:
                names.append(s.fullname)
                std_spaces.append(s)
        if only_names:
            return names
        return std_spaces

    def get_templates(self, dim=(2, 3)):
        """Return output spaces."""
        return [s for s in self._spaces
                if s.standard and hasspec(s.spec, ('res', 'den')) and s.dim in dim]

    def get_nonstd_spaces(self, only_names=True, dim=(2, 3)):
        """Return nonstandard spaces."""
        return [s.name if only_names else s
                for s in self._spaces
                if not s.standard and s.dim in dim]

    def get_fs_spaces(self):
        """
        Return FreeSurfer spaces.

        Discards nonlegacy fsaverage values (i.e., with nonstandard density value).

        Examples
        --------
        >>> SpatialReferences([
        ...     'fsnative',
        ...     'fsaverage6',
        ...     'fsaverage5',
        ...     'MNI152NLin6Asym',
        ... ]).get_fs_spaces()
        ['fsnative', 'fsaverage6', 'fsaverage5']

        >>> SpatialReferences([
        ...     'fsnative',
        ...     'fsaverage6',
        ...     Space(name='fsaverage', spec={'den': '30k'})
        ... ]).get_fs_spaces()
        ['fsnative', 'fsaverage6']

        """
        return [s.legacyname or s.name
                for s in self._spaces
                if s.legacyname or s.name == "fsnative"]


class OutputSpacesAction(argparse.Action):
    """Parse spatial references."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Execute parser."""
        spaces = getattr(namespace, self.dest) or SpatialReferences()
        for val in values:
            for sp in Space.from_string(val):
                spaces.add(sp)
        setattr(namespace, self.dest, spaces)


def hasspec(value, specs):
    """Check whether any of the keys are in a dict."""
    for s in specs:
        if s in value:
            return True
    return False


def _expand_entities(entities):
    """
    Generate multiple replacement queries based on all combinations of values.

    Ported from PyBIDS


    .. testsetup::

        >>> if PY_VERSION < (3, 6):
        ...     pytest.skip("This doctest does not work on python <3.6")

    Examples
    --------
    >>> entities = {'subject': ['01', '02'], 'session': ['1', '2'], 'task': ['rest', 'finger']}
    >>> _expand_entities(entities)
    [{'subject': '01', 'session': '1', 'task': 'rest'},
     {'subject': '01', 'session': '1', 'task': 'finger'},
     {'subject': '01', 'session': '2', 'task': 'rest'},
     {'subject': '01', 'session': '2', 'task': 'finger'},
     {'subject': '02', 'session': '1', 'task': 'rest'},
     {'subject': '02', 'session': '1', 'task': 'finger'},
     {'subject': '02', 'session': '2', 'task': 'rest'},
     {'subject': '02', 'session': '2', 'task': 'finger'}]

    """
    keys = list(entities.keys())
    values = list(product(*[entities[k] for k in keys]))
    return [{k: v for k, v in zip(keys, combs)} for combs in values]
