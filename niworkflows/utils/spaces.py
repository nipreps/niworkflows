"""Utilities for tracking and filtering spaces."""
import attr
import typing
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
"""A map of surface densities and legacy fsaverageX names."""


@attr.s(slots=True, frozen=True)
class Space:
    """
    Represent a (non)standard space specification.

    Examples
    --------
    >>> Space('MNI152NLin2009cAsym')
    Space(name='MNI152NLin2009cAsym', cohort=None, spec={})

    >>> Space('MNI152NLin2009cAsym', {})
    Space(name='MNI152NLin2009cAsym', cohort=None, spec={})

    >>> Space('MNI152NLin2009cAsym', None)
    Space(name='MNI152NLin2009cAsym', cohort=None, spec={})

    >>> Space('MNI152NLin2009cAsym', {'res': 1})
    Space(name='MNI152NLin2009cAsym', cohort=None, spec={'res': 1})

    >>> Space('MNIPediatricAsym', {'cohort': '1'})
    Space(name='MNIPediatricAsym', cohort='1', spec={})

    >>> Space('func')
    Space(name='func', cohort=None, spec={})

    >>> # Checks spaces with cohorts:
    >>> Space('MNIPediatricAsym')
    Traceback (most recent call last):
      ...
    ValueError: standard space "MNIPediatricAsym" is not fully defined.
    ...

    >>> Space(name='MNI152Lin', spec={'cohort': 1})
    Traceback (most recent call last):
      ...
    ValueError: standard space "MNI152Lin" does not contain ...

    >>> Space('MNIPediatricAsym', {'cohort': '100'})
    Traceback (most recent call last):
      ...
    ValueError: standard space "MNIPediatricAsym" does not contain ...
    ...

    >>> Space('MNIPediatricAsym', 'blah')
    Traceback (most recent call last):
      ...
    TypeError: invalid space specification...

    >>> Space('shouldraise')
    Traceback (most recent call last):
      ...
    ValueError: space identifier "shouldraise" is invalid.
    ...

    >>> # Correctly assigns the density of legacy "fsaverage":
    >>> Space(name='fsaverage')
    Space(name='fsaverage', cohort=None, spec={'den': '164k'})
    >>> Space(name='fsaverage6')
    Space(name='fsaverage', cohort=None, spec={'den': '41k'})

    >>> # Overwrites density of legacy "fsaverage" specifications
    >>> Space(name='fsaverage6', spec={'den': '10k'})
    Space(name='fsaverage', cohort=None, spec={'den': '41k'})

    >>> Space('func').standard
    False

    >>> Space('MNI152Lin').standard
    True

    >>> Space('MNIPediatricAsym', {'cohort': 1}).standard
    True

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

    >>> Space.from_string("MNI152NLin2009cAsym")
    [Space(name='MNI152NLin2009cAsym', cohort=None, spec={})]

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
    [Space(name='MNIPediatricAsym', cohort='1', spec={})]

    >>> Space.from_string("MNIPediatricAsym:cohort-1:cohort-2")
    [Space(name='MNIPediatricAsym', cohort='1', spec={}),
     Space(name='MNIPediatricAsym', cohort='2', spec={})]

    >>> Space.from_string("MNIPediatricAsym:cohort-5:cohort-6:res-2")
    [Space(name='MNIPediatricAsym', cohort='5', spec={'res': '2'}),
     Space(name='MNIPediatricAsym', cohort='6', spec={'res': '2'})]

    >>> Space.from_string("MNIPediatricAsym:cohort-5:cohort-6:res-2:res-iso1.6mm")
    [Space(name='MNIPediatricAsym', cohort='5', spec={'res': '2'}),
     Space(name='MNIPediatricAsym', cohort='5', spec={'res': 'iso1.6mm'}),
     Space(name='MNIPediatricAsym', cohort='6', spec={'res': '2'}),
     Space(name='MNIPediatricAsym', cohort='6', spec={'res': 'iso1.6mm'})]

    """

    _standard_spaces = tuple(_tfapi.templates())

    name: str = attr.ib(default=None)
    """Unique name designating this space."""
    cohort: str = attr.ib(init=False, default=None)
    """An attribute to accomodate cohorts from TemplateFlow."""
    spec: typing.Dict = attr.ib(factory=dict)
    """The dictionary of specs."""
    standard: bool = attr.ib(default=False, repr=False)
    """Whether this space is standard or not."""
    dim: int = attr.ib(default=3, repr=False)
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

        if self.spec and 'cohort' in self.spec:
            spec = self.spec.copy()
            value = spec.pop('cohort')
            self._check_cohort('cohort', value)
            object.__setattr__(self, "cohort", value)
            object.__setattr__(self, "spec", spec)
            return

        _cohorts = _tfapi.TF_LAYOUT.get_cohorts(template=self.name)
        if _cohorts:
            _cohorts = ', '.join(['"cohort-%s"' % c for c in _cohorts])
            raise ValueError(
                'standard space "%s" is not fully defined.\n'
                'Set a valid cohort selector from: %s.' % (self.name, _cohorts))

    @name.validator
    def _check_name(self, attribute, value):
        if value.startswith('fsaverage'):
            return
        valid = list(self._standard_spaces) + NONSTANDARD_REFERENCES
        if value not in valid:
            raise ValueError(
                'space identifier "%s" is invalid.\nValid '
                'identifiers are: %s' % (value, ', '.join(valid)))

    @cohort.validator
    def _check_cohort(self, attribute, value):
        valid = ['%s' % c for c in _tfapi.TF_LAYOUT.get_cohorts(
            template=self.name)]
        if value is not None and str(value) not in valid:
            raise ValueError(
                'standard space "%s" does not contain any cohort '
                'named "%s".' % (self.name, value))

    @spec.validator
    def _check_spec(self, attribute, value):
        if value is not None and not isinstance(value, dict):
            raise TypeError(
                "invalid space specification: %s." % str(value))

    @classmethod
    def from_string(cls, value):
        """Create a new Space from string."""
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
    [Space(name='func', cohort=None, spec={}),
     Space(name='anat', cohort=None, spec={})]

    >>> sp.get_std_spaces()
    ['MNI152NLin2009cAsym', 'fsaverage', 'MNIPediatricAsym:cohort-2']

    >>> sp.get_std_spaces(dim=(3,))
    ['MNI152NLin2009cAsym', 'MNIPediatricAsym:cohort-2']

    >>> sp.get_templates()
    [Space(name='fsaverage', cohort=None, spec={'den': '10k'}),
     Space(name='fsaverage', cohort=None, spec={'den': '41k'}),
     Space(name='MNI152NLin2009cAsym', cohort=None, spec={'res': 2}),
     Space(name='MNI152NLin2009cAsym', cohort=None, spec={'res': 1})]

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

    __slots__ = ('_spaces')
    standard_spaces = tuple(_tfapi.templates())
    """List of supported standard reference spaces."""

    @staticmethod
    def check_space(space):
        """Build a :class:`Space` object."""
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
            self.__add__(spaces)

    def __add__(self, b):
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

    def __repr__(self):
        """Representation of this object."""
        return 'Imaging spaces: %s' % ', '.join(self._spaces)

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

    def get_std_spaces(self, dim=(2, 3)):
        """Return only standard spaces."""
        names = []
        for s in self._spaces:
            name = ':cohort-'.join((s.name, s.cohort)) if s.cohort else s.name
            if s.standard and s.dim in dim and name not in names:
                names.append(name)

        return names

    def get_templates(self, dim=(2, 3)):
        """Return output spaces."""
        return [s for s in self._spaces
                if s.standard and s.spec and s.dim in dim]

    def get_nonstd_spaces(self, only_names=True, dim=(2, 3)):
        """Return nonstandard spaces."""
        return [s.name if only_names else s
                for s in self._spaces
                if not s.standard and s.dim in dim]


def _expand_entities(entities):
    """
    Generate multiple replacement queries based on all combinations of values.

    Ported from PyBIDS

    Examples
    --------
    >>> entities = {'subject': ['01', '02'], 'session': ['1', '2'], 'task': ['rest', 'finger']}
    >>> out = _expand_entities(entities)
    >>> len(out)
    8
    >>> {'subject': '01', 'session': '1', 'task': 'rest'} in out
    True
    >>> {'subject': '02', 'session': '1', 'task': 'rest'} in out
    True
    >>> {'subject': '01', 'session': '2', 'task': 'rest'} in out
    True
    >>> {'subject': '02', 'session': '2', 'task': 'rest'} in out
    True
    >>> {'subject': '01', 'session': '1', 'task': 'finger'} in out
    True
    >>> {'subject': '02', 'session': '1', 'task': 'finger'} in out
    True
    >>> {'subject': '01', 'session': '2', 'task': 'finger'} in out
    True
    >>> {'subject': '02', 'session': '2', 'task': 'finger'} in out
    True
    """
    keys = list(entities.keys())
    values = list(product(*[entities[k] for k in keys]))
    return [{k: v for k, v in zip(keys, combs)} for combs in values]
