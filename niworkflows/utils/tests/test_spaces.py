import pytest

from ..spaces import Space, SpatialReferences, StoreSpacesAction


@pytest.fixture
def parser():
    import argparse

    pars = argparse.ArgumentParser()
    pars.add_argument('--spaces', nargs='+', action=StoreSpacesAction,
                      help='user defined spaces')
    return pars


@pytest.mark.parametrize("spaces,expected", [
    (("MNI152NLin6Asym",), 1),
    (("fsaverage:den-10k", "MNI152NLin6Asym"), 2),
    (("fsaverage:den-10k:den-30k", "MNI152NLin6Asym:res-1:res-2"), 4),
])
def test_space_action(parser, spaces, expected):
    pargs = parser.parse_args(args=('--spaces',) + spaces)
    parsed_spaces = pargs.spaces
    assert all(isinstance(sp, Space) for sp in parsed_spaces), "Every element must be a `Space`"

    sparef = SpatialReferences()
    sparef += parsed_spaces
    assert len(sparef.spaces) == len(parsed_spaces) == expected
