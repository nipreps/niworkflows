"""Test spaces."""
import pytest
from ..spaces import Reference, SpatialReferences, OutputReferencesAction


@pytest.fixture
def parser():
    """Create a parser."""
    import argparse

    pars = argparse.ArgumentParser()
    pars.add_argument('--spaces', nargs='+', action=OutputReferencesAction,
                      help='user defined spaces')
    return pars


@pytest.mark.parametrize("spaces,expected", [
    (("MNI152NLin6Asym",), 1),
    (("fsaverage:den-10k", "MNI152NLin6Asym"), 2),
    (("fsaverage:den-10k:den-30k", "MNI152NLin6Asym:res-1:res-2"), 4),
    (("fsaverage:den-10k:den-30k", "MNI152NLin6Asym:res-1:res-2",
      "fsaverage5"), 4),
    (("fsaverage:den-10k:den-30k", "MNI152NLin6Asym:res-1:res-2",
      "fsaverage:den-10k:den-30k", "MNI152NLin6Asym:res-1:res-2"), 4),
])
def test_space_action(parser, spaces, expected):
    """Test action."""
    pargs = parser.parse_args(args=('--spaces',) + spaces)
    parsed_spaces = pargs.spaces
    assert isinstance(parsed_spaces, SpatialReferences)
    assert all(isinstance(sp, Reference) for sp in parsed_spaces.references), \
        "Every element must be a `Reference`"
    assert len(parsed_spaces.references) == expected
