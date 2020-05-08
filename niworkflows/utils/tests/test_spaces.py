"""Test spaces."""
import pytest
from ..spaces import Reference, SpatialReferences, OutputReferencesAction


@pytest.fixture
def parser():
    """Create a parser."""
    import argparse

    pars = argparse.ArgumentParser()
    pars.add_argument(
        "--spaces",
        nargs="*",
        default=SpatialReferences(),
        action=OutputReferencesAction,
        help="user defined spaces",
    )
    return pars


@pytest.mark.parametrize(
    "spaces, expected",
    [
        (("MNI152NLin6Asym",), ("MNI152NLin6Asym:res-native",)),
        (
            ("fsaverage:den-10k", "MNI152NLin6Asym"),
            ("fsaverage:den-10k", "MNI152NLin6Asym:res-native"),
        ),
        (
            ("fsaverage:den-10k:den-30k", "MNI152NLin6Asym:res-1:res-2"),
            (
                "fsaverage:den-10k",
                "fsaverage:den-30k",
                "MNI152NLin6Asym:res-1",
                "MNI152NLin6Asym:res-2",
            ),
        ),
        (
            ("fsaverage:den-10k:den-30k", "MNI152NLin6Asym:res-1:res-2", "fsaverage5"),
            (
                "fsaverage:den-10k",
                "fsaverage:den-30k",
                "MNI152NLin6Asym:res-1",
                "MNI152NLin6Asym:res-2",
            ),
        ),
        (
            (
                "fsaverage:den-10k:den-30k",
                "MNI152NLin6Asym:res-1:res-2",
                "fsaverage:den-10k:den-30k",
                "MNI152NLin6Asym:res-1:res-2",
            ),
            (
                "fsaverage:den-10k",
                "fsaverage:den-30k",
                "MNI152NLin6Asym:res-1",
                "MNI152NLin6Asym:res-2",
            ),
        ),
        (("MNI152NLin6Asym", "func"), ("MNI152NLin6Asym:res-native", "func")),
    ],
)
def test_space_action(parser, spaces, expected):
    """Test action."""
    pargs = parser.parse_known_args(args=("--spaces",) + spaces)[0]
    parsed_spaces = pargs.spaces
    assert isinstance(parsed_spaces, SpatialReferences)
    assert all(
        isinstance(sp, Reference) for sp in parsed_spaces.references
    ), "Every element must be a `Reference`"
    assert len(parsed_spaces.references) == len(expected)
    for ref, expected_ref in zip(parsed_spaces.references, expected):
        assert str(ref) == expected_ref


@pytest.mark.parametrize("flag,expected", [(("--spaces",), True), (None, False)])
def test_space_action_edgecases(parser, flag, expected):
    pargs = parser.parse_known_args(flag)[0]
    spaces = pargs.spaces
    assert spaces.is_cached() is expected
