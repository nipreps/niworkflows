"""KeySelect tests."""
import pytest
from ..utility import KeySelect


def test_KeySelect():
    """Test KeySelect."""
    with pytest.raises(ValueError):
        KeySelect(fields="field1", keys=["a", "b", "c", "a"])

    with pytest.raises(ValueError):
        KeySelect(fields=[])
