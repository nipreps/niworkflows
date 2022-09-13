"""KeySelect tests."""
from pathlib import Path
import pytest
from ..utility import KeySelect
from ..utils import _tsv2json


def test_KeySelect():
    """Test KeySelect."""
    with pytest.raises(ValueError):
        KeySelect(fields="field1", keys=["a", "b", "c", "a"])

    with pytest.raises(ValueError):
        KeySelect(fields=[])


def test_tsv2json(tmp_path):
    Path.write_bytes(tmp_path / 'empty.tsv', bytes())
    res = _tsv2json(tmp_path / 'empty.tsv', None, 'any_column')
    assert res == {}
    res = _tsv2json(tmp_path / 'empty.tsv', None, 'any_column', additional_metadata={'a': 'b'})
    assert res == {}
