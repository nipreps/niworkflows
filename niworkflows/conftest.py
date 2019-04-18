"""py.test configuration"""
import os
from pathlib import Path
import numpy
import pytest

from .utils.bids import collect_data

test_data_env = os.getenv('TEST_DATA_HOME',
                          str(Path.home() / '.cache' / 'stanford-crn'))
data_dir = Path(test_data_env) / 'BIDS-examples-1-enh-ds054'


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace['np'] = numpy
    doctest_namespace['os'] = os
    doctest_namespace['Path'] = Path
    doctest_namespace['datadir'] = data_dir
    doctest_namespace['bids_collect_data'] = collect_data


@pytest.fixture
def testdata_dir():
    return data_dir
