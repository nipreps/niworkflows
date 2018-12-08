"""py.test configuration"""
import os
from pathlib import Path
import numpy
import pytest

from . import data
from .utils.bids import collect_data

data_dir = data.get_bids_examples(variant='BIDS-examples-1-enh-ds054')


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace['np'] = numpy
    doctest_namespace['os'] = os
    doctest_namespace['Path'] = Path
    doctest_namespace['datadir'] = Path(data_dir)
    doctest_namespace['bids_collect_data'] = collect_data
