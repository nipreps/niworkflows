"""py.test configuration"""
import os
import numpy
import pytest
from pathlib import Path

from . import data

data_dir = data.get_bids_examples(variant='BIDS-examples-1-enh-ds054')


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace['np'] = numpy
    doctest_namespace['os'] = os
    doctest_namespace["datadir"] = Path(data_dir)
