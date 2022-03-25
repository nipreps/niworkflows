import pytest
from functools import wraps
import os
from pathlib import Path
from nipype.interfaces import fsl, freesurfer as fs, afni

test_data_env = os.getenv(
    "TEST_DATA_HOME", str(Path.home() / ".cache" / "stanford-crn")
)
test_output_dir = os.getenv("TEST_OUTPUT_DIR")
test_workdir = os.getenv("TEST_WORK_DIR")

data_dir = Path(test_data_env) / "BIDS-examples-1-enh-ds054"


def create_canary(predicate, message):
    def canary():
        if predicate:
            pytest.skip(message)

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            canary()
            return f(*args, **kwargs)
        return wrapper

    return canary, decorator


data_env_canary, needs_data_env = create_canary(
    not os.path.isdir(test_data_env),
    "Test data must be made available in ~/.cache/stanford-crn or in a "
    "directory referenced by the TEST_DATA_HOME environment variable.")

data_dir_canary, needs_data_dir = create_canary(
    not os.path.isdir(data_dir),
    "Test data must be made available in ~/.cache/stanford-crn or in a "
    "directory referenced by the TEST_DATA_HOME environment variable.")

has_fsl = fsl.Info.version() is not None
has_freesurfer = fs.Info.version() is not None
has_afni = afni.Info.version() is not None
