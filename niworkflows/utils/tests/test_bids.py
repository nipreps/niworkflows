import json

import pytest
from ..bids import check_pipeline_version


def test_check_pipeline_version(tmp_path):
    data = {"PipelineDescription": {"Version": "1.1.1"}}
    desc = tmp_path / 'dataset_description.json'
    with open(str(desc), 'wt') as fp:
        json.dump(data, fp)

    check_pipeline_version('1.1.1', str(desc))
    with pytest.warns(UserWarning):
        check_pipeline_version('1.2', desc)
