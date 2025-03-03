import pytest
from nipype.pipeline.engine import Workflow

from ..ants import init_brain_extraction_wf, init_n4_only_wf


@pytest.mark.parametrize('atropos_refine', [True, False])
@pytest.mark.parametrize('use_laplacian', [True, False])
@pytest.mark.parametrize('template', ['OASIS30ANTs', 'MNI152NLin2009cAsym', 'MNI152NLin6Asym'])
def test_brain_extraction_wf_smoketest(atropos_refine, use_laplacian, template):
    wf = init_brain_extraction_wf(
        in_template=template,
        atropos_refine=atropos_refine,
        use_laplacian=use_laplacian,
    )
    assert isinstance(wf, Workflow)


@pytest.mark.parametrize('atropos_refine', [True, False])
def test_n4_only_wf_smoketest(atropos_refine):
    wf = init_n4_only_wf(atropos_refine=atropos_refine)
    assert isinstance(wf, Workflow)
