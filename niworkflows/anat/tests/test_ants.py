import pytest
from nipype.pipeline.engine import Workflow

from ..ants import init_atropos_wf, init_brain_extraction_wf, init_n4_only_wf


def _input_sources(wf, node_name, input_name):
    """Return the names of nodes feeding ``input_name`` of ``node_name``."""
    node = wf.get_node(node_name)
    return {
        src.name
        for src, _, data in wf._graph.in_edges(node, data=True)
        for _, target in data['connect']
        if target == input_name
    }


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


def test_brain_extraction_wf_n4_rescale_masked():
    """N4 ignores ``--rescale-intensities`` unless a mask is passed."""
    wf = init_brain_extraction_wf(atropos_refine=False)
    n4 = wf.get_node('inu_n4_final')
    assert n4.inputs.rescale_intensities is True
    assert _input_sources(wf, 'inu_n4_final', 'mask_image') == {'thr_brainmask'}


def test_atropos_wf_n4_rescale_masked():
    """N4 ignores ``--rescale-intensities`` unless a mask is passed."""
    wf = init_atropos_wf()
    n4 = wf.get_node('inu_n4_final')
    assert n4.inputs.rescale_intensities is True
    assert _input_sources(wf, 'inu_n4_final', 'mask_image') == {'msk_conform'}


@pytest.mark.parametrize('atropos_refine', [True, False])
def test_n4_only_wf_n4_rescale_masked(atropos_refine):
    """N4 ignores ``--rescale-intensities`` unless a mask is passed."""
    wf = init_n4_only_wf(atropos_refine=atropos_refine)
    n4 = wf.get_node('inu_n4')
    assert n4.inputs.rescale_intensities is True
    assert _input_sources(wf, 'inu_n4', 'mask_image') == {'binarize'}
