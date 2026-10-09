from ..skullstrip import afni_wf


def _input_sources(wf, node_name, input_name):
    """Return the names of nodes feeding ``input_name`` of ``node_name``."""
    node = wf.get_node(node_name)
    return {
        src.name
        for src, _, data in wf._graph.in_edges(node, data=True)
        for _, target in data['connect']
        if target == input_name
    }


def test_afni_wf_no_rescale_without_mask():
    """N4 ignores ``--rescale-intensities`` unless a mask is passed.

    This workflow runs N4 before skull-stripping, so no mask exists and
    the option must not be set.
    """
    wf = afni_wf()
    n4 = wf.get_node('inu_n4')
    assert not n4.inputs.rescale_intensities
    assert _input_sources(wf, 'inu_n4', 'mask_image') == set()
