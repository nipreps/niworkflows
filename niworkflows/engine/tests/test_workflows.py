"""Test the LiterateWorkflow."""
from nipype.pipeline.engine import Node
from nipype.interfaces import afni, utility as niu
from ..workflows import LiterateWorkflow as Workflow


def _reorient_wf(name="ReorientWorkflow"):
    """A workflow to reorient images to 'RPI' orientation."""
    workflow = Workflow(name=name)
    workflow.__desc__ = "Inner workflow. "
    inputnode = Node(niu.IdentityInterface(fields=["in_file"]), name="inputnode")
    outputnode = Node(niu.IdentityInterface(fields=["out_file"]), name="outputnode")
    deoblique = Node(afni.Refit(deoblique=True), name="deoblique")
    reorient = Node(
        afni.Resample(orientation="RPI", outputtype="NIFTI_GZ"), name="reorient"
    )
    workflow.connect(
        [
            (inputnode, deoblique, [("in_file", "in_file")]),
            (deoblique, reorient, [("out_file", "in_file")]),
            (reorient, outputnode, [("out_file", "out_file")]),
        ]
    )
    return workflow


def test_boilerplate():
    """Check the boilerplate is generated."""
    workflow = Workflow(name="test")
    workflow.__desc__ = "Outer workflow. "
    workflow.__postdesc__ = "Outer workflow (postdesc)."

    inputnode = Node(niu.IdentityInterface(fields=["in_file"]), name="inputnode")
    inner = _reorient_wf()

    # fmt: off
    workflow.connect([
        (inputnode, inner, [("in_file", "inputnode.in_file")]),
    ])
    # fmt: on

    assert (
        workflow.visit_desc()
        == "Outer workflow. Inner workflow. Outer workflow (postdesc)."
    )
