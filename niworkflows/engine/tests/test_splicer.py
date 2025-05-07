from __future__ import annotations

import pytest
from nipype import Node, Workflow
from nipype.interfaces.base import BaseInterfaceInputSpec, SimpleInterface, TraitedSpec, traits
from nipype.interfaces.utility import IdentityInterface

from ..splicer import splice_workflow, tag


class _NullInterfaceInputSpec(BaseInterfaceInputSpec):
    in1 = traits.Int(default=0, usedefault=True, desc='Input 1')
    in2 = traits.Int(default=0, usedefault=True, desc='Input 2')


class _NullInterfaceOutputSpec(TraitedSpec):
    out1 = traits.Int(desc='Output 1')
    out2 = traits.Int(desc='Output 2')


class NullInterface(SimpleInterface):
    """
    A simple interface that does nothing.
    """

    input_spec = _NullInterfaceInputSpec
    output_spec = _NullInterfaceOutputSpec

    def _run_interface(self, runtime):
        self._results['out1'] = self.inputs.in1
        self._results['out2'] = self.inputs.in2
        return runtime


def _create_nested_null_wf(name: str, tag: str | None = None):
    wf = Workflow(name=name)
    if tag:
        wf._tag = tag

    inputnode = Node(IdentityInterface(fields=['in1', 'in2']), name='inputnode')
    outputnode = Node(IdentityInterface(fields=['out1', 'out2']), name='outputnode')

    n1 = Node(NullInterface(), name='null1')
    n2_wf = _create_null_wf('nested_wf', tag='nested')
    n3 = Node(NullInterface(), name='null3')

    wf.connect([
        (inputnode, n1, [
            ('in1', 'in1'),
            ('in2', 'in2'),
        ]),
        (n1, n2_wf, [('out1', 'inputnode.in1')]),
        (n2_wf, n3, [('outputnode.out1', 'in1')]),
        (n3, outputnode, [
            ('out1', 'out1'),
            ('out2', 'out2'),
        ]),
    ])  # fmt:skip
    return wf


def _create_null_wf(name: str, tag: str | None = None):
    wf = Workflow(name=name)
    if tag:
        wf._tag = tag

    inputnode = Node(IdentityInterface(fields=['in1', 'in2']), name='inputnode')
    outputnode = Node(IdentityInterface(fields=['out1', 'out2']), name='outputnode')

    n1 = Node(NullInterface(), name='null1')
    n2 = Node(NullInterface(), name='null2')
    n3 = Node(NullInterface(), name='null3')

    wf.connect([
        (inputnode, n1, [
            ('in1', 'in1'),
            ('in2', 'in2'),
        ]),
        (n1, n2, [('out1', 'in1')]),
        (n2, n3, [('out1', 'in1')]),
        (n3, outputnode, [
            ('out1', 'out1'),
            ('out2', 'out2'),
        ]),
    ])  # fmt:skip
    return wf


@pytest.fixture
def wf0(tmp_path) -> Workflow:
    """
    Create a tagged workflow.
    """
    wf = Workflow(name='root', base_dir=tmp_path)
    wf._tag = 'root'

    inputnode = Node(IdentityInterface(fields=['in1', 'in2']), name='inputnode')
    inputnode.inputs.in1 = 1
    inputnode.inputs.in2 = 2
    outputnode = Node(IdentityInterface(fields=['out1', 'out2']), name='outputnode')

    a_in = Node(IdentityInterface(fields=['in1', 'in2']), name='a_in')
    a_wf = _create_null_wf('a_wf', tag='a')
    a_out = Node(IdentityInterface(fields=['out1', 'out2']), name='a_out')

    b_in = Node(IdentityInterface(fields=['in1', 'in2']), name='b_in')
    b_wf = _create_nested_null_wf('b_wf', tag='b')
    b_out = Node(IdentityInterface(fields=['in1', 'out2']), name='b_out')

    wf.connect([
        (inputnode, a_in, [
            ('in1', 'in1'),
            ('in2', 'in2'),
        ]),
        (a_in, a_wf, [
            ('in1', 'inputnode.in1'),
            ('in2', 'inputnode.in2'),
        ]),
        (a_wf, a_out, [
            ('outputnode.out1', 'out1'),
            ('outputnode.out2', 'out2'),
        ]),
        (a_out, b_in, [
            ('out1', 'in1'),
            ('out2', 'in2'),
        ]),
        (b_in, b_wf, [
            ('in1', 'inputnode.in1'),
            ('in2', 'inputnode.in2'),
        ]),
        (b_wf, b_out, [
            ('outputnode.out1', 'out1'),
            ('outputnode.out2', 'out2'),
        ]),
        (a_out, outputnode, [
            ('out1', 'out1'),
        ]),
        (b_out, outputnode, [
            ('out2', 'out2'),
        ]),
    ])  # fmt:skip
    return wf


def test_splice(wf0):
    replacements = {
        'a': _create_null_wf('a2_wf', tag='a'),
        'nested': _create_null_wf('nested2_wf', tag='nested'),
        'c': _create_null_wf('c_wf', tag='c'),
    }
    wf = splice_workflow(wf0, replacements)

    assert wf.get_node('a2_wf')
    assert wf.get_node('b_wf').get_node('nested2_wf')
    assert wf.get_node('c_wf') is None


@pytest.mark.parametrize('name', ['foo'])
def test_tag(name):
    @tag(name)
    def init_workflow(name, *, xarg: str):
        return Workflow(name=name)

    wf = init_workflow(name, xarg='bar')
    assert wf._tag == name
