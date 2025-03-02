import logging
from types import SimpleNamespace

import pytest
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from ..plugin import MultiProcPlugin


def add(x, y):  # the Function interface does not support builtin functions
    return x + y


def addall(inlist):
    import time

    time.sleep(0.2)  # Simulate some work
    return sum(inlist)


@pytest.fixture
def workflow(tmp_path):
    workflow = pe.Workflow(name='test_wf', base_dir=tmp_path)

    inputnode = pe.Node(niu.IdentityInterface(fields=['x', 'y']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['z']), name='outputnode')

    # Generate many nodes and claim a lot of memory
    add_nd = pe.MapNode(
        niu.Function(function=add, input_names=['x', 'y'], output_names=['z']),
        name='add',
        iterfield=['x'],
        mem_gb=0.8,
    )

    # Regular node
    sum_nd = pe.Node(niu.Function(function=addall, input_names=['inlist']), name='sum')

    # Run without submitting is another code path
    add_more_nd = pe.Node(
        niu.Function(function=add, input_names=['x', 'y'], output_names=['z']),
        name='add_more',
        run_without_submitting=True,
    )

    workflow.connect(
        [
            (inputnode, add_nd, [('x', 'x'), ('y', 'y')]),
            (add_nd, sum_nd, [('z', 'inlist')]),
            (sum_nd, add_more_nd, [('out', 'x')]),
            (inputnode, add_more_nd, [('y', 'y')]),
            (add_more_nd, outputnode, [('z', 'z')]),
        ]
    )

    inputnode.inputs.x = list(range(30))
    inputnode.inputs.y = 4

    # Avoid unnecessary sleeps
    workflow.config['execution']['poll_sleep_duration'] = 0

    return workflow


def test_plugin_defaults(workflow, caplog):
    """Test the plugin works without any arguments."""
    caplog.set_level(logging.CRITICAL, logger='nipype.workflow')
    workflow.run(plugin=MultiProcPlugin())


def test_plugin_args_noconfig(workflow, caplog):
    """Test the plugin works with typical nipype arguments."""
    caplog.set_level(logging.CRITICAL, logger='nipype.workflow')
    workflow.run(plugin=MultiProcPlugin(plugin_args={'n_procs': 2, 'memory_gb': 0.1}))


def touch_file(file_path: str) -> None:
    """Module-level functions play more nicely with multiprocessing."""
    with open(file_path, 'w') as f:
        f.write('flag')


def test_plugin_app_config(tmp_path, workflow, caplog):
    """Test the plugin works with a nipreps-style configuration."""

    init_flag = tmp_path / 'init_flag.txt'

    app_config = SimpleNamespace(
        environment=SimpleNamespace(total_memory=1),
        _process_initializer=touch_file,
        file_path=str(init_flag),
    )
    caplog.set_level(logging.INFO, logger='nipype.workflow')
    workflow.run(plugin=MultiProcPlugin(plugin_args={'n_procs': 2, 'app_config': app_config}))

    assert init_flag.exists()
    assert init_flag.read_text() == 'flag'
