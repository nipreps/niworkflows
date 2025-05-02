"""Workflow splicing operations."""

import logging
import typing as ty
from functools import wraps

import nipype.pipeline.engine as pe
from nipype.pipeline.engine.base import EngineBase


def tag(tag: str) -> ty.Callable:
    """
    Decorator to set a tag on an `init_...wf` function.

    This is used to mark nodes or workflows for replacement in the splicing process.
    """

    def _decorator(func) -> ty.Callable:
        @wraps(func)
        def _tag(*args, **kwargs) -> EngineBase:
            node = func(*args, **kwargs)
            node._tag = tag
            return node

        return _tag

    return _decorator


def splice_workflow(
    root_wf: pe.Workflow,
    replacements: dict[str, EngineBase],
    *,
    write_graph: bool = False,
    debug: bool = False,
):
    """
    Splice a workflow's tagged nodes / workflows and replace connections with alternatives.

    Requires that the workflow has been tagged with a `_tag` attribute.
    """
    if write_graph:
        root_wf.write_graph('pre-slice.dot', format='png', graph2use='colored')

    substitutions = _get_substitutions(root_wf, replacements)
    _splice_components(root_wf, substitutions, debug=debug)

    if write_graph:
        root_wf.write_graph('post-slice.dot', format='png', graph2use='colored')
    return root_wf


def _get_substitutions(
    workflow: pe.Workflow,
    replacements: dict[str, EngineBase],
) -> dict[EngineBase, EngineBase]:
    """ "Query tags in workflow, and return a list of substitutions to make"""
    substitutions = {}
    tagged_wfs = _fetch_tags(workflow)
    for tag in tagged_wfs:
        if tag in replacements:
            substitutions[tagged_wfs[tag]] = replacements[tag]
    return substitutions


def _fetch_tags(wf: pe.Workflow) -> dict[str, EngineBase]:
    """Query all nodes in a workflow and return a dictionary of tags and nodes."""
    tagged = {}
    for node in wf._graph.nodes:
        if hasattr(node, '_tag'):
            tagged[node._tag] = node
        if isinstance(node, pe.Workflow):
            inner_tags = _fetch_tags(node)
            tagged.update(inner_tags)
    return tagged


def _splice_components(
    workflow: pe.Workflow,
    substitutions: dict[EngineBase, EngineBase],
    debug: bool = False,
) -> tuple[list, list]:
    """Query all connections and return a list of removals and additions to be made."""
    edge_removals = []
    edge_connects = []
    node_removals = set()
    node_adds = set()
    _expanded_workflows = set()

    to_replace = [x.fullname for x in substitutions]

    for src, dst in workflow._graph.edges:
        if dst.fullname in to_replace:
            edge_data = workflow._graph.get_edge_data(src, dst)
            alt_dst = substitutions[dst]
            alt_dst._hierarchy = dst._hierarchy

            edge_removals.append((src, dst))
            node_removals.add(dst)
            node_adds.add(alt_dst)
            edge_connects.append((src, alt_dst, edge_data))
        elif src.fullname in to_replace:
            edge_data = workflow._graph.get_edge_data(src, dst)
            alt_src = substitutions[src]
            alt_src._hierarchy = src._hierarchy

            edge_removals.append((src, dst))
            node_removals.add(src)
            node_adds.add(alt_src)
            edge_connects.append((alt_src, dst, edge_data))
        elif isinstance(dst, pe.Workflow) and dst not in _expanded_workflows:
            _expanded_workflows.add(dst)
            _splice_components(dst, substitutions, debug=debug)
        elif isinstance(src, pe.Workflow) and src not in _expanded_workflows:
            _expanded_workflows.add(src)
            _splice_components(src, substitutions, debug=debug)

    logger = logging.getLogger('nipype.workflow')
    logger.debug(
        'Workflow: %s, \n- edge_removals: %s, \n+ edge_connects: %s',
        workflow,
        edge_removals,
        edge_connects,
    )

    workflow._graph.remove_edges_from(edge_removals)
    workflow.remove_nodes(node_removals)
    workflow.add_nodes(node_adds)
    workflow._graph.add_edges_from(edge_connects)
