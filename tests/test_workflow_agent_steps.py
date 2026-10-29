"""Increasing Coder's default must not change unrelated workflow budgets."""

import pytest

from bioagents.workflows.nodes.agent_nodes import CoderAgentNode, DlAgentNode, MlAgentNode


@pytest.mark.parametrize(
    "node,expected", [(CoderAgentNode, 50), (MlAgentNode, 20), (DlAgentNode, 20)]
)
def test_agent_specific_default_steps(node, expected):
    assert node().params["max_steps"] == expected


@pytest.mark.parametrize("node", [CoderAgentNode, MlAgentNode, DlAgentNode])
def test_explicit_budget_is_preserved(node):
    assert node(max_steps=7).params["max_steps"] == 7
    with pytest.raises(ValueError):
        node(max_steps=0)
