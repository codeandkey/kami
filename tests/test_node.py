# Tests for the MCTS Node type.

import kamitest

from node import Node

# Tests a root node has no children or n.
def test_root_node_sane():
    n = Node()

    assert n.n == 0
    assert len(n.children) == 0

# Tests node backprop behavior.
def test_node_backprop():
    n = Node()

    assert n.n == 0
    assert n.q() == 0

    n.backprop(1)
    
    assert n.n == 1
    assert n.q() == 1

# Tests node PV generation.
def test_node_pv():
    n = Node()
    n.expand(['abcd'], [0.0], 0.0, 0)
    n.children[0].expand(['efgh'], [0.0], 0.0, 0)
    n.children[0].children[0].expand(['ijkl'], [0.0], 0.0, 0)

    assert n.pv() == ['abcd', 'efgh', 'ijkl']