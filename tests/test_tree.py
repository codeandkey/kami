# Tests for Tree type.

import kamitest

from tree import Tree
from position import Position

# Tests the tree can construct.
def test_tree_init():
    Tree()

# Tests the tree can build a batch.
def test_tree_next_batch():
    t = Tree()
    b = t.next_batch()

    assert b.get_size() == 1

# Tests the tree can expand a node.
def test_tree_can_expand():
    t = Tree()
    b = t.next_batch()

    res = b.make_result([[1.0 / 4096] * 4096], [0.0])

    t.expand(res)

# Tests the tree can select a specific action.
def test_tree_can_choose():
    t = Tree()
    b = t.next_batch()

    res = b.make_result([[1.0 / 4096] * 4096], [0.0])

    t.expand(res)
    selected = t.choose()

    assert selected.n == 0

# Tests the tree can advance.
def test_tree_can_advance():
    t = Tree()
    b = t.next_batch()

    res = b.make_result([[1.0 / 4096] * 4096], [0.0])

    t.expand(res)
    selected = t.choose()
    t.push(selected.action)

# Tests the tree will select a mate in 1.
def test_tree_selects_mate_in_one():
    pos = Position()

    pos.push('e2e4')
    pos.push('e7e5')
    pos.push('f1c4')
    pos.push('a7a5')
    pos.push('d1f3')
    pos.push('a5a4')

    t = Tree(pos)

    for _ in range(256):
        b = t.next_batch(4)
        res = b.make_result([[1.0 / 4096] * 4096] * b.get_size(), [0.0] * b.get_size())
        t.expand(res)

    for n in t.root.children:
        print(n.action, n.n, n.w, n.p)

    selected = t.choose(0.01)
    
    assert selected.action == 'f3f7'