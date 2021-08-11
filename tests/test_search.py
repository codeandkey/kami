# Tests for the Search structure.

import kamitest

import consts
from model import Model
from search import Search
from tree import Tree

# Tests the search can initialize and run a short search on a fresh tree.
def test_search_basic(tmp_path):
    tmp_path = tmp_path.joinpath('model.pt')

    m = Model()
    m.save(tmp_path)

    s = Search(tmp_path, consts.WORKER_PORT, 3)
    t = Tree()

    s.run(t, consts.SEARCH_NODES, print)