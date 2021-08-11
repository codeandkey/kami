# Benchmark for torch threading

from mctsbatch import MCTSBatch
from model import Model
from tree import Tree

import threading
import time
import queue

import torch

m = Model()

def do_bench(nthreads):
    total_nodes = 0
    tree = Tree()
    starttime = time.time()

    torch.set_num_threads(nthreads)

    while total_nodes <= 2500:
        nb = tree.next_batch().into_dict()
        m.execute(nb)
        tree.expand(nb)

        if total_nodes == 0:
            starttime = time.time()

        total_nodes += len(nb['value'])

    print('Bench result for {} threads: {} nodes/second'.format(nthreads, total_nodes / (time.time() - starttime)))

for nt in range(1, 7):
    do_bench(nt)