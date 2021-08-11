# Benchmark for torch multiprocessing

import torch
import torch.multiprocessing as mp

from mctsbatch import MCTSBatch
from model import Model
from tree import Tree

import threading
import time
import queue

m = Model()
m.model.share_memory()

def child(i, mdl, tx_queue, rx_queue):
    print(i, 'starting')

    while True:
        rx_queue.put('batch pls')
        nb = tx_queue.get()

        if nb is None:
            break

        if nb.get_size() == 0:
            continue

        nb = nb.into_dict()

        mdl.execute(nb)
        rx_queue.put(nb)
    
    print(i, 'done')

def do_bench(nprocs):
    total_nodes = 0
    tree = Tree()
    starttime = time.time()
    
    tx_queue = mp.Queue()
    rx_queue = mp.Queue()
    procs = []

    for i in range(nprocs):
        newproc = mp.Process(target=child, args=(i, m, tx_queue, rx_queue))
        newproc.start()
        procs.append(newproc)

    while total_nodes <= 2500:
        nreq = rx_queue.get()

        if nreq == 'batch pls':
            tx_queue.put(tree.next_batch())
        else:
            tree.expand(nreq)

            if total_nodes == 0:
                starttime = time.time()

            total_nodes += len(nreq['value'])
    
    for _ in range(nprocs):
        tx_queue.put(None)

    for i, p in enumerate(procs):
        print('joining ', i)
        p.join()

    print('Bench result for {} procs: {} nodes/second'.format(nprocs, total_nodes / (time.time() - starttime)))

if __name__ == '__main__':
    mp.set_start_method('forkserver')

    for nt in range(1, 7):
        do_bench(nt)