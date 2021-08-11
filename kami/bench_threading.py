# Benchmark for python threading-based parallel execution
# The GIL makes threading suck for CPU-bound tasks but it still might provide a decent boost
# for CUDA inference.

from mctsbatch import MCTSBatch
from model import Model
from tree import Tree

import threading
import time
import queue

m = Model()

req_pipe = queue.Queue()
worker_batch_pipe = queue.Queue()

def worker_thread():
    global m

    while True:
        req_pipe.put('batch pls')
        nb = worker_batch_pipe.get()

        if nb is None:
            break

        if nb.get_size() > 0:
            result = nb.into_dict()
            m.execute(result)
            req_pipe.put(result)

            
def do_bench(nthreads):
    threads = [threading.Thread(target=worker_thread) for _ in range(nthreads)]
    total_nodes = 0
    tree = Tree()
    starttime = time.time()

    for t in threads:
        t.start()

    # Wait for requests and send batches
    while True:
        reqnext = req_pipe.get()

        if reqnext == 'batch pls':
            worker_batch_pipe.put(tree.next_batch())
        else:
            tree.expand(reqnext)

            if total_nodes == 0:
                starttime = time.time()

            total_nodes += len(reqnext['value'])

            if total_nodes >= 2500:
                break

    for _ in range(nthreads):
        worker_batch_pipe.put(None)

    for t in threads:
        t.join()

    print('Bench result for {} threads: {} nodes/second'.format(nthreads, total_nodes / (time.time() - starttime)))

for nt in range(1, 6):
    req_pipe = queue.Queue()
    worker_batch_pipe = queue.Queue()

    do_bench(nt)