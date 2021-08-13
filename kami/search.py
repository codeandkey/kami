# Search structure

import consts

import chess
import json
import multiprocessing as mp
import numpy as np
import os
import socket
import subprocess as sp
import time

class Search:
    def __init__(self, modelpath: str, port=consts.WORKER_PORT, nprocs: int = mp.cpu_count() // 2):
        # Build searcher service, synchronously
        self.sproc = sp.Popen([
            'cargo',
            'build',
            '--release',
            '--quiet'
        ], cwd=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'search'))

        if self.sproc.wait() != 0:
            raise RuntimeError('Search build failed!')

        # Start searcher service

        self.sproc = sp.Popen([
            'cargo',
            'run',
            '--quiet',
            '--release',
            '--',
            str(port),
        ], cwd=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'search'))

        # Connect to service

        ok = False
        for _ in range(consts.MAX_RETRIES):
            try:
                self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
                self.conn.connect(('127.0.0.1', port))
                ok = True
                break
            except ConnectionRefusedError:
                time.sleep(consts.RETRY_DELAY)

        if not ok:
            raise RuntimeError("Couldn't establish connection with search service after {} retries.".format(consts.MAX_RETRIES))

        self.connfile = self.conn.makefile('rw')

        # Write configuration

        config = {
            'Config': {
                'puct_policy_weight': consts.PUCT_POLICY_WEIGHT,
                'puct_noise_weight': consts.PUCT_NOISE_WEIGHT,
                'puct_noise_alpha': consts.PUCT_NOISE_ALPHA,
                'batch_size': consts.BATCH_SIZE,
                'num_threads': nprocs,
                'model_path': modelpath,
                'search_nodes': consts.SEARCH_NODES,
                'temperature': consts.TREE_TEMPERATURE,
            }
        }

        self.write(json.dumps(config))

    def write(self, msg):
        self.connfile.write(msg + '\n')
        self.connfile.flush()

    def readline(self):
        return self.connfile.readline()

    def reset(self):
        self.write('{"Load":[]}')

    def stop(self):
        """Stops the search service and waits for the process to join."""
        self.write('"Stop"')
        self.sproc.wait()

    def go(self, status_data):
        """Starts the search and waits for status data. Returns the final
           status response when the search is complete."""

        self.write('"Go"')
        
        while True:
            resp = json.loads(self.readline())
            
            if 'Searching' in resp:
                status_data(resp['Searching'])

            if 'Outcome' in resp:
                return resp['Outcome']
            
            if 'Done' in resp:
                return resp['Done']

    def push(self, action):
        """Advances the tree by a single move."""
        self.write('{"Push":"%s"}' % action)

    def make_batch(self, positions):
        """Generates a training batch given a list of game positions.
           Input should be a list of triplets `(actions, mcts, result)` where
           `actions` is a list of move strings, `mcts` is a list of mcts values
           for each corresponding action, and `result` is the game result from
           white's POV."""

        batch = (
            (
                np.zeros((len(positions), consts.HEADER_SIZE)),
                np.zeros((len(positions), consts.FRAME_COUNT, 8, 8, consts.FRAME_SIZE)),
                np.zeros((len(positions), 4096)),
            ),
            (
                np.zeros((len(positions), 4096)),
                np.zeros((len(positions), 1))
            ),
        )

        for i, (actions, mcts_pairs, result) in enumerate(positions):
            self.write(json.dumps({ "Input": actions }))
            resp = json.loads(self.readline())

            flip = (len(actions) % 2) == 1

            mcts = [0.0] * 4096

            if flip:
                result *= -1
                
                for pair in mcts_pairs:
                    value = pair[0]
                    mv = pair[1]
                    m = chess.Move.from_uci(mv)
                    mcts[(63 - m.from_square) * 64 + 63 - m.to_square] = value
            else:
                for pair in mcts_pairs:
                    value = pair[0]
                    mv = pair[1]
                    m = chess.Move.from_uci(mv)
                    mcts[m.from_square * 64 + m.to_square] = value

            resp = resp['Input']

            batch[0][0][i] = resp['headers']
            batch[0][1][i] = np.array(resp['frames']).reshape((consts.FRAME_COUNT, 8, 8, consts.FRAME_SIZE))
            batch[0][2][i] = resp['lmm']
            batch[1][0][i] = mcts
            batch[1][1][i] = result

        return batch