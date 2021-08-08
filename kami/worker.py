import consts
from model import Model
from mctsbatch import MCTSBatch

import asyncio
import json
import socket
import sys

# Initialize model
model = Model(sys.argv[2])

conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
conn.connect(('127.0.0.1', int(sys.argv[1])))

def send(msg: str):
    conn.send(len(msg).to_bytes(4, 'little', signed=False))
    conn.send(msg.encode('utf-8'))

def nextbytes(n):
    bout = b''
    while len(bout) < n:
        bout += conn.recv(n - len(bout))
    return bout

def recvnext():
    msglen = int.from_bytes(nextbytes(4), 'little')
    return nextbytes(msglen).decode('utf-8')

while True:
    send('READY')

    msg = recvnext()

    if msg == 'STOP':
        break

    batch = json.loads(msg)

    if len(batch['headers']) == 0:
        continue

    # Batch should have fields 'headers', 'frames', 'lmm', 'nodes', 'actions'
    # We will transform them into 'policy', 'value', 'nodes', 'actions'.

    model.execute(batch)
    send(json.dumps(batch))