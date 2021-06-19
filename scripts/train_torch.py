# Training script for torch modules

import json
import os
import pathlib
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

LEARNING_RATE=1e-4
FRAME_COUNT=6
FRAME_SIZE=14
HEADER_SIZE=18
EPOCHS = 10

# Load module from path
module = torch.jit.load(sys.argv[1])

# Load training data from path
train_json = None
with open(sys.argv[2]) as f:
    train_json = json.load(f)

# Split training batches
batches = []

for b in train_json:
    bsize = len(b['inner']['lmm']) // 4096

    batches.append((
    (
        torch.FloatTensor(b['inner']['headers']).reshape(bsize, HEADER_SIZE),
        torch.FloatTensor(b['inner']['frames']).reshape(bsize, 8, 8, FRAME_COUNT * FRAME_SIZE),
        torch.FloatTensor(b['inner']['lmm']).reshape(bsize, 4096),
    ),
    (
        torch.FloatTensor(b['mcts']).reshape(bsize, 4096),
        torch.FloatTensor(b['results']).reshape(bsize, 1)
    )
    ))

# Set training mode
module.train()

# Define loss function
def loss(policy, value, mcts, result):
    #print('loss: policy.shape={}, value.shape={}, mcts.shape={}, result.shape={}'.format(policy.shape, value.shape, mcts.shape, result.shape))
    return nn.MSELoss()(value, result) - torch.dot(torch.flatten(mcts), torch.log(torch.flatten(policy)))

# Train model!

optimizer = torch.optim.SGD(module.parameters(), lr = LEARNING_RATE)
first_avg_loss = None
last_avg_loss = None

def train_loop():
    for b, ((headers, frames, lmm), (mcts, result)) in enumerate(batches):
        policy, value = module(headers, frames, lmm)
        current_loss = loss(policy, value, mcts, result)

        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()

        print('loss: {:>7f}'.format(current_loss.item()))

def test_loop():
    global first_avg_loss
    global last_avg_loss

    test_loss, correct = 0, 0

    with torch.no_grad():
        for b, ((headers, frames, lmm), (mcts, result)) in enumerate(batches):
            policy, value = module(headers, frames, lmm)
            test_loss += loss(policy, value, mcts, result)
        
    test_loss /= len(batches)

    print('Current average loss: {:>8f}'.format(test_loss))

    if first_avg_loss is None:
        first_avg_loss = test_loss

    last_avg_loss = test_loss


print('Starting model training.')

for i in range(EPOCHS):
    print('--------------------\nEpoch {}/{}'.format(i, EPOCHS))
    train_loop()
    test_loop()
    print('--------------------\n')

print('Finished model training!')
print('Final average loss: {} ---> {}'.format(first_avg_loss, last_avg_loss))

# Write trained model back to source.
module.save(sys.argv[1])
print('Saved model to {}'.format(sys.argv[1]))