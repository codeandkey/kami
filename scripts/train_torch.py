# Training script for torch modules

import json
import os
import pathlib
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

LEARNING_RATE=3e-4
FRAME_COUNT=6
FRAME_SIZE=14
HEADER_SIZE=18
EPOCHS = 10
POLICY_EPSILON=1e-6
L2_REG_WEIGHT = 0.01
DROPOUT = 0.3

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

def value_loss(value, result):
    return nn.MSELoss()(value, result)

def policy_loss(policy, mcts):
    return -torch.sum(mcts * torch.log(torch.add(policy, POLICY_EPSILON)))

# Define loss function
def loss(policy, value, mcts, result):
    return value_loss(value, result) + policy_loss(policy, mcts)

# Train model!

optimizer = torch.optim.RMSprop(module.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG_WEIGHT)
first_avg_p_loss = None
last_avg_p_loss = None
first_avg_v_loss = None
last_avg_v_loss = None

torch.set_printoptions(edgeitems=4096)

def train_loop():
    for b, ((headers, frames, lmm), (mcts, result)) in enumerate(batches):
        headers = torch.nn.Dropout(DROPOUT)(headers)
        frames = torch.nn.Dropout(DROPOUT)(frames)

        # don't apply dropout to LMM

        policy, value = module(headers, frames, lmm)

        p_loss = policy_loss(policy, mcts)
        v_loss = value_loss(value, result)

        actual_loss = p_loss + v_loss

        #print("BEGIN")
        #print("policy: {}", policy)
        #print("lmm: {}", lmm)
        #print("policy * lmm: {}", policy * lmm)
        #print("policy * lmm * mcts", policy * lmm * mcts)
        #print("END")

        optimizer.zero_grad()
        actual_loss.backward()
        optimizer.step()

        print('loss: {:>7f} P {:>7f} V {:>7f}'.format(actual_loss.item(), p_loss.item(), v_loss.item()))

def test_loop():
    global first_avg_p_loss
    global last_avg_p_loss
    global first_avg_v_loss
    global last_avg_v_loss

    test_p_loss, test_v_loss, correct = 0, 0, 0

    with torch.no_grad():
        for b, ((headers, frames, lmm), (mcts, result)) in enumerate(batches):
            policy, value = module(headers, frames, lmm)
            test_p_loss += policy_loss(policy, mcts)
            test_v_loss += value_loss(value, result)
        
    test_p_loss /= len(batches)
    test_v_loss /= len(batches)

    print('Current average loss: P{:>8f} V{:>8f}'.format(test_p_loss, test_v_loss))

    if first_avg_p_loss is None:
        first_avg_p_loss = test_p_loss

    if first_avg_v_loss is None:
        first_avg_v_loss = test_v_loss

    last_avg_p_loss = test_p_loss
    last_avg_v_loss = test_v_loss

print('Starting model training.')

for i in range(EPOCHS):
    print('--------------------\nEpoch {}/{}'.format(i, EPOCHS))
    train_loop()
    test_loop()
    print('--------------------\n')

print('Finished model training!')
print('Final average loss: P {} ---> {} V {} ---> {} T {} ---> {}'.format(
    first_avg_p_loss, last_avg_p_loss,
    first_avg_v_loss, last_avg_v_loss,
    first_avg_p_loss + first_avg_v_loss, last_avg_p_loss + last_avg_v_loss
))

# Write initial and ending average loss
with open(sys.argv[3], 'w') as f:
    f.write('{} {} {} {}\n'.format(first_avg_p_loss, last_avg_p_loss, first_avg_v_loss, last_avg_v_loss))

# Write trained model back to source.
module.save(sys.argv[1])
print('Saved model to {}'.format(sys.argv[1]))