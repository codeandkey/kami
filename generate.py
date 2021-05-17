import numpy as np
import os
import pathlib
import random
import string
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data_dir = None

if len(sys.argv) > 1:
    data_dir = sys.argv[1]
else:
    if os.name == 'posix':
        data_dir = os.path.join(os.getenv('HOME'), '.local', 'share', 'kami')
    elif os.name == 'nt':
        data_dir = os.path.join(os.getenv('APPDATA'), 'kami')
    else:
        raise RuntimeError('Unsupported platform "{}".'.format(os.name))

    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)

print('Using data directory "{}".'.format(data_dir))
print('Generating model.')

# Model version number
VERSION = 1

# Number of previous board states to include in input.
HISTORY_PLY = 5

# Bits per square.
# | bits | item
# |------+-----------------------------------
# |   14 | full move number
# |    6 | halfmove clock
# |    4 | castling weights (WwBb) from pov
# |  14H | > 12 piece values from pov
# |      | > 2 repetition bits
SQUARE_BITS = 14 + 6 + 4 + 14 * (HISTORY_PLY + 1)

# Input layers
input_board = keras.Input(shape=(8, 8, SQUARE_BITS))
input_lmm   = keras.Input(shape=(4096,))

# Using keras functional API.
x = input_board

# First convolution
x = layers.Conv2D(64, (3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

# Residual layers
NUM_RESIDUAL_LAYERS = 16

for _ in range(NUM_RESIDUAL_LAYERS):
    skip = x
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, skip])

# Value head
value = layers.Conv2D(1, (1, 1), padding='same')(x)
value = layers.BatchNormalization()(value)
value = layers.ReLU()(value)
value = layers.Dense(256)(value)
value = layers.ReLU()(value)
value = layers.Dense(256)(value)
value = layers.Activation('tanh')(value)

# Policy head
policy = layers.Conv2D(2, (1, 1))(x)
policy = layers.BatchNormalization()(policy)
policy = layers.ReLU()(policy)
policy = layers.Dense(4096)(policy)
policy = layers.Multiply()([policy, input_lmm])
policy = layers.Activation('softmax')(policy)

# Finalize model
model = keras.Model(inputs=[input_board, input_lmm], outputs=[policy, value], name='kami')

# Add optimizer

def lossfn(y_true, y_pred):
    # Policy loss with metric
    ploss = keras.losses.SparseCategoricalCrossentropy(y_true, y_pred)

    # Value loss
    vloss = keras.losses.MeanSquaredError(y_true, y_pred)

    # Return loss sum.
    return tf.add(ploss, vloss)

model.compile(
    optimizer='adam',
    loss=lossfn,
)

print('Initialized model.')

# Write output
output = os.path.join(data_dir, 'model')
print('Writing model to "%s".' % output)
model.save(output)

# Write version and uid
with open(os.path.join(output, 'version'), 'w') as f:
    f.write('{}\n'.format(VERSION))

uid = ''.join(random.choices(string.ascii_uppercase, k=4))

with open(os.path.join(output, 'id'), 'w') as f:
    f.write(uid)

# All done!
model.summary()

print('Wrote model {}, version {}'.format(uid, VERSION))
