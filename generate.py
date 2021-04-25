import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print('Generating model.')

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
SQUARE_BITS = 14 + 6 + 4 + 12 * (HISTORY_PLY + 1)

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
output = 'model' if len(sys.argv) < 2 else sys.argv[1]
print('Writing model to "%s".' % output)
model.save(output)

# All done!
model.summary()
print('Ready.')