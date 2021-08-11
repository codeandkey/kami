# Tests for the input Batch type.

import kamitest

from batch import Batch
import consts

import numpy as np
import random

# Checks the Batch can be initialized with various maxsize.
def test_batch_can_init_with_various_maxsize():
    for i in range(1, 32):
        Batch(i)

# Checks the batch input is correct.
def test_batch_input_is_correct():
    b = Batch(2)

    def random_lmm():
        return list(map(lambda _: float(random.randint(0, 1)), range(4096)))

    def random_input():
        return (
            np.random.randn(consts.HEADER_SIZE),
            np.random.randn(consts.FRAME_COUNT, 8, 8, consts.FRAME_SIZE),
            random_lmm()
        )

    (h1, f1, l1) = random_input()
    (h2, f2, l2) = random_input()

    b.add(h1, f1, l1)
    b.add(h2, f2, l2)

    headers, frames, lmm = b.get_inputs()

    h1 = np.expand_dims(h1, axis=0)
    f1 = np.expand_dims(f1, axis=0)
    l1 = np.expand_dims(l1, axis=0)
    h2 = np.expand_dims(h2, axis=0)
    f2 = np.expand_dims(f2, axis=0)
    l2 = np.expand_dims(l2, axis=0)

    assert np.allclose(headers, np.concatenate((h1, h2)))
    assert np.allclose(frames, np.concatenate((f1, f2)))
    assert np.allclose(lmm, np.concatenate((l1, l2)))