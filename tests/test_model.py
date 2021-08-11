# Tests for the Model type.

import kamitest

import consts
from model import Model

import numpy as np
import random

# Tests a model can generate without throwing execptions.
def test_model_generate():
    Model()

# Tests a model can be saved to a directory.
def test_model_save(tmp_path:str):
    tmp_path = tmp_path.joinpath('model.pt')
    model = Model()
    model.save(tmp_path)

# Tests a model can be loaded from a directory.
def test_model_load(tmp_path:str):
    tmp_path = tmp_path.joinpath('model.pt')
    model = Model()
    model.save(tmp_path)
    model = Model(tmp_path)

# Tests a model can be trained on a single batch.
def test_model_train():
    model = Model()

    def random_lmm():
        return list(map(lambda _: float(random.randint(0, 1)), range(4096)))

    batches = [(
        (
            np.random.randn(16, consts.HEADER_SIZE),
            np.random.randn(16, 8, 8, consts.FRAME_SIZE * consts.FRAME_COUNT),
            [random_lmm()] * 16,
        ),
        (
            [[1 / 4096.0] * 4096] * 16,
            np.random.randn(16, 1),
        )
    )]

    lfirst, llast = model.train(batches)

    assert type(lfirst) == float
    assert type(llast) == float