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
    tmp_path = tmp_path.joinpath('model.onnx')
    model = Model()
    model.save(tmp_path)

# Tests a model can be loaded from a directory.
def test_model_load(tmp_path:str):
    tmp_path = tmp_path.joinpath('model.onnx')
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

# Tests a model can execute a single batch.
def test_model_execute():
    model = Model()

    def random_lmm():
        return list(map(lambda _: float(random.randint(0, 1)), range(4096)))

    batch = (
        np.random.randn(16, consts.HEADER_SIZE),
        np.random.randn(16, 8, 8, consts.FRAME_SIZE * consts.FRAME_COUNT),
        [random_lmm()] * 16,
    )

    policy, value = model.execute(batch)

    # Check output shapes
    assert len(policy) == 16
    assert len(policy[0]) == 4096
    assert len(value) == 16
    assert len(value[0]) == 1

    # Check value in bounds
    assert value[0][0] >= -1.0
    assert value[0][0] <= 1.0
    
    # Check policy sums to 1
    assert abs(sum(policy[0]) - 1.0) < 1e-6