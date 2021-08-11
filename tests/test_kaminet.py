# Tests for Kami network architecture

import kamitest

import consts
from model import Model, transform_input

import numpy as np

# Tests the network board input is constructed properly.
def test_kaminet_transform_input():
    model = Model()

    input_header_data = np.random.randn(1, consts.HEADER_SIZE)
    input_frames_data = np.random.randn(1, consts.FRAME_COUNT, 8, 8, consts.FRAME_SIZE)

    input_headers = model.to_tensor(input_header_data)
    input_frames = model.to_tensor(input_frames_data)

    board_input = transform_input(input_headers, input_frames, True)
    
    assert board_input.shape == (1, consts.HEADER_SIZE + consts.FRAME_COUNT * consts.FRAME_SIZE, 8, 8)

    input_frames_data = input_frames_data.swapaxes(1, 3)
    input_frames_data = input_frames_data.reshape((1, 8, 8, consts.FRAME_COUNT * consts.FRAME_SIZE))

    for r in range(8):
        for f in range(8):
            assert np.allclose(board_input[0, :, r, f], np.concatenate((input_header_data[0, :], input_frames_data[0, r, f, :])))