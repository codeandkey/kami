# Tests for Kami network architecture

import kamitest

import consts
from model import Model, KamiNet

import numpy as np

# Tests the network board input is constructed properly.
def test_kaminet_board_input():
    model = Model()

    input_header_data = np.random.randn(1, consts.HEADER_SIZE)
    input_frames_data = np.random.randn(1, 8, 8, consts.FRAME_COUNT * consts.FRAME_SIZE)

    input_headers = model.to_tensor(input_header_data)
    input_frames = model.to_tensor(input_frames_data)

    board_input = KamiNet.transform_input(input_headers, input_frames, True)
    
    assert str(board_input.shape) == '(1, %s, 8, 8)' % str(consts.HEADER_SIZE + consts.FRAME_COUNT * consts.FRAME_SIZE)

    for r in range(8):
        for f in range(8):
            assert np.allclose(board_input[0, :, r, f], np.concatenate((input_header_data[0, :], input_frames_data[0, r, f, :])))