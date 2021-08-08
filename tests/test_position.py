# Position tests.

import kamitest

import consts
from position import Position

import numpy as np

# Checks the position input shape is correct while making moves.
def test_position_input_shape():
    pos = Position()

    def check_input():
        headers, frames = pos.get_input()

        assert frames.shape == (consts.FRAME_COUNT, 8, 8, consts.FRAME_SIZE)
        assert headers.shape == (consts.HEADER_SIZE,)

    check_input()
    for mv in ['e2e4', 'e7e5', 'a2a4', 'a7a5']:
        pos.push(mv)
        check_input()

# Checks the position can make and unmake moves.
def test_position_make_unmake_moves():
    pos = Position()

    pos.push('e2e4')
    pos.push('c7c5')
    pos.push('d2d4')
    pos.push('c5d4')
    pos.push('c2c3')

    for _ in range(5):
        pos.pop()

# Checks the position can detect checkmate.
def test_position_checkmate():
    pos = Position()

    assert pos.is_game_over() is None

    pos.push('e2e4')
    pos.push('c7c5')
    pos.push('d1f3')
    pos.push('c5c4')
    pos.push('f1c4')
    pos.push('a7a5')
    pos.push('f3f7')

    assert pos.is_game_over() == 1

# Checks the position can detect a draw by repetition.
def test_position_repetition():
    pos = Position()

    pos.push('g1f3')
    pos.push('g8f6')
    pos.push('f3g1')
    pos.push('f6g8')
    pos.push('g1f3')
    pos.push('g8f6')
    pos.push('f3g1')
    pos.push('f6g8')
    pos.push('g1f3')
    pos.push('g8f6')
    pos.push('f3g1')
    pos.push('f6g8')

    assert pos.is_game_over() == 0

# Checks LMM and moves are correct for each color
def test_position_moves_lmm():
    pos = Position()
    moves, lmm = pos.moves()

    assert 'a2a4' in moves
    assert 'b2b4' in moves
    assert 'c2c4' in moves
    assert 'd2d4' in moves
    assert 'e2e4' in moves
    assert 'f2f4' in moves
    assert 'g2g4' in moves
    assert 'h2h4' in moves
    assert 'a2a3' in moves
    assert 'b2b3' in moves
    assert 'c2c3' in moves
    assert 'd2d3' in moves
    assert 'e2e3' in moves
    assert 'f2f3' in moves
    assert 'g2g3' in moves
    assert 'h2h3' in moves
    assert 'b1a3' in moves
    assert 'b1c3' in moves
    assert 'g1f3' in moves
    assert 'g1h3' in moves

    assert len(moves) == 20

# Checks the first 2 board inputs to ensure they are correct.
def test_position_initial_input():
    pos = Position()

    headers, frames = pos.get_input()

    # First input, white to move

    assert np.allclose(headers, np.array([
        1, 0, 0, 0, 0, 0, 0, 0, # move number (1)
        0, 0, 0, 0, 0, 0,       # halfmove clock (0)
        1, 1, 1, 1,             # castling rights (F)
    ]))

    assert np.allclose(frames[-1, 0, 0], np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) # A1, our rook
    assert np.allclose(frames[-1, 0, 1], np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) # B1, our knight
    assert np.allclose(frames[-1, 0, 2], np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) # C1, our bishop
    assert np.allclose(frames[-1, 0, 3], np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])) # D1, our queen
    assert np.allclose(frames[-1, 0, 4], np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])) # E1, our king
    assert np.allclose(frames[-1, 0, 5], np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) # F1, our bishop
    assert np.allclose(frames[-1, 0, 6], np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) # G1, our knight
    assert np.allclose(frames[-1, 0, 7], np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) # H1, our rook

    for f in range(8):
        assert np.allclose(frames[-1, 1, f], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) # X2, our pawn

    for r in range(2, 6):
        for f in range(8):
            assert np.allclose(frames[-1, r, f], np.zeros(consts.FRAME_SIZE)), (r, f) # central empty squares

    for f in range(8):
        assert np.allclose(frames[-1, 6, f], np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])) # X7, their pawn

    assert np.allclose(frames[-1, 7, 0], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])) # A1, their rook
    assert np.allclose(frames[-1, 7, 1], np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])) # B1, their knight
    assert np.allclose(frames[-1, 7, 2], np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])) # C1, their bishop
    assert np.allclose(frames[-1, 7, 3], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])) # D1, their queen
    assert np.allclose(frames[-1, 7, 4], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])) # E1, their king
    assert np.allclose(frames[-1, 7, 5], np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])) # F1, their bishop
    assert np.allclose(frames[-1, 7, 6], np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])) # G1, their knight
    assert np.allclose(frames[-1, 7, 7], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])) # H1, their rook

    pos.push('e2e4')

    # Second input, black to move

    headers, frames = pos.get_input()

    assert np.allclose(headers, np.array([
        1, 0, 0, 0, 0, 0, 0, 0, # move number (1)
        0, 0, 0, 0, 0, 0,       # halfmove clock (0)
        1, 1, 1, 1,             # castling rights (F)
    ]))

    assert np.allclose(frames[-1, 0, 0], np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) # A1, our rook
    assert np.allclose(frames[-1, 0, 1], np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) # B1, our knight
    assert np.allclose(frames[-1, 0, 2], np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) # C1, our bishop
    assert np.allclose(frames[-1, 0, 3], np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])) # D1, our king
    assert np.allclose(frames[-1, 0, 4], np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])) # E1, our queen
    assert np.allclose(frames[-1, 0, 5], np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) # F1, our bishop
    assert np.allclose(frames[-1, 0, 6], np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) # G1, our knight
    assert np.allclose(frames[-1, 0, 7], np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) # H1, our rook

    for f in range(8):
        assert np.allclose(frames[-1, 1, f], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) # X2, our pawn

    for r in range(2, 6):
        for f in range(8):
            if r == 4 and f == 3:
                assert np.allclose(frames[-1, 4, 3], np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])) # D5, their pawn (E4 from wpov)
            else:
                assert np.allclose(frames[-1, r, f], np.zeros(consts.FRAME_SIZE)), (r, f) # central empty squares

    for f in range(8):
        if f == 3:
            assert np.allclose(frames[-1, 6, f], np.zeros(consts.FRAME_SIZE)), (r, f) # empty square where e2 was
        else:
            assert np.allclose(frames[-1, 6, f], np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])) # X7, their pawn

    assert np.allclose(frames[-1, 7, 0], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])) # A1, their rook
    assert np.allclose(frames[-1, 7, 1], np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])) # B1, their knight
    assert np.allclose(frames[-1, 7, 2], np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])) # C1, their bishop
    assert np.allclose(frames[-1, 7, 3], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])) # D1, their king
    assert np.allclose(frames[-1, 7, 4], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])) # E1, their queen
    assert np.allclose(frames[-1, 7, 5], np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])) # F1, their bishop
    assert np.allclose(frames[-1, 7, 6], np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])) # G1, their knight
    assert np.allclose(frames[-1, 7, 7], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])) # H1, their rook