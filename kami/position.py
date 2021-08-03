# Position type.

import consts

import chess
import numpy as np

from typing import Optional

class Position():
    """Manages a chess position and its input layer."""

    def __init__(self):
        """Initializes a Position with the starting position."""

        self.bframes = np.zeros((consts.FRAME_COUNT - 1, 8, 8, consts.FRAME_SIZE))
        self.wframes = np.zeros((consts.FRAME_COUNT - 1, 8, 8, consts.FRAME_SIZE))
        self.headers = np.zeros(consts.HEADER_SIZE)

        self.frame_count = consts.FRAME_COUNT - 1

        self.board = chess.Board()
        self.key_stack = []
        self.key_stack.append(self.board._transposition_key())

        self.write_headers()
        self.push_frame()

    def moves(self) -> (list[str], np.ndarray):
        """Gets a list of legal actions along with a LMM from the point of view
           of the color to move."""

        moves_out = []
        lmm = np.zeros(4096)

        for mv in self.board.generate_legal_moves():
            moves_out.append(str(mv))

            src = mv.from_square
            dst = mv.to_square

            if self.board.turn == chess.BLACK:
                src = 63 - src
                dst = 63 - dst

            lmm[src * 64 + dst] = 1.0
        
        return moves_out, lmm

    def get_input(self) -> (np.ndarray, np.ndarray):
        """Gets the board input frames."""
        if self.board.turn == chess.WHITE:
            return self.headers, self.wframes[self.frame_count-consts.FRAME_COUNT:self.frame_count]
        else:
            return self.headers, self.bframes[self.frame_count-consts.FRAME_COUNT:self.frame_count]

    def write_headers(self):
        """Updates the input header field."""
        self.headers.fill(0.0)
        
        # Write move number
        move_number = self.board.fullmove_number
        
        for i in range(8):
            if (move_number >> i) & 1:
                self.headers[i] = 1.0
        
        # Write halfmove clock
        halfmove_clock = self.board.halfmove_clock

        for i in range(6):
            if (halfmove_clock >> i) & 1:
                self.headers[i + 8] = 1.0

        # Write castling rights
        us = self.board.turn
        them = not us # huh

        if self.board.has_kingside_castling_rights(us):
            self.headers[14] = 1.0
        
        if self.board.has_queenside_castling_rights(us):
            self.headers[15] = 1.0

        if self.board.has_kingside_castling_rights(them):
            self.headers[16] = 1.0
        
        if self.board.has_queenside_castling_rights(them):
            self.headers[17] = 1.0

    def push_frame(self):
        """Pushes a new board frame onto the stack."""
        self.frame_count += 1

        # Expand buffer to hold new frames if necessary
        if self.wframes.shape[0] <= self.frame_count:
            
            self.wframes = np.concatenate((self.wframes, np.zeros((1, 8, 8, consts.FRAME_SIZE))))
            self.bframes = np.concatenate((self.bframes, np.zeros((1, 8, 8, consts.FRAME_SIZE))))

        # Compute repetition bits
        reps = self.key_stack.count(self.key_stack[-1]) - 1

        rbitlow = reps & 1
        rbithigh = reps >> 1

        for r in range(8):
            for f in range(8):
                pc = self.board.piece_at(chess.square(f, r))

                # Write repetition bits
                # No need to pov flip here, all squares are written anyway
                self.wframes[self.frame_count - 1, r, f, 12] = rbitlow
                self.wframes[self.frame_count - 1, r, f, 13] = rbithigh
                self.bframes[self.frame_count - 1, r, f, 12] = rbitlow
                self.bframes[self.frame_count - 1, r, f, 13] = rbithigh

                # Write piece bits, if there is a piece
                if pc is not None:
                    ptype = pc.piece_type - 1
                    
                    if pc.color == chess.WHITE:
                        wpbit = ptype
                        bpbit = 6 + ptype
                    else:
                        wpbit = 6 + ptype
                        bpbit = ptype

                    self.wframes[self.frame_count - 1, r, f, wpbit] = 1.0
                    self.bframes[self.frame_count - 1, 7 - r, 7 - f, bpbit] = 1.0

    def pop_frame(self):
        """Pops an input frame off the stack."""
        self.frame_count -= 1
    
    def push(self, action):
        """Makes a move on the board and updates the input layers."""

        self.board.push(chess.Move.from_uci(action))
        self.key_stack.append(self.board._transposition_key())
        
        # Update headers
        self.write_headers()
        self.push_frame()
    
    def pop(self):
        """Unmakes the last move."""

        self.board.pop()
        self.key_stack.pop()
        self.pop_frame()
        self.write_headers()

    def is_game_over(self) -> Optional[float]:
        """Returns the terminal value of this node from the turn's POV,
           or None if the game is ongoing."""
        res = self.board.outcome(claim_draw=True)

        if res is None:
            return None

        if res.result() == '1/2-1/2':
            return 0

        # Any non-draw ending is a loss for the CTM (checkmate)
        return -1