# Input batch type.

import consts

import numpy as np

class BatchResult:
    def __init__(self, policy, value):
        self.policy = policy
        self.value  = value

    def get_size(self):
        return len(self.value)
    
    def get_value(self, ind):
        return self.value[ind]
    
    def get_policy(self, ind):
        return self.policy[ind]

class Batch:
    """Manages an input batch. This is the most basic input type to the network,
       containing only headers, frames, and LMM. No other information is tracked."""

    def __init__(self, maxsize):
        """Initializes a new empty batch with maximum size `maxsize`."""
        self.headers = np.empty((maxsize, consts.HEADER_SIZE))
        self.frames  = np.empty((maxsize, consts.FRAME_COUNT, 8, 8, consts.FRAME_SIZE))
        self.lmm     = np.empty((maxsize, 4096))
        self.size    = 0

    def add(self, headers, frames: np.ndarray, lmm):
        """Adds an input to the batch."""

        np.copyto(self.headers[self.size], headers)
        np.copyto(self.frames[self.size], frames)
        np.copyto(self.lmm[self.size], lmm)

        self.size += 1

    def get_size(self):
        """Returns the number of positions in the batch."""
        return self.size

    def get_headers(self):
        """Returns the available headers."""
        return self.headers[:self.size]

    def get_frames(self):
        """Returns the available frames."""
        return self.frames[:self.size]

    def get_lmm(self):
        """Returns the available legal move masks."""
        return self.lmm[:self.size]

    def get_inputs(self):
        """Returns a tuple of all the available inputs `headers`, `frames`, and `lmm`."""
        return self.get_headers(), self.get_frames(), self.get_lmm()
    
    def make_result(self, policy, value):
        return BatchResult(policy, value)