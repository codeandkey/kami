# MCTS-specialized input batch type.

from batch import Batch, BatchResult

class MCTSResult(BatchResult):
    def __init__(self, policy, value, actions, nodes, depths):
        super().__init__(policy, value)

        self.actions = actions
        self.nodes   = nodes
        self.depths  = depths

    def get_actions(self, ind):
        """Returns actions for a batch index."""
        return self.actions[ind]

    def get_node(self, ind):
        """Returns target nodeid for a batch index."""
        return self.nodes[ind]

    def get_depth(self, ind):
        """Returns node depth for a batch index."""
        return self.depths[ind]

class MCTSBatch(Batch):
    """Manages an MCTS input batch. This is a specialized input batch tracking
       child actions and target nodes for expansion."""

    def __init__(self, maxsize):
        """Initializes a new empty batch with maximum size `maxsize`."""
        super().__init__(maxsize)

        self.actions = []
        self.nodes   = []
        self.depths  = []

    def add(self, headers, frames, lmm, actions, nodeid, depth):
        """Adds an input to the batch."""
        super().add(headers, frames, lmm)

        self.nodes.append(nodeid)
        self.actions.append(actions)
        self.depths.append(depth)

    def make_result(self, policy, value):
        """Creates a result structure from a network output."""
        return MCTSResult(policy, value, self.actions, self.nodes, self.depths)