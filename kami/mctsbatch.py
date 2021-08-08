# MCTS-specialized input batch type.

from batch import Batch

class MCTSBatch(Batch):
    """Manages an MCTS input batch. This is a specialized input batch tracking
       child actions and target nodes for expansion."""

    def __init__(self, maxsize):
        """Initializes a new empty batch with maximum size `maxsize`."""
        super().__init__(maxsize)

        self.actions = []
        self.nodes   = []

    def add(self, headers, frames, lmm, actions, nodeid):
        """Adds an input to the batch."""
        super().add(headers, frames, lmm)

        self.nodes.append(nodeid)
        self.actions.append(actions)

    def into_dict(self):
        ret = super().into_dict()
        ret['nodes'] = self.nodes
        ret['actions'] = self.actions
        return ret

    def make_result(self, policy, value):
        ret = super().make_result(policy, value)
        ret['nodes'] = self.nodes
        ret['actions'] = self.actions
        return ret