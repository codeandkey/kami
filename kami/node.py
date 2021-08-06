# MCTS node type.

import consts
import numpy as np

class Node:
    """Manages a single MCTS tree node."""

    def __init__(self, turn=None, parent=None, action=None, p=0.0):
        """Initializes a new node with no children."""

        self.n        = 0
        self.tn       = 0
        self.p        = p
        self.w        = 0.0
        self.terminal = None
        self.parent   = parent
        self.children = []
        self.claimed  = False
        self.expanded = False
        self.action   = action
        self.turn     = turn
        self.maxdepth = 0

    def puct(self, noise):
        out = -self.q()

        # policy component
        out += consts.PUCT_POLICY_WEIGHT * (self.p * (1 - consts.PUCT_NOISE_WEIGHT) + noise * consts.PUCT_NOISE_WEIGHT)

        # exploration component
        out += np.sqrt(self.parent.n) / (self.n + 1)
        
        return out
    
    def expand(self, actions, p, v, depth):
        """Expands a node."""
        def next_child(pair):
            (action, p) = pair
            return Node(not self.turn, self, action, p)

        self.claimed = False
        self.children = list(map(next_child, zip(actions, p)))
        self.backprop(v, 0, depth)

    def backprop(self, v, terminal=0, depth=0):
        """Backpropagates a value through connected parent nodes."""
        self.n  += 1
        self.tn += terminal
        self.w  += v
        self.maxdepth = max(depth, self.maxdepth)

        if self.parent:
            self.parent.backprop(-v, terminal, depth - 1)

    def q(self):
        """Gets the average value at this node."""
        if self.n == 0:
            return 0.0

        return self.w / self.n

    def action_path(self, root):
        """Gets the actions to traverse from a root to this node."""
        if self == root:
            return []

        parpath = self.parent.action_path(root)
        parpath.append(self.action)

        return parpath

    def select(self):
        """Gets an MCTS selection at this node."""

        if self.claimed:
            return None

        if self.terminal:
            return self

        if len(self.children) == 0:
            return self
        
        # Compute noise
        noise = np.random.dirichlet([consts.PUCT_NOISE_ALPHA] * len(self.children))

        def puct(pair):
            (child, noise) = pair
            return child.puct(noise), child

        puct_pairs = list(map(puct, zip(self.children, noise)))
        puct_pairs.sort(key=lambda p: p[0], reverse=True)

        # Select child
        for (_, child) in puct_pairs:
            child_sel = child.select()

            if child_sel is not None:
                return child_sel
        
        return None

    def pv(self) -> list[str]:
        """Returns the PV line from this node, excluding the action from this node."""
        if len(self.children) == 0:
            return []
        
        self.children.sort(key=lambda nd: nd.n, reverse=True)
        return [self.children[0].action] + self.children[0].pv()

    def advance(self, action):
        """Destroys references to parent and children, and return
           a subtree under a specific action."""

        ret = None

        for child in self.children:
            if child.action == action:
                ret = child
            else:
                child.advance(None)

        del self.children

        self.parent = None
        return ret