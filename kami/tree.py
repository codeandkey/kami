# Tree type.

import consts
from mctsbatch import MCTSBatch
from node import Node
from position import Position

import chess
import numpy as np

class Tree():
    """Provides an interface for an MCTS search tree."""

    def __init__(self, pos=Position()):
        """Initializes a new tree with a single root node."""

        self.root    = Node(pos.board.turn)
        self.pos     = pos
        self.nodemap = {}

    def push(self, action):
        self.root = self.root.advance(action)
    
    def next_batch(self, maxsize=consts.BATCH_SIZE):
        """Gets the next batch from the tree. The batch will contain
           up to consts.MAX_BATCHSIZE nodes. It is possible for the
           returned batch to contain 0 nodes."""

        out = MCTSBatch(maxsize)

        for _ in range(maxsize):
            selected = self.root.select()

            # Stop if MCTS fails and there is no node to select.
            if selected is None:
                break

            # Get the position at this node.
            action_path = selected.action_path(self.root)

            for mv in action_path:
                self.pos.push(mv)

            # Check if position is terminal.
            result = self.pos.is_game_over()

            if result is not None:
                # Terminal! Backprop here and call it good.
                selected.backprop(result, 1)
            else:
                # Nonterminal, claim this node and add it to the batch.
                # Compute next moves and lmm

                moves, lmm = self.pos.moves()
                headers, frames = self.pos.get_input()

                # Map the node id so we can expand it later.
                self.nodemap[id(selected)] = selected

                out.add(headers, frames, lmm, moves, id(selected))

                selected.claimed = True
            
            # Reset position
            for _ in action_path:
                self.pos.pop()

        return out

    def choose(self, temperature=consts.TREE_TEMPERATURE):
        """Picks the next action node."""

        def nn(node):
            return node.n ** (1 / temperature)

        pairs = list(map(lambda nd: (nd, nn(nd) + 1), self.root.children))
        nodes = list(map(lambda p: p[0], pairs))
        p = np.array(list(map(lambda p: p[1], pairs)))
        p = p / p.sum()

        selected = np.random.choice(nodes, p=p)

        return selected
    
    def expand(self, result: dict):
        """Expands one or more nodes in the tree."""

        for i in range(len(result['value'])):
            target = self.nodemap[result['nodes'][i]]

            actions = result['actions'][i]
            policy = result['policy'][i]

            def get_policy_value(action):
                mv = chess.Move.from_uci(action)

                src = mv.from_square
                dst = mv.to_square

                if target.turn == chess.BLACK:
                    src = 63 - src
                    dst = 63 - dst
                
                return policy[src * 64 + dst]

            actual_policy = list(map(get_policy_value, actions))

            target.expand(
                actions,
                actual_policy,
                result['value'][i]
            )

            del self.nodemap[result['nodes'][i]]