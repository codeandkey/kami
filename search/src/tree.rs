use crate::batch::{Batch, BatchResult};
use crate::node::{Node, TerminalStatus};
use crate::params::Params;
use crate::position::Position;

use chess::{ChessMove, Color};
use rand::prelude::*;
use rand::distributions::WeightedIndex;
use serde::ser::{Serialize, Serializer, SerializeSeq};
use std::cmp::Ordering;
use std::ops::{Index, IndexMut};

/// Manages a search tree.
#[derive(Clone)]
pub struct Tree {
    nodes: Vec<Node>,
    params: Params,
    position: Position,
}

impl Tree {
    /// Initializes a tree with a single root node.
    pub fn new(rootpos: Position, params: &Params) -> Self {
        Tree {
            nodes: vec![Node::root(rootpos.side_to_move())],
            params: params.clone(),
            position: rootpos,
        }
    }

    /// Gets the next batch from the tree.
    pub fn next_batch(&mut self) -> Batch {
        let mut batch = Batch::new(self.params.batch_size.into());

        for _ in 0..self.params.batch_size {
            if !self.mcts_select(&mut batch, 0) {
                break;
            }
        }
        
        return batch;
    }

    /// Advances the tree by a single action, maintaining the computed subtree.
    pub fn push(&mut self, action: ChessMove) {
        let mut new_nodes = Vec::new();
        let mut found = false;

        assert!(self[0].children.is_some());

        for c in self[0].children.as_ref().unwrap() {
            if self[*c].action.unwrap() == action {
                self.copy_subtree(*c, &mut new_nodes, None);
                found = true;
            }
        }

        assert!(found);
        self.nodes = new_nodes;
        assert!(self.position.make_move(action));
    }

    /// Copies tree data into a new packed tree structure.
    fn copy_subtree(&self, root: usize, nodes: &mut Vec<Node>, new_parent: Option<usize>) -> usize {
        // Push self
        let new_id = nodes.len();
        nodes.push(self[root].clone());

        // Push children, if there are any
        if let Some(children) = &self[root].children {
            let mut new_children = Vec::new();
            
            for nd in children {
                new_children.push(self.copy_subtree(*nd, nodes, Some(new_id)));
            }

            nodes[new_id].children = Some(new_children);
        }

        // Assign new parent
        nodes[new_id].parent = new_parent;
        return new_id;
    }

    /// Rolls out a node until a terminal is reached.
    /// Rollout path is not saved in the tree.
    /// Returns the value of the reached terminal.
    pub fn rollout(&mut self, start: usize) -> f64 {
        // Trace actions to node
        let mut path_to_start = Vec::new();
        let mut current = start;

        while current != 0 {
            path_to_start.push(current);
            current = self[current].parent.unwrap();
        }

        // Build position
        let mut pos = self.position.clone();

        for &i in path_to_start.iter().rev() {
            assert!(pos.make_move(self[i].action.unwrap()));
        }

        // Until node is terminal, expand children
        while pos.is_game_over().is_none() {
            let moves = pos.generate_moves();
            let mut rng = thread_rng();

            let mv = moves[(rng.next_u32() as usize) % moves.len()];

            assert!(pos.make_move(mv));
        }

        // It's a draw!
        if pos.is_game_over().unwrap() == 0.0 {
            return 0.0;
        }

        if self[start].color == pos.side_to_move() {
            // The node's pov color has lost.
            // So the color about to move loses, we are happy

            return 1.0;
        } else {
            return -1.0;
        }
    }

    /// Expands a batch result.
    pub fn expand(&mut self, result: BatchResult) {
        for idx in 0..result.size {
            let target = result.nodes[idx];
            let moves = &result.moves[idx];

            let mut new_children: Vec<usize> = Vec::new();
            new_children.reserve(moves.len());

            assert_ne!(moves.len(), 0);

            // Compute noise dist
            let mut noise = match moves.len() {
                1 => {
                    vec![1.0]
                },
                x => {
                    let dist = rand::distributions::Dirichlet::new_with_param(
                        self.params.puct_noise_alpha,
                        x,
                    );

                    let mut rng = thread_rng();
                    dist.sample(&mut rng)
                }
            };

            for mv in moves {
                let new_color = match self[target].color {
                    Color::White => Color::Black,
                    Color::Black => Color::White,
                };

                let model_p = result.policy_for_action(idx, &mv, self[target].color);
                let noise_p = noise.pop().unwrap();

                let p = (noise_p * self.params.puct_noise_weight) + (model_p * (1.0 - self.params.puct_noise_weight));

                assert!(!model_p.is_nan(), "policy is NaN out of network");
                new_children.push(self.nodes.len());

                self.nodes.push(Node::child(target, p.into(), *mv, new_color));
            }

            self[target].children = Some(new_children);
            self[target].claim = false;

            if self.params.rollout_weight > 0.0 {
                let rollout = self.rollout(target);
                self.backprop(target, result.value[idx] * (1.0 - self.params.rollout_weight) + rollout * self.params.rollout_weight, 0, 0);
            } else {
                self.backprop(target, result.value[idx], 0, 0);
            }
        }
    }

    /// Performs a single MCTS node selection.
    /// Returns true if a leaf was reached, false otherwise.
    fn mcts_select(&mut self, b: &mut Batch, this: usize) -> bool {
        // Is this node claimed? If so, stop here.
        if self[this].claim {
            return false;
        }

        // Does this node have children? If not, claim this node.
        if self[this].children.is_none() {
            // Check if this node is terminal. If so, immediately backprop the value.
            if matches!(self[this].terminal, TerminalStatus::Unknown) {
                self[this].terminal = match self.position.is_game_over() {
                    Some(v) => TerminalStatus::Terminal(v),
                    None => TerminalStatus::NotTerminal,
                };
            }

            if let TerminalStatus::Terminal(mut res) = self[this].terminal {
                // Our move, and we are lost.
                // We are not happy.
                
                if res == 1.0 {
                    res = -1.0;
                }

                self.backprop(this, res, 0, 1);
                return true;
            } else {
                // Nonterminal, claim the node and add to the batch.
                self[this].claim = true;
                b.add(&self.position, this);

                return true;
            }
        }

        // Node has children, so we walk through them in the proper order and try to find a new leaf.

        // Collect node children IDs, order by UCT
        let children = self[this].children.as_ref().unwrap().clone();
        let cur_n = self[this].n;

        let mut pairs: Vec<(usize, f64)> = children
            .iter()
            .map(|&cidx| {
                let child = &self[cidx];
                let uct = -child.q()
                    + (child.p
                        * self.params.puct_policy_weight
                        * (cur_n as f64).sqrt())
                        / (child.n as f64 + 1.0);

                assert!(
                    !uct.is_nan(),
                    "n: {}, w: {}, p: {}",
                    child.n,
                    child.w,
                    child.p
                );

                (cidx, uct)
            })
            .collect();

        pairs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let child = pairs[0].0;

        // Perform child action
        assert!(self.position.make_move(self[child].action.unwrap()));

        // Try MCTS iteration
        let result = self.mcts_select(b, child);

        self.position.unmake_move();

        return result;
    }

    /// Backpropagates a value up through the tree.
    fn backprop(&mut self, idx: usize, value: f64, depth: usize, terminal: u32) {
        let mut node = &mut self[idx];

        node.n += 1;
        node.tn += terminal;
        node.w += value;

        if depth > node.maxdepth {
            node.maxdepth = depth;
        }

        if let Some(pidx) = node.parent {
            self.backprop(pidx, -value, depth + 1, terminal);
        }
    }

    /// Selects an action randomly from the tree given an MCTS temperature.
    pub fn pick(&self) -> (ChessMove, f64) {
        assert!(
            self[0].children.is_some(),
            "Will not pick moves with no children (n={}, t={:?}): {}",
            self[0].n,
            self[0].terminal,
            self.position.get_fen()
        );

        let child_nodes = self.nodes[0].children.as_ref().unwrap().clone();
        let mut actions = Vec::new();
        let mut probs = Vec::new();

        let mut temp = self.params.temperature;

        if self.get_position().ply() >= self.params.temperature_drop_ply {
            temp = self.params.temperature_drop;
        }

        for &nd in child_nodes.iter() {
            actions.push((self[nd].action.unwrap(), self[nd].q()));
            probs.push(((self[nd].n + 1) as f64).powf(1.0 / temp as f64));
        }

        let index = WeightedIndex::new(probs).expect("failed to initialize rand index");
        let mut rng = thread_rng();

        return actions[index.sample(&mut rng)];
    }

    /// Gets the MCTS pairs for this tree.
    pub fn get_mcts_pairs(&self) -> Vec<(f64, String)> {
        let child_nodes = self.nodes[0].children.as_ref().unwrap().clone();

        let total_n = child_nodes.iter().map(|x| self[*x].n as f64).sum::<f64>();

        child_nodes
            .iter()
            .map(|x| (self[*x].n as f64 / total_n, self[*x].action.as_ref().unwrap().to_string()))
            .collect()
    }

    /// Gets a reference to the root position.
    pub fn get_position(&self) -> &Position {
        &self.position
    }
}

impl Index<usize> for Tree {
    type Output = Node;

    /// Gets an immutable reference to a tree node by index <idx>.
    fn index(&self, idx: usize) -> &Self::Output {
        &self.nodes[idx]
    }
}

impl IndexMut<usize> for Tree {
    /// Gets a mutable reference to a tree node by index <idx>.
    fn index_mut(&mut self, idx: usize) -> &mut Node {
        &mut self.nodes[idx]
    }
}

impl Serialize for Tree {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let empty = Vec::new();

        let children = match &self[0].children {
            Some(c) => c,
            None => &empty,
        };

        let mut state = serializer.serialize_seq(Some(children.len()))?;

        for nd in children {
            state.serialize_element(&self[*nd])?;
        }

        state.end()
    }
}