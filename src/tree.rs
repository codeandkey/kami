use crate::input::treebatch::TreeBatch;
use crate::model::Output;
use crate::node::{Node, TerminalStatus};
use crate::position::Position;

use chess::ChessMove;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};
use std::sync::mpsc::{channel, Sender};
use std::thread::{spawn, JoinHandle};

const EXPLORATION: f32 = 1.414; // MCTS exploration parameter - theoretically sqrt(2)
const POLICY_SCALE: f32 = 1.0; // MCTS parameter ; how important is policy in UCT calculation

/// Status of a single node.
#[derive(Clone, Serialize, Deserialize)]
pub struct StatusNode {
    pub action: String,
    pub n: u32,
    pub nn: f32,
    pub p_pct: f32,
    pub w: f32,
    pub q: f32,
}

/// Status of the first level of children.
#[derive(Clone, Serialize, Deserialize)]
pub struct Status {
    pub nodes: Vec<StatusNode>,
    pub total_n: u32,
    pub total_nn: f32,
    pub temperature: f32,
}

impl Status {
    pub fn print(&self) {
        println!(
            "+=> Tree status: {} nodes searched, current score {:3.2}, temperature {}",
            self.total_n, self.nodes[0].q, self.temperature
        );
        for nd in &self.nodes {
            println!(
                "|=> {:>5} | N: {:4.1}% [{:>4}] | P: {:3.1}% | Q: {:3.2}",
                nd.action,
                nd.nn * 100.0 / self.total_nn,
                nd.n,
                nd.p_pct * 100.0,
                nd.q
            );
        }
        println!(
            "+=> End tree status, deterministic bestmove {:>5}",
            self.nodes[0].action
        );
    }
}

/// Response to tree batch requests.
/// Can contain either a new batch or a request to stop.
pub enum BatchResponse {
    NextBatch(Box<TreeBatch>),
    Stop,
}

/// Response to a tree status requests.
/// Can contain either a serializable tree status or a request to stop.
pub enum StatusResponse {
    NextStatus(Option<Status>),
    Stop,
}

/// Request to the tree service.
///
/// BuildBatch(tx) requests a new batch of nodes from the tree. The generated batch is sent back through tx.
/// Expand(out, batch) applies network results <output> over nodes in <batch>.
/// RequestStop requests the tree service to stop sending batches or status to any pending receivers.
/// Done requests the tree service to stop.
pub enum TreeReq {
    BuildBatch(Sender<BatchResponse>),
    Expand(Box<Output>, Box<TreeBatch>),
    RequestStop,
    GetStatus(Sender<StatusResponse>),
    Done,
}

/// Tree service object.
/// Manages a search tree and modifies it on a single thread using mpsc requests.
pub struct Tree {
    nodes: Vec<Node>,
    pos: Position,
    temperature: f32,
    batch_size: usize,
}

impl Tree {
    /// Creates a new tree service with an initial position.
    /// The initial tree has a single root node with no action.
    pub fn new(rootpos: Position, temp: f32, batch_size: usize) -> Self {
        Tree {
            nodes: vec![Node::root()],
            pos: rootpos,
            temperature: temp,
            batch_size: batch_size,
        }
    }

    /// Starts the tree service.
    /// Returns a Sender which can be used to send requests to the tree, as
    /// well as a JoinHandle for the tree service thread. The service can only
    /// be stopped by sending the TreeReq::Done request.
    ///
    /// The JoinHandle will return ownership of Tree to the joining thread,
    /// allowing the final state of the tree to be serialized.
    pub fn run(mut self) -> (Sender<TreeReq>, JoinHandle<Tree>) {
        let (inp, rx) = channel();

        let handle = spawn(move || {
            let mut stop_requested = false;

            loop {
                // Wait for batch request
                match rx.recv().expect("tree recv failed") {
                    TreeReq::BuildBatch(resp_tx) => {
                        if stop_requested {
                            resp_tx.send(BatchResponse::Stop).expect("tree send failed");
                        } else {
                            // Build batch and send it
                            let next_batch = Box::new(self.make_batch(self.batch_size));

                            resp_tx
                                .send(BatchResponse::NextBatch(next_batch))
                                .expect("tree send failed");
                        }
                    }
                    TreeReq::GetStatus(resp_tx) => {
                        if stop_requested {
                            resp_tx
                                .send(StatusResponse::Stop)
                                .expect("tree send failed");
                        } else {
                            // Get status and send it over
                            resp_tx
                                .send(StatusResponse::NextStatus(self.get_status()))
                                .expect("tree send failed");
                        }
                    }
                    TreeReq::Expand(output, batch) => {
                        for i in 0..batch.get_inner().get_size() {
                            self.expand(
                                batch.get_selected(i),
                                output.get_policy(i),
                                batch.get_inner().get_moves(i),
                            );
                            self.backprop(batch.get_selected(i), output.get_value(i));
                        }
                    }
                    TreeReq::RequestStop => stop_requested = true,
                    TreeReq::Done => break,
                }
            }

            println!("Tree thread stopped.");

            self
        });

        println!("Started tree.");

        (inp, handle)
    }

    /// Gets a serializable tree status object.
    /// Returns None if there are no root children.
    pub fn get_status(&self) -> Option<Status> {
        if self[0].children.is_none() {
            return None;
        }

        let children = self[0].children.as_ref().unwrap().clone();
        let p_total = self[0].p_total;
        let mut nodes = children
            .iter()
            .map(|&c| StatusNode {
                action: self[c].action.unwrap().to_string(), // action string
                n: self[c].n,                                // visit count
                nn: (self[c].n as f32).powf(1.0 / self.temperature), // normalized visit count
                w: self[c].w,                                // total value
                p_pct: self[c].p / p_total,                  // normalized policy
                q: self[c].q(),                              // average value
            })
            .collect::<Vec<StatusNode>>();

        let total_nn: f32 = nodes.iter().map(|x| x.nn).sum();
        let total_n: u32 = nodes.iter().map(|x| x.n).sum();

        // Reorder children by N
        nodes.sort_unstable_by(|a, b| b.n.cmp(&a.n));

        Some(Status {
            nodes: nodes,
            total_n: total_n,
            total_nn: total_nn,
            temperature: self.temperature,
        })
    }

    /// Generates a single batch with maximum size <bsize>.
    /// Returns the generated batch.
    fn make_batch(&mut self, bsize: usize) -> TreeBatch {
        let mut b = TreeBatch::new(bsize);
        self.build_batch(&mut b, 0, bsize);
        b
    }

    /// Selects nodes for a batch.
    /// Returns the number of positions in the new Batch.
    fn build_batch(&mut self, b: &mut TreeBatch, this: usize, allocated: usize) -> usize {
        // Is this node claimed? If so, stop here.
        if self[this].claim {
            return 0;
        }

        // Does this node have children? If not, try and claim this node.
        if self[this].children.is_none() {
            if self[this].claim {
                return 0;
            }

            // Check if this node is terminal. If so, immediately backprop the value.
            let terminal = match self[this].terminal {
                TerminalStatus::Terminal(r) => Some(r),
                TerminalStatus::NotTerminal => None,
                TerminalStatus::Unknown => match self.pos.is_game_over() {
                    Some(res) => {
                        self[this].terminal = TerminalStatus::Terminal(res);
                        Some(res)
                    }
                    None => {
                        self[this].terminal = TerminalStatus::NotTerminal;
                        None
                    }
                },
            };

            if let Some(mut res) = terminal {
                if res == -1.0 {
                    res = 1.0; // If the position is checkmate, then this node (decision) has a high value.
                }

                self.backprop(this, res);
                return 0;
            } else {
                // Nonterminal, claim the node and add to the batch.
                self[this].claim = true;
                b.add(&self.pos, this);

                return 1;
            }
        }

        // Node has children, so we walk through them in the proper order and allocate batches accordingly.
        let mut total_batch_size = 0;

        // Collect node children IDs, order by UCT
        let children = self[this].children.as_ref().unwrap().clone();
        let cur_n = self[this].n;
        let cur_ptotal = self[this].p_total;

        let mut pairs: Vec<(usize, f32)> = children
            .iter()
            .map(|&cidx| {
                let child = &mut self[cidx];
                let uct = child.q()
                    + (child.p / cur_ptotal) * POLICY_SCALE
                    + EXPLORATION * ((cur_n as f32).ln() / (child.n as f32 + 1.0)).sqrt();

                (cidx, uct)
            })
            .collect();

        pairs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut uct_total: f32 = pairs.iter().map(|x| x.1).sum();

        // Iterate over children
        for (child, uct) in pairs {
            let remaining = allocated - total_batch_size;

            if remaining <= 0 {
                break;
            }

            let mut child_alloc = (remaining as f32 * (uct / uct_total)).ceil() as usize;
            uct_total -= uct;

            if child_alloc == 0 {
                continue;
            }

            if child_alloc > remaining {
                child_alloc = remaining;
            }

            // Perform child action
            assert!(self.pos.make_move(self[child].action.unwrap()));

            // Allocate batches
            total_batch_size += self.build_batch(b, child, child_alloc);

            // Unmake child move
            self.pos.unmake_move();
        }

        total_batch_size
    }

    /// Backpropagates a value up through the tree.
    fn backprop(&mut self, idx: usize, value: f32) {
        let mut node = &mut self[idx];

        node.n += 1;
        node.w += value;

        if let Some(pidx) = node.parent {
            self.backprop(pidx, -value);
        }
    }

    /// Expands a node.
    /// The node <idx> is assigned children from <moves>, and the policy from <policy> is applied to each child.
    fn expand(&mut self, idx: usize, policy: &[f32], moves: &[ChessMove]) {
        let mut new_p_total = 0.0;
        let mut new_children: Vec<usize> = Vec::new();

        assert_ne!(moves.len(), 0);

        for mv in moves {
            let p = policy[mv.get_source().to_index() * 64 + mv.get_dest().to_index()].exp();

            new_p_total += p;
            new_children.push(self.nodes.len());

            self.nodes.push(Node::child(idx, p, *mv));
        }

        assert_ne!(new_p_total, 0.0);

        self[idx].p_total = new_p_total;
        self[idx].children = Some(new_children);
        self[idx].claim = false;
    }

    /// Selects an action randomly from the tree given an MCTS temperature.
    pub fn select(&self) -> chess::ChessMove {
        let child_nodes = self.nodes[0].children.as_ref().unwrap().clone();
        let mut actions = Vec::new();
        let mut probs = Vec::new();

        for &nd in child_nodes.iter() {
            actions.push(self[nd].action.unwrap());
            probs.push(((self[nd].n + 1) as f64).powf(1.0 / self.temperature as f64));
        }

        let index = WeightedIndex::new(probs).expect("failed to initialize rand index");
        let mut rng = thread_rng();

        return actions[index.sample(&mut rng)];
    }

    /// Gets MCTS visit counts for the root children (in LMM format)
    /// Output is normalized with temperature + softmax function.
    pub fn get_mcts_data(&self) -> Vec<f32> {
        let child_nodes = self.nodes[0].children.as_ref().unwrap().clone();
        let mut actions = Vec::new();
        let mut probs = Vec::new();

        for &nd in child_nodes.iter() {
            actions.push(self[nd].action.unwrap());
            probs.push(((self[nd].n + 1) as f32).powf(1.0 / self.temperature));
        }

        let sum: f32 = probs.iter().sum();
        let mut out = [0.0; 4096];

        for i in 0..probs.len() {
            out[actions[i].get_source().to_index() * 64 + actions[i].get_dest().to_index()] =
                probs[i] / sum;
        }

        return out.to_vec();
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::model::{mock::MockModel, Model};
    

    /// Tests the tree can be initialized without crashing.
    #[test]
    fn tree_can_init() {
        Tree::new(Position::new(), 1.0, 16);
    }

    /// Tests the tree service can be started and stopped without crashing.
    #[test]
    fn tree_can_start_stop() {
        let (tx, handle) = Tree::new(Position::new(), 1.0, 16).run();

        tx.send(TreeReq::Done).expect("Failed to write to tree tx.");

        handle.join().expect("Failed to join tree thread.");
    }

    /// Tests the tree service can build a batch.
    #[test]
    fn tree_can_build_batch() {
        let (tx, handle) = Tree::new(Position::new(), 1.0, 16).run();
        let (btx, brx) = channel();

        tx.send(TreeReq::BuildBatch(btx))
            .expect("Failed to write to tree tx.");

        let new_batch = match brx.recv().expect("Failed to receive from request rx.") {
            BatchResponse::NextBatch(b) => b,
            BatchResponse::Stop => panic!("Unexpected early tree stop!"),
        };

        // The first batch should always be of size 1.
        assert_eq!(new_batch.get_inner().get_size(), 1);

        tx.send(TreeReq::Done).expect("Failed to write to tree tx.");

        handle.join().expect("Failed to join tree thread.");
    }

    /// Tests the tree service can expand a node with dummy network results.
    #[test]
    fn tree_can_expand_node() {
        let (tx, handle) = Tree::new(Position::new(), 1.0, 16).run();
        let (btx, brx) = channel();

        tx.send(TreeReq::BuildBatch(btx))
            .expect("Failed to write to tree tx.");

        let new_batch = match brx.recv().expect("Failed to receive from request rx.") {
            BatchResponse::NextBatch(b) => b,
            BatchResponse::Stop => panic!("Unexpected early tree stop!"),
        };

        // The first batch should always be of size 1.
        assert_eq!(new_batch.get_inner().get_size(), 1);

        // Generate dummy network output and send it to the service.
        let output = MockModel::generate()
            .expect("model gen failed")
            .execute(new_batch.get_inner());

        tx.send(TreeReq::Expand(Box::new(output), new_batch))
            .expect("Failed to write to tree tx.");
        tx.send(TreeReq::Done).expect("Failed to write to tree tx.");

        handle.join().expect("Failed to join tree thread.");
    }

    /// Checks the tree can return correct MCTS data.
    #[test]
    fn tree_can_get_mcts_data() {
        let (tx, handle) = Tree::new(Position::new(), 1.0, 16).run();
        let (btx, brx) = channel();

        tx.send(TreeReq::BuildBatch(btx))
            .expect("Failed to write to tree tx.");

        let new_batch = match brx.recv().expect("Failed to receive from request rx.") {
            BatchResponse::NextBatch(b) => b,
            BatchResponse::Stop => panic!("Unexpected early tree stop!"),
        };

        // The first batch should always be of size 1.
        assert_eq!(new_batch.get_inner().get_size(), 1);

        // Generate dummy network output and send it to the service.
        let output = MockModel::generate()
            .expect("model gen failed")
            .execute(new_batch.get_inner());

        tx.send(TreeReq::Expand(Box::new(output), new_batch))
            .expect("Failed to write to tree tx.");
        tx.send(TreeReq::Done).expect("Failed to write to tree tx.");

        let final_tree = handle.join().expect("Failed to join tree thread.");
        let mcts_data = final_tree.get_mcts_data();

        // MCTS data should be equal across all nodes.
        // 20 legal first moves -> 1/20 expected value per legal move.

        let children = final_tree[0].children.as_ref().unwrap().clone();
        let mut action_indices = Vec::new();

        for &nd in &children {
            let action = final_tree[nd].action.unwrap();
            let src = action.get_source().to_index();
            let dst = action.get_dest().to_index();

            let ind = src * 64 + dst;
            action_indices.push(ind);

            assert_eq!(mcts_data[ind], 1.0 / 20.0);
        }

        // Check the rest of the data is 0
        for i in 0..4096 {
            if action_indices.contains(&i) {
                continue;
            }

            assert_eq!(mcts_data[i], 0.0);
        }
    }
}
