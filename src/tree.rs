use crate::batch::Batch;
use crate::model::Output;
use crate::node::Node;
use crate::position::Position;

use chess::ChessMove;
use serde::ser::{Serialize, SerializeSeq, Serializer};
use std::ops::{Index, IndexMut};
use std::sync::mpsc::{channel, Sender};
use std::thread::{spawn, JoinHandle};

const EXPLORATION: f32 = 1.414; // MCTS exploration parameter - theoretically sqrt(2)
const POLICY_SCALE: f32 = 1.0; // MCTS parameter ; how important is policy in UCT calculation

/// Request to the tree service.
///
/// BuildBatch(tx) requests a new batch of nodes from the tree. The generated batch is sent back through tx.
/// Expand(out, batch) applies network results <output> over nodes in <batch>.
/// Done requests the tree service to stop.
pub enum TreeReq {
    BuildBatch(Sender<Box<Batch>>),
    Expand(Box<Output>, Box<Batch>),
    Done,
}

/// Tree service object.
/// Manages a search tree and modifies it on a single thread using mpsc requests.
pub struct Tree {
    nodes: Vec<Node>,
    pos: Position,
}

impl Tree {
    /// Creates a new tree service with an initial position.
    /// The initial tree hjas a single root node with no action.
    pub fn new(rootpos: Position) -> Self {
        Tree {
            nodes: vec![Node::root()],
            pos: rootpos,
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
            loop {
                // Wait for batch request
                match rx.recv().expect("tree recv failed") {
                    TreeReq::BuildBatch(resp_tx) => {
                        // Build batch and send it
                        let next_batch = Box::new(self.make_batch(16));

                        resp_tx.send(next_batch).expect("tree send failed");
                    }
                    TreeReq::Expand(output, batch) => {
                        for i in 0..batch.get_size() {
                            self.expand(
                                batch.get_selected(i),
                                output.get_policy(i),
                                batch.get_moves(i),
                            );
                            self.backprop(batch.get_selected(i), output.get_value(i));
                        }
                    }
                    TreeReq::Done => break,
                }
            }

            self
        });

        (inp, handle)
    }

    /// Generates a single batch with maximum size <bsize>.
    /// Returns the generated batch.
    fn make_batch(&mut self, bsize: usize) -> Batch {
        let mut b = Batch::new(bsize);
        self.build_batch(&mut b, 0, bsize);
        b
    }

    /// Selects nodes for a batch.
    /// Returns the number of positions in the new Batch.
    fn build_batch(&mut self, b: &mut Batch, this: usize, allocated: usize) -> usize {
        // Is this node claimed? If so, stop here.
        if self[this].claim {
            return 0;
        }

        // Does this node have children? If not, try and claim this node.
        if self[this].children.is_none() {
            if self[this].claim {
                return 0;
            }

            self[this].claim = true;

            // Node is now claimed, add it to the batch.
            b.add(&self.pos, this);

            return 1;
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
                let uct = (child.w / (child.n as f32))
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

            let child_alloc = (remaining as f32 * (uct / uct_total)).round() as usize;
            uct_total -= uct;

            if child_alloc == 0 {
                continue;
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

        for mv in moves {
            let p = policy[mv.get_source().to_index() * 64 + mv.get_dest().to_index()].exp();

            new_p_total += p;
            new_children.push(self.nodes.len());

            self.nodes.push(Node::child(idx, p, *mv));
        }

        self[idx].p_total = new_p_total;
        self[idx].children = Some(new_children);
    }

    /// Gets the number of nodes in the tree.
    pub fn size(&self) -> usize {
        self.nodes.len()
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
    /// Serializes the tree.
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_seq(None)?;

        let results: Vec<Result<(), S::Error>> = self[0]
            .children
            .as_ref()
            .unwrap()
            .iter()
            .map(|x| state.serialize_element(x))
            .collect();

        for res in results {
            res?;
        }

        state.end()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::model::Model;
    use crate::models::mock::MockModel;

    /// Tests the tree can be initialized without crashing.
    #[test]
    fn tree_can_init() {
        Tree::new(Position::new());
    }

    /// Tests the tree service can be started and stopped without crashing.
    #[test]
    fn tree_can_start_stop() {
        let (tx, handle) = Tree::new(Position::new()).run();

        tx.send(TreeReq::Done).expect("Failed to write to tree tx.");

        handle.join().expect("Failed to join tree thread.");
    }

    /// Tests the tree service can build a batch.
    #[test]
    fn tree_can_build_batch() {
        let (tx, handle) = Tree::new(Position::new()).run();
        let (btx, brx) = channel();

        tx.send(TreeReq::BuildBatch(btx))
            .expect("Failed to write to tree tx.");

        let new_batch = brx.recv().expect("Failed to receive from request rx.");

        // The first batch should always be of size 1.
        assert_eq!(new_batch.get_size(), 1);

        tx.send(TreeReq::Done).expect("Failed to write to tree tx.");

        handle.join().expect("Failed to join tree thread.");
    }

    /// Tests the tree service can expand a node with dummy network results.
    #[test]
    fn tree_can_expand_node() {
        let (tx, handle) = Tree::new(Position::new()).run();
        let (btx, brx) = channel();

        tx.send(TreeReq::BuildBatch(btx))
            .expect("Failed to write to tree tx.");

        let new_batch = brx.recv().expect("Failed to receive from request rx.");

        // The first batch should always be of size 1.
        assert_eq!(new_batch.get_size(), 1);

        // Generate dummy network output and send it to the service.
        let output = MockModel::new(None).execute(&new_batch);

        tx.send(TreeReq::Expand(Box::new(output), new_batch))
            .expect("Failed to write to tree tx.");
        tx.send(TreeReq::Done).expect("Failed to write to tree tx.");

        let tree = handle.join().expect("Failed to join tree thread.");

        // The resulting tree should have expanded 20 moves, with 21 nodes total.
        assert_eq!(tree.size(), 21);
    }
}
