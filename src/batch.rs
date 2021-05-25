/**
 * Manages a single batch of positions.
 */

use crate::net;
use crate::node::Node;
use crate::position::Position;
use crate::tree::Tree;

use chess::ChessMove;
use tensorflow::Tensor;

pub struct Batch {
    headers: Vec<f32>,
    frames: Vec<Vec<f32>>,
    lmm: Vec<f32>,
    selected: Vec<usize>,
    moves: Vec<Vec<ChessMove>>,
    current_size: usize,
    max_size: usize,
}

impl Batch {
    pub fn new(max_batch_size: usize) -> Self {
        let mut headers: Vec<f32> = Vec::new();
        let mut frames: Vec<Vec<f32>> = Vec::new();

        headers.reserve(max_batch_size * 24);
        frames.reserve(net::PLY_FRAME_COUNT);
        
        for f in &mut frames {
            f.reserve(net::PLY_FRAME_SIZE * 64);
        }

        Batch {
            headers: headers,
            frames: frames,
            current_size: 0,
            max_size: max_batch_size,
            lmm: Vec::new(),
            selected: Vec::new(),
            moves: Vec::new(),
        }
    }

    pub fn build(max_batch_size: usize, rootpos: &mut Position, tree: &mut Tree) -> Self {
        let mut new_batch = Batch::new(max_batch_size);
        new_batch.build_from(rootpos, tree);
        new_batch
    }

    pub fn add(&mut self, p: &Position, idx: usize) {
        let headers = p.get_input().get_headers();
        let frames = p.get_input().get_frames();

        // Store position network inputs
        self.headers.extend_from_slice(headers);

        for (i, f) in frames.enumerate() {
            self.frames[i].extend_from_slice(f.get_data());
        }

        // Store node identifier
        self.selected.push(idx);

        // Generate moves and LMM
        let moves = p.generate_moves();

        let mut lmm = [0.0; 4096];
        for mv in &moves {
            lmm[mv.get_source().to_index() * 64 + mv.get_dest().to_index()] = 1.0;
        }

        self.moves.push(moves);
        self.lmm.extend_from_slice(&lmm);
        self.current_size += 1;
    }

    pub fn apply_results(&self, results: net::Output, tree: &mut Tree) {
        // Walk through selected nodes.

        for (i, idx) in self.selected.iter().enumerate() {
            // Backprop node value
            tree.backprop(*idx, results.get_value(i));

            // Expand node children, apply policy
            let policy = results.get_policy(i);
            let mut policy_total = 0.0;

            // Walk through moves once to find policy total
            for mv in &self.moves[i] {
                policy_total += policy[mv.get_source().to_index() * 64 + mv.get_dest().to_index()];
            }

            // Expand new node children
            let mut new_children: Vec<usize> = Vec::new();

            for mv in &self.moves[i] {
                let child = Node::child(
                    *idx,
                    policy[mv.get_source().to_index() * 64 + mv.get_dest().to_index()].exp() / policy_total.exp(),
                    *mv
                );

                new_children.push(tree.add(child));
            }

            tree.nodes()[*idx].assign_children(new_children);
            tree.assign_children(*idx, new_children);

            // Unclaim node
            tree.nodes()[*idx].unclaim();
        }
    }

    pub fn get_size(&self) -> usize {
        self.current_size
    }

    pub fn get_frame_tensors(&self) -> Vec<Tensor<f32>> {
        self.frames.iter().map(|f| {
            Tensor::from(f.as_slice())
        }).collect::<Vec<Tensor<f32>>>()
    }

    pub fn get_lmm_tensor(&self) -> Tensor<f32> {
        Tensor::from(self.lmm.as_slice())
    }

    pub fn get_header_tensor(&self) -> Tensor<f32> {
        Tensor::from(self.headers.as_slice())
    }

    pub fn build_from(&mut self, pos: &mut Position, tree: &mut Tree) {
        self.build_batch(0, self.max_size, tree, pos);
    }

    /**
     * Selects nodes for a batch.
     * Returns a list of selected node IDs, along with the legal moves for each leaf.
     */
    fn build_batch(&mut self, start: usize, allocated: usize, tree: &mut Tree, pos: &mut Position) -> usize {
        // Is this node claimed? If so, stop here.
        if tree.nodes()[start].is_claimed() {
            return 0;
        }

        // Does this node have children? If not, try and claim this node.
        if !tree.nodes()[start].has_children() {
            if !tree.nodes()[start].claim() {
                return 0;
            }

            // Node is now claimed, add it to the batch.
            self.add(pos, start);

            return 1;
        }

        // Node has children, so we walk through them in the proper order and allocate batches accordingly.
        let mut total_batch_size = 0;
        let made_move = false;

        // Collect node children IDs, order by UCT
        let pairs = tree.uct_pairs(start);

        let mut uct_total: f32 = pairs.iter().map(|x| x.1).sum();

        // Iterate over children
        for (child, uct) in &pairs {
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
            pos.make_move(tree.nodes()[*child].action().unwrap());

            // Allocate batches
            total_batch_size += self.build_batch(*child, child_alloc, tree, pos);

            // Unmake child move
            pos.unmake_move();
        }

        total_batch_size
    }
}

