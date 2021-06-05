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
    frames: Vec<f32>,
    lmm: Vec<f32>,
    selected: Vec<usize>,
    moves: Vec<Vec<ChessMove>>,
    current_size: usize,
}

impl Batch {
    pub fn new(max_batch_size: usize) -> Self {
        let mut headers = Vec::new();
        let mut frames = Vec::new();

        headers.reserve(max_batch_size * 24);
        frames.reserve(max_batch_size * net::PLY_FRAME_COUNT * net::PLY_FRAME_SIZE * 64);

        Batch {
            headers: headers,
            frames: frames,
            current_size: 0,
            lmm: Vec::new(),
            selected: Vec::new(),
            moves: Vec::new(),
        }
    }

    pub fn add(&mut self, p: &Position, idx: usize) {
        // Store position network inputs
        self.headers.extend_from_slice(p.get_headers());
        self.frames.extend_from_slice(p.get_frames());

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

    pub fn get_size(&self) -> usize {
        self.current_size
    }

    pub fn get_selected(&self, idx: usize) -> usize {
        self.selected[idx]
    }

    pub fn get_moves(&self, idx: usize) -> &[ChessMove] {
        &self.moves[idx]
    }

    pub fn get_frames_tensor(&self) -> Tensor<f32> {
        Tensor::from(self.frames.as_slice())
    }

    pub fn get_lmm_tensor(&self) -> Tensor<f32> {
        Tensor::from(self.lmm.as_slice())
    }

    pub fn get_header_tensor(&self) -> Tensor<f32> {
        Tensor::from(self.headers.as_slice())
    }
}
