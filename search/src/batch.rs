/// Manages multiple inputs to the network.
use crate::position::Position;
use crate::consts;
use chess::{ChessMove, Color};

/// Manages an executed batch.
pub struct BatchResult {
    pub moves: Vec<Vec<ChessMove>>,
    pub nodes: Vec<usize>,
    pub value: Vec<f64>,
    pub policy: Vec<f64>,
    pub size: usize,
}

impl BatchResult {
    /// Gets the corresponding policy output value given a batch index, action and pov.
    pub fn policy_for_action(&self, idx: usize, action: &ChessMove, pov: Color) -> f64 {
        match pov {
            Color::White => {
                self.policy[idx * 4096 + action.get_source().to_index() * 64 + action.get_dest().to_index()]
            },
            Color::Black => {
                self.policy[idx * 4096 + (63 - action.get_source().to_index()) * 64 + (63 - action.get_dest().to_index())]
            }
        }
    }
}

/// Manages a batch of inputs to the model.
pub struct Batch {
    headers: Vec<f32>,
    frames: Vec<f32>,
    lmm: Vec<f32>,
    moves: Vec<Vec<ChessMove>>,
    nodes: Vec<usize>,
    current_size: usize,
}

impl Batch {
    /// Returns a new batch instance with <reserve_size> preallocated space.
    pub fn new(reserve_size: usize) -> Self {
        let mut headers = Vec::new();
        let mut frames = Vec::new();
        let mut moves = Vec::new();
        let mut lmm = Vec::new();
        let mut nodes = Vec::new();

        headers.reserve(reserve_size * consts::HEADER_SIZE);
        frames.reserve(reserve_size * consts::FRAME_COUNT * consts::FRAME_SIZE);
        moves.reserve(reserve_size);
        nodes.reserve(reserve_size);
        lmm.reserve(4096 * reserve_size);

        Batch {
            headers: headers,
            frames: frames,
            current_size: 0,
            lmm: lmm,
            moves: moves,
            nodes: nodes,
        }
    }

    /// Adds a position snapshot to the batch.
    pub fn add(&mut self, p: &Position, node: usize) {
        // Store position network inputs
        self.headers.extend_from_slice(p.get_headers());
        self.frames.extend_from_slice(p.get_frames());

        // Generate moves and LMM
        let (lmm, moves) = p.get_lmm();

        self.moves.push(moves);
        self.nodes.push(node);
        self.lmm.extend_from_slice(&lmm);
        self.current_size += 1;
    }

    /// Gets the number of positions in this batch.
    pub fn get_size(&self) -> usize {
        self.current_size
    }

    /// Returns the batch frames input data.
    pub fn get_frames(&self) -> &[f32] {
        &self.frames
    }

    /// Returns the batch legal move mask input data.
    pub fn get_lmm(&self) -> &[f32] {
        &self.lmm
    }

    /// Returns the batch headers input data.
    pub fn get_headers(&self) -> &[f32] {
        &self.headers
    }

    /// Converts this batch into a result.
    pub fn into_result(self, policy: Vec<f64>, value: Vec<f64>) -> BatchResult {
        BatchResult {
            policy: policy,
            value: value,
            nodes: self.nodes,
            moves: self.moves,
            size: self.current_size
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// Tests the batch can be initialized.
    #[test]
    fn batch_can_initialize() {
        Batch::new(16);
    }

    /// Tests the batch can be added to.
    #[test]
    fn batch_can_add() {
        let mut b = Batch::new(16);
        b.add(&Position::new(), 0);
    }

    /// Tests the batch size is correctly updated.
    #[test]
    fn batch_can_get_size() {
        let mut b = Batch::new(16);

        b.add(&Position::new(), 0);
        assert_eq!(b.get_size(), 1);
        b.add(&Position::new(), 0);
        assert_eq!(b.get_size(), 2);
        b.add(&Position::new(), 0);
        assert_eq!(b.get_size(), 3);
    }

    /// Tests the batch can return frames data.
    #[test]
    fn batch_can_get_frames() {
        let mut b = Batch::new(16);
        b.add(&Position::new(), 0);

        assert_eq!(b.get_frames().len(), consts::TOTAL_FRAMES_SIZE);
    }

    /// Tests the batch can return header data.
    #[test]
    fn batch_can_get_headers() {
        let mut b = Batch::new(16);
        b.add(&Position::new(), 0);

        assert_eq!(b.get_headers().len(), consts::HEADER_SIZE);
    }
}
