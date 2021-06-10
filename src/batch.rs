/**
 * Manages a single batch of positions.
 */
use crate::model;
use crate::position::Position;
use chess::ChessMove;

/// Manages a batch of inputs to the model.
pub struct Batch {
    headers: Vec<f32>,
    frames: Vec<f32>,
    lmm: Vec<f32>,
    selected: Vec<usize>,
    moves: Vec<Vec<ChessMove>>,
    current_size: usize,
}

impl Batch {
    /// Returns a new batch instance with <max_batch_size> preallocated space.
    pub fn new(max_batch_size: usize) -> Self {
        let mut headers = Vec::new();
        let mut frames = Vec::new();

        headers.reserve(max_batch_size * 24);
        frames.reserve(max_batch_size * model::PLY_FRAME_COUNT * model::PLY_FRAME_SIZE * 64);

        Batch {
            headers: headers,
            frames: frames,
            current_size: 0,
            lmm: Vec::new(),
            selected: Vec::new(),
            moves: Vec::new(),
        }
    }

    /// Adds a position snapshot to the batch.
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

    /// Gets the number of positions in this batch.
    pub fn get_size(&self) -> usize {
        self.current_size
    }

    /// Gets the node index for the <idx>-th position in this batch.
    pub fn get_selected(&self, idx: usize) -> usize {
        self.selected[idx]
    }

    /// Gets the legal moves for the <idx>-th position in this batch.
    pub fn get_moves(&self, idx: usize) -> &[ChessMove] {
        &self.moves[idx]
    }

    /// Returns the batch frames input tensor.
    pub fn get_frames(&self) -> &[f32] {
        &self.frames
    }

    /// Returns the batch legal move mask input tensor.
    pub fn get_lmm(&self) -> &[f32] {
        &self.lmm
    }

    /// Returns the batch headers input tensor.
    pub fn get_headers(&self) -> &[f32] {
        &self.headers
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

    /// Tests the selected nodes can be returned.
    #[test]
    fn batch_can_get_selected() {
        let mut b = Batch::new(16);

        b.add(&Position::new(), 0);
        b.add(&Position::new(), 1);
        b.add(&Position::new(), 2);

        assert_eq!(b.get_selected(0), 0);
        assert_eq!(b.get_selected(1), 1);
        assert_eq!(b.get_selected(2), 2);
    }

    /// Tests the selected moves are correctly generated.
    #[test]
    fn batch_can_get_moves() {
        let p = Position::new();
        let moves = p.generate_moves();

        let mut b = Batch::new(4);
        b.add(&p, 0);

        let batch_moves = b.get_moves(0);

        assert_eq!(moves.len(), batch_moves.len());

        for i in 0..moves.len() {
            assert_eq!(moves[i], batch_moves[i]);
        }
    }

    /// Tests the batch can return a frames tensor.
    #[test]
    fn batch_can_get_frames_tensor() {
        let mut b = Batch::new(16);
        b.add(&Position::new(), 0);

        assert_eq!(
            b.get_frames().len(),
            model::PLY_FRAME_COUNT * model::PLY_FRAME_SIZE * 64
        );
    }

    /// Tests the batch can return a header tensor.
    #[test]
    fn batch_can_get_header_tensor() {
        let mut b = Batch::new(16);
        b.add(&Position::new(), 0);

        assert_eq!(b.get_headers().len(), model::SQUARE_HEADER_SIZE);
    }
}
