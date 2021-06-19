/// Manages multiple inputs to the network.
use crate::model;
use crate::position::Position;
use chess::ChessMove;
use serde::ser::{Serialize, SerializeStruct, Serializer};

/// Manages a batch of inputs to the model.
pub struct Batch {
    headers: Vec<f32>,
    frames: Vec<f32>,
    lmm: Vec<f32>,
    pov: Vec<f32>,
    moves: Vec<Vec<ChessMove>>,
    current_size: usize,
}

impl Batch {
    /// Returns a new batch instance with <reserve_size> preallocated space.
    pub fn new(reserve_size: usize) -> Self {
        let mut headers = Vec::new();
        let mut frames = Vec::new();
        let mut povs = Vec::new();
        let mut moves = Vec::new();
        let mut lmm = Vec::new();

        headers.reserve(reserve_size * 24);
        povs.reserve(reserve_size);
        frames.reserve(reserve_size * model::FRAMES_SIZE);
        moves.reserve(reserve_size);
        lmm.reserve(4096 * reserve_size);

        Batch {
            headers: headers,
            frames: frames,
            current_size: 0,
            lmm: lmm,
            moves: moves,
            pov: povs,
        }
    }

    /// Adds a position snapshot to the batch.
    pub fn add(&mut self, p: &Position) {
        // Store position network inputs
        self.headers.extend_from_slice(p.get_headers());
        self.frames.extend_from_slice(p.get_frames());
        self.pov.push(p.get_pov());

        // Generate moves and LMM
        let (lmm, moves) = p.get_lmm();

        self.moves.push(moves);
        self.lmm.extend_from_slice(&lmm);
        self.current_size += 1;
    }

    /// Gets the number of positions in this batch.
    pub fn get_size(&self) -> usize {
        self.current_size
    }

    /// Gets the legal moves for the <idx>-th position in this batch.
    pub fn get_moves(&self, idx: usize) -> &[ChessMove] {
        &self.moves[idx]
    }

    /// Returns the batch frames input data.
    pub fn get_frames(&self) -> &[f32] {
        &self.frames
    }

    /// Returns the batch legal move mask input data.
    pub fn get_lmm(&self) -> &[f32] {
        &self.lmm
    }

    /// Returns the batch POV data.
    pub fn get_pov(&self) -> &[f32] {
        &self.pov
    }

    /// Returns the batch headers input tensor.
    pub fn get_headers(&self) -> &[f32] {
        &self.headers
    }
}

impl Serialize for Batch {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("batch", 5)?;

        state.serialize_field("headers", &self.headers)?;
        state.serialize_field("frames", &self.frames)?;
        state.serialize_field("lmm", &self.lmm)?;

        state.end()
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
        b.add(&Position::new());
    }

    /// Tests the batch size is correctly updated.
    #[test]
    fn batch_can_get_size() {
        let mut b = Batch::new(16);

        b.add(&Position::new());
        assert_eq!(b.get_size(), 1);
        b.add(&Position::new());
        assert_eq!(b.get_size(), 2);
        b.add(&Position::new());
        assert_eq!(b.get_size(), 3);
    }

    /// Tests the selected moves are correctly generated.
    #[test]
    fn batch_can_get_moves() {
        let p = Position::new();
        let moves = p.generate_moves();

        let mut b = Batch::new(4);
        b.add(&p);

        let batch_moves = b.get_moves(0);

        assert_eq!(moves.len(), batch_moves.len());

        for i in 0..moves.len() {
            assert_eq!(moves[i], batch_moves[i]);
        }
    }

    /// Tests the batch can return frames data.
    #[test]
    fn batch_can_get_frames() {
        let mut b = Batch::new(16);
        b.add(&Position::new());

        assert_eq!(b.get_frames().len(), model::FRAMES_SIZE);
    }

    /// Tests the batch can return header data.
    #[test]
    fn batch_can_get_headers() {
        let mut b = Batch::new(16);
        b.add(&Position::new());

        assert_eq!(b.get_headers().len(), model::SQUARE_HEADER_SIZE);
    }

    /// Tests the batch can return pov data.
    #[test]
    fn batch_can_get_pov() {
        let mut b = Batch::new(16);
        b.add(&Position::new());

        assert_eq!(b.get_headers().len(), model::SQUARE_HEADER_SIZE);
    }
}
