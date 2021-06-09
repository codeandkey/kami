/// Contains structures used in model training.

/// Manages a batch of training data.
pub struct TrainBatch {
    mcts_frames: Vec<f32>, // policy loss
    results: Vec<f32>,     // value loss
}

impl TrainBatch {
    /// Returns a new empty instance of TrainBatch.
    pub fn new() -> Self {
        TrainBatch {
            mcts_frames: Vec::new(),
            results: Vec::new(),
        }
    }

    /// Adds a position to the training batch.
    /// NOTE: No transformation is done to the result or MCTS frame, they should both be from the appropriate POV already.
    pub fn add(&mut self, mcts: &[f32], result: f32) {
        assert_eq!(mcts.len(), 4096);

        self.mcts_frames.extend_from_slice(mcts);
        self.results.push(result);
    }

    /// Gets a reference to the MCTS frames of this batch.
    pub fn get_mcts_frames(&self) -> &Vec<f32> {
        &self.mcts_frames
    }

    /// Gets a reference to the results of this batch.
    pub fn get_results(&self) -> &Vec<f32> {
        &self.results
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// Tests the TrainBatch can be initialized.
    #[test]
    fn train_batch_can_init() {
        TrainBatch::new();
    }

    /// Tests positions can be added to the trainbatch.
    #[test]
    fn train_batch_can_add() {
        let mut b = TrainBatch::new();

        b.add(&[0.0; 4096], 0.0);
        b.add(&[1.0; 4096], 1.0);
    }

    /// Tests the correct MCTS data is returned from the trainbatch.
    #[test]
    fn train_batch_get_mcts() {
        let mut b = TrainBatch::new();

        b.add(&[0.0; 4096], 0.0);
        b.add(&[0.0; 4096], 1.0);
        b.add(&[0.0; 4096], 1.0);

        assert_eq!(b.get_mcts_frames(), &[0.0; 4096 * 3]);
    }

    /// Tests the correct MCTS data is returned from the trainbatch.
    #[test]
    fn train_batch_get_results() {
        let mut b = TrainBatch::new();

        b.add(&[1.0; 4096], 1.0);
        b.add(&[0.0; 4096], 1.0);
        b.add(&[1.0; 4096], 1.0);

        assert_eq!(b.get_results(), &[1.0; 3]);
    }

    /// Tests that invalid MCTS frames cause a panic.
    #[test]
    #[should_panic]
    fn train_batch_invalid_mcts_small() {
        let mut b = TrainBatch::new();

        b.add(&[1.0; 128], 1.0);
    }

    /// Tests that invalid MCTS frames cause a panic.
    #[test]
    #[should_panic]
    fn train_batch_invalid_mcts_large() {
        let mut b = TrainBatch::new();

        b.add(&[1.0; 8192], 1.0);
    }
}