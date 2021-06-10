/// Contains structures used in model training.
use crate::model;

/// Manages a batch of training data.
pub struct TrainBatch {
    mcts: Vec<f32>,
    lmm: Vec<f32>,
    frames: Vec<f32>,
    headers: Vec<f32>,
    results: Vec<f32>,
    pov: Vec<f32>,
}

impl TrainBatch {
    /// Returns a new empty instance of TrainBatch.
    pub fn new() -> Self {
        TrainBatch {
            mcts: Vec::new(),
            lmm: Vec::new(),
            frames: Vec::new(),
            headers: Vec::new(),
            results: Vec::new(),
            pov: Vec::new(),
        }
    }

    /// Adds a position line to the training batch.
    /// Panics if the training line is invalid.
    pub fn add_from_line(&mut self, line: String, result: f32, pov: f32) {
        let floats: Vec<f32> = line.split(' ').skip(1).map(|x| x.parse::<f32>().unwrap()).collect();
        
        let lmm = &floats[0..4096];
        let mcts = &floats[4096..8192];
        let frames = &floats[8192..8192 + model::FRAMES_SIZE];
        let headers = &floats[8192 + model::FRAMES_SIZE..];

        self.add(lmm, mcts, frames, headers, pov, result);
    }

    pub fn add(&mut self, lmm: &[f32], mcts: &[f32], frames: &[f32], headers: &[f32], pov: f32, result: f32) {
        assert_eq!(lmm.len(), 4096);
        assert_eq!(mcts.len(), 4096);
        assert_eq!(frames.len(), model::FRAMES_SIZE);
        assert_eq!(headers.len(), model::SQUARE_HEADER_SIZE);

        self.lmm.extend_from_slice(lmm);
        self.mcts.extend_from_slice(mcts);
        self.frames.extend_from_slice(frames);
        self.headers.extend_from_slice(headers);
        self.pov.push(pov);
        self.results.push(result);
    }

    /// Gets a reference to the MCTS frames of this batch.
    pub fn get_mcts(&self) -> &Vec<f32> {
        &self.mcts
    }

    /// Gets a reference to the LMM of this batch.
    pub fn get_lmm(&self) -> &Vec<f32> {
        &self.lmm
    }

    /// Gets a reference to the frames of this batch.
    pub fn get_frames(&self) -> &Vec<f32> {
        &self.frames
    }

    /// Gets a reference to the headers of this batch.
    pub fn get_headers(&self) -> &Vec<f32> {
        &self.headers
    }

    /// Gets a reference to the povs of this batch.
    pub fn get_pov(&self) -> &Vec<f32> {
        &self.pov
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

        b.add(&[0.0; 4096], &[0.0; 4096], &[0.0; model::FRAMES_SIZE], &[0.0; model::SQUARE_HEADER_SIZE], 0.0, 0.0);
    }

    /// Tests the correct MCTS data is returned from the trainbatch.
    #[test]
    fn train_batch_get_mcts() {
        let mut b = TrainBatch::new();

        b.add(&[0.0; 4096], &[0.0; 4096], &[0.0; model::FRAMES_SIZE], &[0.0; model::SQUARE_HEADER_SIZE], 0.0, 0.0);
        b.add(&[0.0; 4096], &[0.0; 4096], &[0.0; model::FRAMES_SIZE], &[0.0; model::SQUARE_HEADER_SIZE], 0.0, 0.0);
        b.add(&[0.0; 4096], &[0.0; 4096], &[0.0; model::FRAMES_SIZE], &[0.0; model::SQUARE_HEADER_SIZE], 0.0, 0.0);

        assert_eq!(b.get_mcts(), &[0.0; 4096 * 3]);
    }

    /// Tests the correct MCTS data is returned from the trainbatch.
    #[test]
    fn train_batch_get_results() {
        let mut b = TrainBatch::new();

        b.add(&[0.0; 4096], &[0.0; 4096], &[0.0; model::FRAMES_SIZE], &[0.0; model::SQUARE_HEADER_SIZE], 0.0, 1.0);
        b.add(&[0.0; 4096], &[0.0; 4096], &[0.0; model::FRAMES_SIZE], &[0.0; model::SQUARE_HEADER_SIZE], 0.0, 0.0);
        b.add(&[0.0; 4096], &[0.0; 4096], &[0.0; model::FRAMES_SIZE], &[0.0; model::SQUARE_HEADER_SIZE], 0.0, 1.0);

        assert_eq!(b.get_results(), &[1.0, 0.0, 1.0]);
    }

    /// Tests the correct frames data is returned from the trainbatch.
    #[test]
    fn train_batch_get_frames() {
        let mut b = TrainBatch::new();

        b.add(&[0.0; 4096], &[0.0; 4096], &[0.0; model::FRAMES_SIZE], &[0.0; model::SQUARE_HEADER_SIZE], 0.0, 1.0);
        b.add(&[0.0; 4096], &[0.0; 4096], &[0.0; model::FRAMES_SIZE], &[0.0; model::SQUARE_HEADER_SIZE], 0.0, 0.0);
        b.add(&[0.0; 4096], &[0.0; 4096], &[0.0; model::FRAMES_SIZE], &[0.0; model::SQUARE_HEADER_SIZE], 0.0, 1.0);

        assert_eq!(b.get_frames(), &[0.0; model::FRAMES_SIZE * 3]);
    }

    /// Tests the correct headers data is returned from the trainbatch.
    #[test]
    fn train_batch_get_headers() {
        let mut b = TrainBatch::new();

        b.add(&[0.0; 4096], &[0.0; 4096], &[0.0; model::FRAMES_SIZE], &[0.0; model::SQUARE_HEADER_SIZE], 0.0, 1.0);
        b.add(&[0.0; 4096], &[0.0; 4096], &[0.0; model::FRAMES_SIZE], &[0.0; model::SQUARE_HEADER_SIZE], 0.0, 0.0);
        b.add(&[0.0; 4096], &[0.0; 4096], &[0.0; model::FRAMES_SIZE], &[0.0; model::SQUARE_HEADER_SIZE], 0.0, 1.0);

        assert_eq!(b.get_headers(), &[0.0; model::SQUARE_HEADER_SIZE * 3]);
    }

    /// Tests the correct LMM data is returned from the trainbatch.
    #[test]
    fn train_batch_get_lmm() {
        let mut b = TrainBatch::new();

        b.add(&[0.0; 4096], &[0.0; 4096], &[0.0; model::FRAMES_SIZE], &[0.0; model::SQUARE_HEADER_SIZE], 0.0, 1.0);
        b.add(&[0.0; 4096], &[0.0; 4096], &[0.0; model::FRAMES_SIZE], &[0.0; model::SQUARE_HEADER_SIZE], 0.0, 0.0);
        b.add(&[0.0; 4096], &[0.0; 4096], &[0.0; model::FRAMES_SIZE], &[0.0; model::SQUARE_HEADER_SIZE], 0.0, 1.0);

        assert_eq!(b.get_lmm(), &[0.0; 4096 * 3]);
    }

    /// Tests the correct POV data is returned from the trainbatch.
    #[test]
    fn train_batch_get_pov() {
        let mut b = TrainBatch::new();

        b.add(&[0.0; 4096], &[0.0; 4096], &[0.0; model::FRAMES_SIZE], &[0.0; model::SQUARE_HEADER_SIZE], 0.0, 1.0);
        b.add(&[0.0; 4096], &[0.0; 4096], &[0.0; model::FRAMES_SIZE], &[0.0; model::SQUARE_HEADER_SIZE], 1.0, 0.0);
        b.add(&[0.0; 4096], &[0.0; 4096], &[0.0; model::FRAMES_SIZE], &[0.0; model::SQUARE_HEADER_SIZE], 0.0, 1.0);

        assert_eq!(b.get_pov(), &[0.0, 1.0, 0.0]);
    }

    /// Tests that invalid MCTS causes a panic.
    #[test]
    #[should_panic]
    fn train_batch_invalid_mcts() {
        let mut b = TrainBatch::new();

        b.add(&[0.0; 4096], &[0.0; 128], &[0.0; model::FRAMES_SIZE], &[0.0; model::SQUARE_HEADER_SIZE], 0.0, 1.0);
    }

    /// Tests that invalid LMM causes a panic.
    #[test]
    #[should_panic]
    fn train_batch_invalid_lmm() {
        let mut b = TrainBatch::new();

        b.add(&[0.0; 4098], &[0.0; 4096], &[0.0; model::FRAMES_SIZE], &[0.0; model::SQUARE_HEADER_SIZE], 0.0, 1.0);
    }

    /// Tests that invalid frames cause a panic.
    #[test]
    #[should_panic]
    fn train_batch_invalid_frames() {
        let mut b = TrainBatch::new();

        b.add(&[0.0; 4096], &[0.0; 4096], &[0.0; model::FRAMES_SIZE + 1], &[0.0; model::SQUARE_HEADER_SIZE], 0.0, 1.0);
    }

    /// Tests that invalid headers cause a panic.
    #[test]
    #[should_panic]
    fn train_batch_invalid_headers() {
        let mut b = TrainBatch::new();

        b.add(&[0.0; 4096], &[0.0; 4096], &[0.0; model::FRAMES_SIZE], &[0.0; model::SQUARE_HEADER_SIZE - 12], 0.0, 1.0);
    }
}
