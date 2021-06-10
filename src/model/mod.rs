/// Interface for NN model generation, execution, and training.
pub mod mock;

use crate::input::{batch::Batch, trainbatch::TrainBatch};

use std::io;
use std::path::Path;
use std::sync::{Arc, RwLock};

pub const PLY_FRAME_SIZE: usize = 14;
pub const PLY_FRAME_COUNT: usize = 6;
pub const SQUARE_HEADER_SIZE: usize = 18; // 8 bits move number, 6 bits halfmove clock, 4 bits castling rights
pub const FRAMES_SIZE: usize = 64 * PLY_FRAME_SIZE * PLY_FRAME_COUNT;

/// Stores outputs from a model.
pub struct Output {
    policy: Vec<f32>,
    value: Vec<f32>,
}

impl Output {
    pub fn new(policy: Vec<f32>, value: Vec<f32>) -> Self {
        assert_eq!(policy.len() / 4096, value.len());
        assert_eq!(policy.len() % 4096, 0);

        Output {
            policy: policy,
            value: value,
        }
    }

    pub fn get_policy(&self, idx: usize) -> &[f32] {
        &self.policy[idx * 4096..(idx + 1) * 4096]
    }

    pub fn get_value(&self, idx: usize) -> f32 {
        self.value[idx]
    }
}

/// Generic model trait.
pub trait Model {
    /// Returns a threadsafe instance of this model.
    fn new(p: &Path) -> ModelPtr
    where
        Self: Sized;

    /// Evaluates a batch of inputs and produces a batch of outputs.
    fn execute(&self, b: &Batch) -> Output;

    /// Trains the model on a set of training batches.
    fn train(&mut self, batches: Vec<TrainBatch>);

    /// Writes a model to a path.
    fn write(&self, p: &Path) -> Result<(), io::Error>;
}

pub type ModelPtr = Arc<RwLock<dyn Model + Send + Sync>>;

#[cfg(test)]
mod test {
    use super::*;

    /// Tests that a blank output can be initialized
    #[test]
    fn output_can_initialize_empty() {
        Output::new(Vec::new(), Vec::new());
    }

    /// Tests that an output with 1 frame can be initialized
    #[test]
    fn output_can_initialize_single() {
        Output::new(vec![1.0 / 4096.0; 4096], vec![1.0]);
    }

    /// Tests that an output with an invalid frame cannot be initialized
    #[test]
    #[should_panic]
    fn output_invalid_initialize() {
        Output::new(vec![1.0 / 4096.0; 4098], vec![1.0, 2.0]);
    }

    /// Tests that an output with an invalid frame cannot be initialized (when integer truncation applies)
    #[test]
    #[should_panic]
    fn output_invalid_initialize_second() {
        Output::new(vec![1.0 / 4096.0; 4098], vec![1.0]);
    }
}
