/// Interface for NN model generation, execution, and training.

use crate::batch::Batch;
use crate::train::TrainBatch;

use std::error::Error;
use std::path::Path;
use std::sync::Arc;

pub const PLY_FRAME_SIZE: usize = 14;
pub const PLY_FRAME_COUNT: usize = 6;
pub const SQUARE_HEADER_SIZE: usize = 18; // 8 bits move number, 6 bits halfmove clock, 4 bits castling rights

/// Stores outputs from a model.
pub struct Output {
    policy: Vec<f32>,
    value: Vec<f32>,
}

impl Output {
    pub fn new(policy: Vec<f32>, value: Vec<f32>) -> Self {
        assert_eq!(policy.len() / 4096, value.len());

        Output {
            policy: policy,
            value: value,
        }
    }

    pub fn dummy(bsize: usize) -> Self {
        let mut policy = Vec::new();
        let mut value = Vec::new();

        policy.resize(4096 * bsize, 1.0 / 4096.0);
        value.resize(bsize, 0.0);

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
    fn new(p: Option<&Path>) -> ModelPtr where Self: Sized;

    /// Evaluates a batch of inputs and produces a batch of outputs.
    fn execute(&self, b: &Batch) -> Output;

    /// Trains the model on a training batch.
    fn train(&mut self, b: &TrainBatch);

    /// Writes a model to a path.
    fn write(&self, p: &Path);
}

pub type ModelPtr = Arc<dyn Model + Send + Sync>;