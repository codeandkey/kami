/// Interface for NN model generation, execution, and training.
pub mod mock;

use crate::input::{batch::Batch, trainbatch::TrainBatch};
use mock::MockModel;

use std::error::Error;
use std::fs::{self, File};
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
pub trait Model: Send + Sync + 'static {
    /// Reads an existing model from a directory on the disk.
    fn read(p: &Path) -> Result<Self, Box<dyn Error>>
    where
        Self: Sized;

    /// Evaluates a batch of inputs and produces a batch of outputs.
    fn execute(&self, b: &Batch) -> Output;

    /// Generates a new model.
    fn generate() -> Result<Self, Box<dyn Error>>
    where
        Self: Sized;

    /// Trains the model on a set of training batches.
    fn train(&mut self, batches: Vec<TrainBatch>);

    /// Saves the model data to a directory.
    fn write(&self, p: &Path) -> Result<(), Box<dyn Error>>;

    /// Gets the type of model in string format.
    fn get_type(&self) -> &'static str;
}

/// Threadsafe pointer to a model.
pub type ModelPtr = Arc<RwLock<dyn Model + Send + Sync + 'static>>;

/// Creates a threadsafe model pointer.
pub fn make_ptr<T: Model + Send + Sync + 'static>(m: T) -> ModelPtr {
    Arc::new(RwLock::new(m))
}

/// Loads a generic model from the disk.
/// Tries to figure out the type of the saved model,
/// and returns a threadsafe pointer to the loaded model.
pub fn load(p: &Path) -> Result<Option<ModelPtr>, Box<dyn Error>> {
    if !p.exists() {
        return Ok(None);
    }

    let out;

    // Check for mock model type.
    if p.join("mock.type").exists() {
        out = Some(make_ptr(MockModel::read(p)?));
    } else {
        return Err("Couldn't detect model type!".into());
    }

    println!(
        "Loaded [{}] model from {}",
        out.as_ref().unwrap().read().unwrap().get_type(),
        p.display()
    );
    Ok(out)
}

/// Writes a model to a directory.
/// Creates the destination directory if it does not exist.
/// Creates a type marker to indicate the type of model that is saved.
pub fn save(md: &ModelPtr, p: &Path) -> Result<(), Box<dyn Error>> {
    if p.exists() && !p.is_dir() {
        return Err(format!("Refusing to destroy non-directory model {}", p.display()).into());
    }

    fs::create_dir_all(p)?;
    File::create(p.join(format!("{}.type", md.read().unwrap().get_type())))?;
    md.read().unwrap().write(p)?;

    println!(
        "Wrote [{}] model to {}",
        md.read().unwrap().get_type(),
        p.display()
    );

    Ok(())
}

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

    /// Tests that the mock model type can be loaded.
    #[test]
    fn mock_model_can_load() {
        let path = tempfile::tempdir()
            .expect("failed to gen tempdir")
            .into_path();

        save(
            &make_ptr(MockModel::generate().expect("model gen failed")),
            &path,
        )
        .expect("write failed");

        let loaded = load(&path).expect("load failed").expect("missing model");

        assert_eq!(loaded.read().unwrap().get_type(), "mock".to_string());
    }
}
