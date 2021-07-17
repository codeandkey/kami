/// Interface for NN model generation, execution, and training.
use crate::input::{batch::Batch, trainbatch::TrainBatch};

use rand::{prelude::*, thread_rng};
use std::error::Error;
use std::fs::{self, File};
use std::path::Path;

#[cfg(feature = "tch")]
mod torch;

/// Model input layer constants (TODO move to input)
pub const PLY_FRAME_SIZE: usize = 14;
pub const PLY_FRAME_COUNT: usize = 6;
pub const SQUARE_HEADER_SIZE: usize = 18; // 8 bits move number, 6 bits halfmove clock, 4 bits castling rights
pub const FRAMES_SIZE: usize = 64 * PLY_FRAME_SIZE * PLY_FRAME_COUNT;

/// Stores outputs from a model.
pub struct Output {
    policy: Vec<f64>,
    value: Vec<f32>,
}

impl Output {
    pub fn new(policy: Vec<f64>, value: Vec<f32>) -> Self {
        assert_eq!(policy.len() / 4096, value.len());
        assert_eq!(policy.len() % 4096, 0);

        //println!("policy: {:?}, value: {:?}", &policy, &value);

        Output {
            policy: policy,
            value: value,
        }
    }

    pub fn get_policy(&self, idx: usize) -> &[f64] {
        &self.policy[idx * 4096..(idx + 1) * 4096]
    }

    pub fn get_value(&self, idx: usize) -> f32 {
        self.value[idx]
    }
}

/// Stores a model in memory.
pub enum Model {
    Mock,

    #[cfg(feature = "tch")]
    Torch(torch::Model),
}

/// Types used for generating new models.
pub enum Type {
    Mock,

    #[cfg(feature = "tch")]
    Torch,
}

pub fn get_type(m: &Model) -> Type {
    match m {
        Model::Mock => Type::Mock,

        #[cfg(feature = "tch")]
        Model::Torch(_) => Type::Torch,
    }
}

/// Generates a new model on the disk.
pub fn generate(p: &Path, nt: Type) -> Result<(), Box<dyn Error>> {
    if p.exists() && !p.is_dir() {
        return Err(
            "Will not generate model, destination exists and is not a dir".into(),
        );
    }

    fs::create_dir_all(p)?;

    match nt {
        Type::Mock => {
            File::create(p.join("mock.type"))?;
            Ok(())
        }

        #[cfg(feature = "tch")]
        Type::Torch => torch::generate(p),
    }
}

/// Loads a model from a path.
pub fn load(p: &Path, quiet: bool) -> Result<Model, Box<dyn Error>> {
    // Test if no model exists.
    if !p.exists() {
        return Err("Path does not exist".into());
    }

    // Test if the model path is corrupted
    if !p.is_dir() {
        return Err("Path exists, but is not a directory..".into());
    }

    // Find the model type.
    if !quiet {
        println!("Loading model from '{}'.", p.display());
    }

    if p.join("mock.type").exists() {
        if !quiet {
            println!("Detected model type mock");
        }

        return Ok(Model::Mock);
    }

    #[cfg(feature = "tch")]
    if p.join("torch.type").exists() {
        return Ok(Model::Torch(torch::load(p, quiet)?));
    }

    return Err(
        "Could not detect model type. Make sure the model type is enabled in build features."
            .into(),
    );
}

/// Executes a batch on a model.
pub fn execute(m: &Model, b: &Batch) -> Output {
    match m {
        Model::Mock => {
            // Generate dummy output (random)
            let mut policy = Vec::new();
            let mut rng = thread_rng();
            policy.resize_with(b.get_size() * 4096, || {
                rng.next_u32() as f64 / u32::MAX as f64
            });

            let mut value = Vec::new();
            value.resize_with(b.get_size(), || {
                (rng.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
            });

            Output::new(policy, value)
        }

        #[cfg(feature = "tch")]
        Model::Torch(tmod) => torch::execute(b, &tmod),
    }
}

/// Trains a model at a path.
/// Returns the entire contents of stdout.
pub fn train<F>(p: &Path, tb: Vec<TrainBatch>, mtype: Type, sout: F, loss_path: &Path) -> Result<Vec<String>, Box<dyn Error>> 
    where F: Fn(&String) {

    match mtype {
        Type::Mock => Ok(Vec::new()),

        #[cfg(feature = "tch")]
        Type::Torch => torch::train(p, tb, sout, loss_path),
    }
}

/// Archives the current model.

#[cfg(test)]
mod test {
    use super::*;
    use crate::position::Position;
    use std::path::PathBuf;

    /// Creates a temp directory for testing.
    fn tdir() -> PathBuf {
        tempfile::tempdir()
            .expect("failed to gen tempdir")
            .into_path()
    }

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

    // ## Mock model tests ##

    /// Tests that the mock model can be generated.
    #[test]
    fn mock_model_can_generate() {
        let dst = tdir();

        generate(&dst, Type::Mock).expect("generate failed");
        assert!(dst.join("mock.type").is_file());
    }

    /// Tests that the mock model can be executed.
    #[test]
    fn mock_model_can_execute() {
        let mut b = Batch::new(4);
        b.add(&Position::new());

        let output = execute(&Model::Mock, &b);
        assert_eq!(output.get_policy(0).len(), 4096);
    }

    /// Tests that the mock model can be trained.
    #[test]
    fn mock_model_can_train() {
        let dst = tdir();

        generate(&dst, Type::Mock).expect("generate failed");
        train(&dst, vec![TrainBatch::new(1)], Type::Mock, |_| (), &dst.join("loss")).expect("train failed");
    }

    /// Tests that the mock model type can be loaded.
    #[test]
    fn mock_model_can_load() {
        let path = tdir();

        generate(&path, Type::Mock).expect("model gen failed");
        assert!(matches!(
            load(&path, false).expect("load failed"),
            Model::Mock
        ));
    }

    /// Tests that torch models can be generated.
    #[cfg(feature = "tch")]
    #[test]
    fn torch_model_can_generate() {
        let path = tdir();

        generate(&path, Type::Torch).expect("generate failed");
    }

    /// Tests that torch models can be loaded.
    #[cfg(feature = "tch")]
    #[test]
    fn torch_model_can_load() {
        let path = tdir();

        generate(&path, Type::Torch).expect("generate failed");

        let md = load(&path, false).expect("load failed");

        assert!(matches!(get_type(&md), Type::Torch));
    }
}