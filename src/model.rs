/// Interface for NN model generation, execution, and training.
use crate::input::{batch::Batch, trainbatch::TrainBatch};

use rand::{prelude::*, thread_rng};
use std::error::Error;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "tch")]
use tch::{CModule, Device, IValue, Tensor};

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
    Torch(CModule, bool),
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
        Model::Torch(_, _) => Type::Torch,
    }
}

/// Generates a new model on the disk.
pub fn generate(p: &Path, nt: Type) -> Result<(), Box<dyn Error>> {
    match nt {
        Type::Mock => {
            if p.exists() && !p.is_dir() {
                return Err(
                    "Will not generate mock model, destination exists and is not a dir".into(),
                );
            }

            fs::create_dir_all(p)?;
            File::create(p.join("mock.type"))?;
            Ok(())
        }

        #[cfg(feature = "tch")]
        Type::Torch => {
            if p.exists() && !p.is_dir() {
                return Err(
                    "Will not generate torch model, destination exists and is not a dir".into(),
                );
            }

            fs::create_dir_all(p)?;
            File::create(p.join("torch.type"))?;

            let status = std::process::Command::new("python")
                .args(&[
                    "scripts/init_torch.py",
                    p.join("model.pt").to_str().unwrap(),
                ])
                .spawn()?
                .wait()?
                .success();

            if !status {
                return Err("Torch init script returned failure status.".into());
            }

            println!("Torch init script returned success.");
            Ok(())
        }
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
        // Load torch model.
        if !quiet {
            println!("Detected model type torch");
        }

        let mut cmod = CModule::load(p.join("model.pt"))?;

        tch::maybe_init_cuda();

        if tch::Cuda::is_available() {
            if !quiet {
                println!("CUDA support enabled");
                println!("{} CUDA devices available", tch::Cuda::device_count());

                if tch::Cuda::cudnn_is_available() {
                    println!("CUDNN acceleration enabled");
                }
            }

            cmod.to(Device::Cuda(0), tch::Kind::Float, false);
        } else {
            if !quiet {
                println!("CUDA is not available, using CPU for evaluation");
            }
        }

        return Ok(Model::Torch(cmod, tch::Cuda::is_available()));
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
        Model::Torch(cmod, cuda) => {
            let mut headers_tensor = Tensor::of_slice(b.get_headers())
                .reshape(&[b.get_size() as i64, SQUARE_HEADER_SIZE as i64]);

            let mut frames_tensor = Tensor::of_slice(b.get_frames()).reshape(&[
                b.get_size() as i64,
                8,
                8,
                (PLY_FRAME_COUNT * PLY_FRAME_SIZE) as i64,
            ]);

            let mut lmm_tensor =
                Tensor::of_slice(b.get_lmm()).reshape(&[b.get_size() as i64, 4096]);

            if *cuda {
                headers_tensor = headers_tensor.to(Device::Cuda(0));
                frames_tensor = frames_tensor.to(Device::Cuda(0));
                lmm_tensor = lmm_tensor.to(Device::Cuda(0));
            }

            let headers_tensor = IValue::Tensor(headers_tensor);
            let frames_tensor = IValue::Tensor(frames_tensor);
            let lmm_tensor = IValue::Tensor(lmm_tensor);

            let nn_result = cmod
                .forward_is(&[headers_tensor, frames_tensor, lmm_tensor])
                .expect("network eval failed");

            let (policy, value): (Vec<f64>, Vec<f32>) = match nn_result {
                IValue::Tuple(v) => match &v[..] {
                    [IValue::Tensor(policy), IValue::Tensor(value)] => {
                        (policy.into(), value.into())
                    }
                    _ => panic!("Unexpected network output tuple size"),
                },
                _ => panic!("Unexpected network output type"),
            };

            assert_eq!(policy.len(), b.get_size() * 4096);
            assert_eq!(value.len(), b.get_size());

            Output::new(policy, value)
        }
    }
}

/// Trains a model at a path.
pub fn train(p: &Path, tb: Vec<TrainBatch>, mtype: Type) -> Result<(), Box<dyn Error>> {
    match mtype {
        Type::Mock => (),

        #[cfg(feature = "tch")]
        Type::Torch => {
            let tdata_path = p.join("train_data.json");
            let mut tdata = File::create(&tdata_path)?;

            tdata.write(serde_json::to_string(&tb)?.as_bytes())?;
            tdata.flush()?;

            let output = std::process::Command::new("python")
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .args(&[
                    "scripts/train_torch.py",
                    p.join("model.pt").to_str().unwrap(),
                    tdata_path
                        .to_str()
                        .expect("failed to get str from tdata pathbuf"),
                ])
                .spawn()?
                .wait_with_output()?;

            let mut tlog = File::create(&p.join("train.stdout"))?;
            tlog.write(&output.stdout)?;

            let mut terr = File::create(&p.join("train.stderr"))?;
            terr.write(&output.stderr)?;

            if !output.status.success() {
                return Err("Torch init script returned failure status.".into());
            }
        }
    }

    Ok(())
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
        train(&dst, vec![TrainBatch::new(1)], Type::Mock).expect("train failed");
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
}
