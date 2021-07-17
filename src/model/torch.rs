/// Torch model implementation.
use crate::input::{batch::Batch, trainbatch::TrainBatch};

use crate::model::*;

use std::error::Error;
use std::fs::File;
use std::io::{Write, BufReader, BufRead};
use std::path::Path;

use tch::{CModule, Device, IValue, Tensor};

pub struct Model {
    cmod: CModule,
    cuda: bool,
}

/// Generates a new model on the disk.
pub fn generate(p: &Path) -> Result<(), Box<dyn Error>> {
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

/// Loads a model from a path.
pub fn load(p: &Path, quiet: bool) -> Result<Model, Box<dyn Error>> {
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

    return Ok(Model {
        cmod: cmod,
        cuda: tch::Cuda::is_available(),
    });
}

/// Executes a batch on a model.
pub fn execute(b: &Batch, tmod: &Model) -> Output {
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

    if tmod.cuda {
        headers_tensor = headers_tensor.to(Device::Cuda(0));
        frames_tensor = frames_tensor.to(Device::Cuda(0));
        lmm_tensor = lmm_tensor.to(Device::Cuda(0));
    }

    let headers_tensor = IValue::Tensor(headers_tensor);
    let frames_tensor = IValue::Tensor(frames_tensor);
    let lmm_tensor = IValue::Tensor(lmm_tensor);

    let nn_result = tch::no_grad(|| {
        tmod
            .cmod
            .forward_is(&[headers_tensor, frames_tensor, lmm_tensor])
            .expect("network eval failed")
    });

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

/// Trains a model at a path.
/// Returns the entire contents of stdout.
pub fn train<F>(p: &Path, tb: Vec<TrainBatch>, sout: F, loss_path: &Path) -> Result<Vec<String>, Box<dyn Error>> 
    where F: Fn(&String)
{
    let tdata_path = p.join("train_data.json");
    let mut tdata = File::create(&tdata_path)?;

    tdata.write(serde_json::to_string(&tb)?.as_bytes())?;
    tdata.flush()?;

    let mut output = std::process::Command::new("python")
        .stdout(std::process::Stdio::piped())
        .args(&[
            "scripts/train_torch.py",
            p.join("model.pt").to_str().unwrap(),
            tdata_path
                .to_str()
                .expect("failed to get str from tdata pathbuf"),
            loss_path
                .to_str()
                .expect("failed to get str from loss pathbuf"),
        ])
        .spawn()?;

    let stdout_reader = BufReader::new(output.stdout.take().unwrap());
    let mut stdout_lines = Vec::new();

    for line in stdout_reader.lines() {
        match line {
            Ok(line) => {
                sout(&line);
                stdout_lines.push(line);
            },
            Err(e) => sout(format!("sout err: {}", e)),
        }
    }

    let output = output.wait()?;

    if !output.success() {
        return Err("Torch init script returned failure status.".into());
    }

    return Ok(stdout_lines);
}