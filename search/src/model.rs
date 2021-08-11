use crate::batch::{Batch, BatchResult};
use crate::consts;

use std::error::Error;
use std::path::Path;

use tch::{CModule, Device, IValue, Tensor};

pub struct Model {
    cmod: CModule,
    cuda: bool,
}

impl Model {
    /// Loads a model from a path.
    pub fn load(p: &Path) -> Result<Model, Box<dyn Error>> {
        let mut cmod = CModule::load(p)?;

        tch::maybe_init_cuda();

        if tch::Cuda::is_available() {
            println!("CUDA support enabled");
            println!("{} CUDA devices available", tch::Cuda::device_count());

            if tch::Cuda::cudnn_is_available() {
                println!("CUDNN acceleration enabled");
            }

            cmod.to(Device::Cuda(0), tch::Kind::Float, false);
        } else {
            println!("CUDA is not available, using CPU for inference");
        }

        return Ok(Model {
            cmod: cmod,
            cuda: tch::Cuda::is_available(),
        });
    }

    /// Executes a batch on a model.
    pub fn execute(&self, b: Batch) -> BatchResult {
        let mut headers_tensor = Tensor::of_slice(b.get_headers())
            .reshape(&[b.get_size() as i64, consts::HEADER_SIZE as i64]);

        let mut frames_tensor = Tensor::of_slice(b.get_frames()).reshape(&[
            b.get_size() as i64,
            consts::FRAME_COUNT as i64,
            8,
            8,
            consts::FRAME_SIZE as i64,
        ]);

        let mut lmm_tensor =
            Tensor::of_slice(b.get_lmm()).reshape(&[b.get_size() as i64, 4096]);

        if self.cuda {
            headers_tensor = headers_tensor.to(Device::Cuda(0));
            frames_tensor = frames_tensor.to(Device::Cuda(0));
            lmm_tensor = lmm_tensor.to(Device::Cuda(0));
        }

        let headers_tensor = IValue::Tensor(headers_tensor);
        let frames_tensor = IValue::Tensor(frames_tensor);
        let lmm_tensor = IValue::Tensor(lmm_tensor);

        let nn_result = tch::no_grad(|| {
            self
                .cmod
                .forward_is(&[headers_tensor, frames_tensor, lmm_tensor])
                .expect("network eval failed")
        });

        let (policy, value): (Vec<f64>, Vec<f64>) = match nn_result {
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

        b.into_result(policy, value)
    }
}