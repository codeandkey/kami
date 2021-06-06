use crate::batch::Batch;

use std::error::Error;
use std::path::Path;

use tch::{CModule, Device, Cuda};

pub const PLY_FRAME_SIZE: usize = 14;
pub const PLY_FRAME_COUNT: usize = 6;
pub const SQUARE_HEADER_SIZE: usize = 18; // 8 bits move number, 6 bits halfmove clock, 4 bits castling rights

pub struct Output {
    policy: Vec<f32>,
    value: Vec<f32>,
}

impl Output {
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

pub struct Model {
    module: CModule,
    device: Device,
}

impl Model {
    pub fn load(p: &Path) -> Result<Model, Box<dyn Error>> {
        debug!("Loading model from {}", p.display());

        let module = CModule::load(p)?;

        debug!("Model ready.");

        let device = Device::cuda_if_available();

        if device.is_cuda() {
            println!("CUDA support enabled.");
            println!("Using {} GPUs for computing.", Cuda::device_count());
            println!("CUDNN support: {}", if Cuda::cudnn_is_available() { "yes" } else { "no" });
        } else {
            println!("CUDA not available, running on CPU.");
        }

        Ok(Model {
            module: module,
            device: device,
        })
    }

    pub fn execute(&self, b: &Batch) -> Result<Output, Box<dyn Error>> {
        let header_tensor = b.get_header_tensor();
        let lmm_tensor = b.get_lmm_tensor();
        let frames_tensor = b.get_frames_tensor();

        // TODO: implement network evaluation
        return Ok(Output::dummy(b.get_size()));

        /*let mut args = SessionRunArgs::new();

        args.add_feed(&self.op_serve, 0, &header_tensor);
        args.add_feed(&self.op_serve, 1, &frames_tensor);
        args.add_feed(&self.op_serve, 2, &lmm_tensor);

        let policy_req = args.request_fetch(&self.op_serve, 0);
        let value_req = args.request_fetch(&self.op_serve, 1);

        self.session.run(&mut args)?;

        let policy_tensor = args.fetch::<f32>(policy_req)?;
        let value_tensor = args.fetch::<f32>(value_req)?;

        Ok(Output {
            policy: policy_tensor.to_vec(),
            value: value_tensor.to_vec(),
        })*/
    }
}
