use crate::batch::Batch;

use std::error::Error;
use std::path::Path;

use tensorflow::{Graph, SavedModelBundle, SessionOptions, Operation, SessionRunArgs, Session};

pub const PLY_FRAME_SIZE: usize = 14;
pub const PLY_FRAME_COUNT: usize = 6;
pub const SQUARE_HEADER_SIZE: usize = 24;
pub const SQUARE_BITS: usize = SQUARE_HEADER_SIZE + PLY_FRAME_SIZE * PLY_FRAME_COUNT;
pub const COUNTER_BITS: usize = 20;

pub struct Output {
    policy: Vec<f32>,
    value: Vec<f32>,
}

impl Output {
    pub fn get_policy(&self, idx: usize) -> &[f32] {
        &self.policy[idx * 4096 .. (idx + 1) * 4096]
    }

    pub fn get_value(&self, idx: usize) -> f32 {
        self.value[idx]
    }
}

pub struct Model {
    session: Session,
    graph: Graph,
    op_serve: Operation,
}

impl Model {
    pub fn load(p: &Path, opts: SessionOptions) -> Result<Model, Box<dyn Error>> {
        let mut g = Graph::new();

        debug!("Loading model from {}", p.display());

        let session = SavedModelBundle::load(
            &opts,
            &["serve"],
            &mut g,
            p,
        )?.session;

        debug!("Model ready.");
        debug!("Using tensorflow version {}", tensorflow::version().unwrap());

        debug!("Enumerating devices:");

        let devices = session.device_list()?;

        for c in &devices {
            debug!(
                ">> {} {}: {:.1}GB",
                c.device_type,
                c.name,
                c.memory_bytes as f64 / ((1u64 << 30) as f64)
            );
        }

        debug!("{} total devices available", devices.len());

        Ok(Model {
            session: session,
            graph: g,
            op_serve: g.operation_by_name_required("serve")?,
        })
    }

    pub fn execute(&mut self, b: &Batch) -> Result<Output, Box<dyn Error>> {
        let mut args = SessionRunArgs::new();

        let header_tensor = b.get_header_tensor();
        let lmm_tensor = b.get_lmm_tensor();

        args.add_feed(&self.op_serve, 0, &header_tensor);
        args.add_feed(&self.op_serve, 1, &lmm_tensor);

        b.get_frame_tensors().iter().enumerate().for_each(|(i, t)| {
            args.add_feed(&self.op_serve, (2 + i) as i32, &t);
        });

        let policy_req = args.request_fetch(&self.op_serve, 0);
        let value_req = args.request_fetch(&self.op_serve, 1);

        self.session.run(&mut args)?;

        let policy_tensor = args.fetch::<f32>(policy_req)?;
        let value_tensor = args.fetch::<f32>(value_req)?;
        
        Ok(Output {
            policy: policy_tensor.to_vec(),
            value: value_tensor.to_vec(),
        })
    }
}
