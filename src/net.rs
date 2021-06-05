use crate::batch::Batch;

use std::convert::TryInto;
use std::error::Error;
use std::path::Path;

use tensorflow::{Graph, Operation, SavedModelBundle, Session, SessionOptions, SessionRunArgs};

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

        policy.resize_with(4096 * bsize, || 1.0 / 4096.0);
        value.resize_with(bsize, || 0.0);

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
    session: Session,
    graph: Graph,
    op_serve: Operation,
}

impl Model {
    pub fn load(p: &Path, opts: SessionOptions) -> Result<Model, Box<dyn Error>> {
        let mut g = Graph::new();

        debug!("Loading model from {}", p.display());

        let session = SavedModelBundle::load(&opts, &["serve"], &mut g, p)?.session;

        debug!("Model ready.");

        debug!("Available operations:");
        for op in g.operation_iter() {
            debug!("> {}", op.name()?);
        }

        debug!(
            "Using tensorflow version {}",
            tensorflow::version().unwrap()
        );

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
            op_serve: g.operation_by_name_required("serving_default_input_1")?,
            graph: g,
        })
    }

    pub fn execute(&self, b: &Batch) -> Result<Output, Box<dyn Error>> {
        let header_tensor = b.get_header_tensor();
        let lmm_tensor = b.get_lmm_tensor();
        let frames_tensor = b.get_frames_tensor();

        let mut args = SessionRunArgs::new();

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
        })
    }
}
