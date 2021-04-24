use std::error::Error;
use std::path::Path;

use tensorflow::{Graph, SavedModelBundle, SessionOptions};

pub struct Model {
    tf_bundle: SavedModelBundle,
    graph: Graph,
}

impl Model {
    pub fn load(p: &Path, opts: SessionOptions) -> Result<Model, Box<dyn Error>> {
        let mut g = Graph::new();

        debug!("Loading model from {}", p.display());

        let md = SavedModelBundle::load(
            &opts,
            &["serve"],
            &mut g,
            p,
        )?;

        debug!("Model ready.");
        debug!("Using tensorflow version {}", tensorflow::version().unwrap());

        debug!("Enumerating devices:");

        let devices = md.session.device_list()?;

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
            tf_bundle: md,
            graph: g
        })
    }
}
