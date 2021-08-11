use serde::{Serialize, Deserialize};
use std::path::PathBuf;

/// Parameter structure, defines all tunable search options.
#[derive(Clone, Serialize, Deserialize)]
pub struct Params {
    pub search_nodes: usize,
    pub puct_policy_weight: f64,
    pub puct_noise_weight: f64,
    pub puct_noise_alpha: f64,
    pub batch_size: u8,
    pub model_path: PathBuf,
    pub num_threads: usize,
    pub temperature: f64,
}