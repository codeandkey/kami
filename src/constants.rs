/// Several configuration constants for the search.

pub const TRAINING_SET_SIZE: usize = 8; // number of games played per generation
pub const SEARCH_TIME: usize = 2500; // milliseconds per move
pub const TEMPERATURE: f64 = 1.0; // MCTS initial temperature
pub const TEMPERATURE_DROPOFF: f64 = 0.1; // MCTS final temperature
pub const TEMPERATURE_DROPOFF_PLY: usize = 25; // ply to switch from initial to dropoff temperature
pub const SEARCH_BATCH_SIZE: usize = 16; // number of nodes to expand at once on a single thread
pub const TRAINING_BATCH_SIZE: usize = 16; // number of decisions in each training batch
pub const TRAINING_BATCH_COUNT: usize = 32; // number of training batches
