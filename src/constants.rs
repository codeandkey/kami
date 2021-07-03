/// Several configuration constants for the search.

pub const TRAINING_SET_SIZE: usize = 8; // number of games played per generation
pub const SEARCH_TIME: usize = 10000; // milliseconds per move (if node target not reached)
pub const SEARCH_MAXNODES: usize = 12000; // maximum nodes searched per move
pub const SEARCH_STATUS_RATE: u64 = 100; // milliseconds between search status reports
pub const TEMPERATURE: f64 = 1.0; // MCTS initial temperature
pub const TEMPERATURE_DROPOFF: f64 = 0.1; // MCTS final temperature
pub const TEMPERATURE_DROPOFF_PLY: usize = 1000; // ply to switch from initial to dropoff temperature
pub const SEARCH_BATCH_SIZE: usize = 16; // number of nodes to expand at once on a single thread
pub const TRAINING_BATCH_SIZE: usize = 32; // number of decisions in each training batch
pub const TRAINING_BATCH_COUNT: usize = 32; // number of training batches
