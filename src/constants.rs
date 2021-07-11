/// Several configuration constants.

use crate::model;

pub const TRAINING_SET_SIZE: usize = 4; // number of games played per generation
pub const SEARCH_TIME: usize = 10000; // milliseconds per move (if node target not reached)
pub const SEARCH_MAXNODES: usize = 8000; // maximum nodes searched per move
pub const SEARCH_STATUS_RATE: u64 = 100; // milliseconds between search status reports
pub const TEMPERATURE: f64 = 1.0; // MCTS initial temperature
pub const TEMPERATURE_DROPOFF: f64 = 0.1; // MCTS final temperature
pub const TEMPERATURE_DROPOFF_PLY: usize = 1000; // ply to switch from initial to dropoff temperature
pub const SEARCH_BATCH_SIZE: usize = 16; // number of nodes to expand at once on a single thread
pub const TRAINING_BATCH_SIZE: usize = 32; // number of decisions in each training batch
pub const TRAINING_BATCH_COUNT: usize = 32; // number of training batches
pub const TUI_FRAME_RATE: u64 = 15; // TUI framerate (frames/second)
pub const MOVETIME_ELO: usize = 2000; // MS per move during ELO evaluation
pub const ELO_EVALUATION_NUM_GAMES: usize = 10;
pub const STOCKFISH_ELO: [usize; ELO_EVALUATION_NUM_GAMES] = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]; // stockfish elo settings for ELO evaluation
pub const ELO_EVALUATION_INTERVAL: usize = 25; // Perform ELO evaluation every n generations

#[cfg(feature = "tch")]
pub const DEFAULT_MODEL_TYPE: model::Type = model::Type::Torch; // model type to initialize

#[cfg(not(feature = "tch"))]
pub const DEFAULT_MODEL_TYPE: model::Type = model::Type::Torch; // model type to initialize