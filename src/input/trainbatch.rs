/// Represents a batch for training the network.
/// This batch contains normal inputs to the network,
/// as well as the actual MCTS counts from the search and the results of the games.
use crate::constants;
use crate::game::Game;
use crate::input::batch::Batch;
use crate::position::Position;

use rand::{prelude::*, thread_rng};
use serde::Serialize;
use std::error::Error;
use std::path::Path;

#[derive(Serialize)]
pub struct TrainBatch {
    results: Vec<f32>,
    mcts: Vec<f64>,
    inner: Batch,
}

impl TrainBatch {
    /// Returns a new batch instance with <reserve_size> preallocated space.
    pub fn new(reserve_size: usize) -> Self {
        let mut mcts = Vec::new();
        let mut results = Vec::new();

        mcts.reserve(reserve_size);
        results.reserve(reserve_size);

        TrainBatch {
            inner: Batch::new(reserve_size),
            mcts: mcts,
            results: results,
        }
    }

    /// Generates a trainbatch from the disk.
    pub fn generate(games_dir: &Path) -> Result<TrainBatch, Box<dyn Error>> {
        let mut saved_games: Vec<Game> = Vec::new();

        for i in 0..constants::TRAINING_SET_SIZE {
            let game_path = games_dir.join(format!("{}.game", i));

            // Parse the game.
            let gm = Game::load(&game_path)?;

            // Check the game was completed.
            if !gm.is_complete() {
                return Err("Training set contains incomplete games!".into());
            }

            saved_games.push(gm);
        }

        let mut rng = thread_rng();
        let mut tb = TrainBatch::new(constants::TRAINING_BATCH_SIZE);

        for _ in 0..constants::TRAINING_BATCH_SIZE {
            saved_games[rng.next_u32() as usize % constants::TRAINING_SET_SIZE]
                .add_to_batch(&mut tb);
        }

        Ok(tb)
    }

    /// Adds a position snapshot to the batch.
    pub fn add(&mut self, p: &Position, mcts: &[f64], result: f32) {
        // Store MCTS counts
        assert_eq!(mcts.len(), 4096);
        self.mcts.extend_from_slice(mcts);

        // Store result
        self.results.push(result);

        // Add input to inner batch
        self.inner.add(p);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::str::FromStr;

    /// Tests the treebatch can be initialized.
    #[test]
    fn trainbatch_can_initialize() {
        TrainBatch::new(16);
    }

    /// Tests the treebatch can be added to.
    #[test]
    fn trainbatch_can_add() {
        let mut b = TrainBatch::new(16);
        b.add(&Position::new(), &[0.0; 4096], 0.0);
    }

    /// Tests a training batch can be generated from the disk.
    #[test]
    fn trainbatch_can_generate() {
        let games_dir = tempfile::tempdir().expect("tempdir failed");
        let games_path = games_dir.path();

        for i in 0..constants::TRAINING_SET_SIZE {
            let mut g = Game::new();

            g.make_move(
                chess::ChessMove::from_str("e2e4").expect("bad move"),
                Vec::new(),
            );

            g.finalize(1.0);
            g.save(&games_path.join(format!("{}.game", i)))
                .expect("game save failed");
        }

        TrainBatch::generate(&games_path).expect("trainbatch generated failed");
    }

    /// Tests a training batch can be generated from the disk.
    #[test]
    fn trainbatch_can_not_generate_incomplete() {
        let games_dir = tempfile::tempdir().expect("tempdir failed");
        let games_path = games_dir.path();

        for i in 0..constants::TRAINING_SET_SIZE {
            let mut g = Game::new();

            g.make_move(
                chess::ChessMove::from_str("e2e4").expect("bad move"),
                Vec::new(),
            );
            g.save(&games_path.join(format!("{}.game", i)))
                .expect("game save failed");
        }

        assert!(TrainBatch::generate(&games_path).is_err());
    }
}
