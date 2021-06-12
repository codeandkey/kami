use crate::input::trainbatch::TrainBatch;
use crate::position::Position;

use chess::ChessMove;
use rand::prelude::*;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::str::FromStr;

/// Holds data relevant to a single played game.
#[derive(Serialize, Deserialize)]
pub struct Game {
    mcts: Vec<Vec<f32>>,
    actions: Vec<String>,
    result: Option<f32>,
}

impl Game {
    /// Creates a new game at the starting position.
    pub fn new() -> Self {
        Game {
            actions: Vec::new(),
            mcts: Vec::new(),
            result: None,
        }
    }

    /// Loads a game from a path.
    pub fn load(p: &Path) -> Result<Game, Box<dyn Error>> {
        let g = serde_json::from_reader(BufReader::new(File::open(p)?))?;

        Ok(g)
    }

    /// Writes a game to a path.
    pub fn save(&self, p: &Path) -> Result<(), Box<dyn Error>> {
        serde_json::to_writer_pretty(BufWriter::new(File::create(p)?), self)?;
        Ok(())
    }

    /// Makes a move.
    pub fn make_move(&mut self, action: ChessMove, mcts: &[f32]) {
        self.actions.push(action.to_string());
        self.mcts.push(mcts.to_vec());
    }

    /// Assigns the result for this game and allows it to be saved.
    pub fn finalize(&mut self, result: f32) {
        assert!(self.result.is_none());

        self.result = Some(result);
    }

    /// Tests if the game is completed.
    pub fn is_complete(&self) -> bool {
        self.result.is_some()
    }

    /// Gets the game moves in string format.
    pub fn to_string(&self) -> String {
        self.actions.join(" ")
    }

    /// Gets the game moves in vector format.
    pub fn get_actions(&self) -> Vec<ChessMove> {
        self.actions.iter().map(|x| ChessMove::from_str(x).expect("bad move")).collect()
    }

    /// Extends a TrainBatch from a random position in this game.
    pub fn add_to_batch(&self, tb: &mut TrainBatch) {
        let mut rng = thread_rng();
        let idx = rng.next_u32() as usize % self.actions.len();
        let mut pos = Position::new();

        // Make moves up to this point.
        for i in 0..idx {
            assert!(pos.make_move(ChessMove::from_str(&self.actions[i]).expect("bad move")));
        }

        tb.add(&pos, &self.mcts[idx], self.result.unwrap());
    }
}
