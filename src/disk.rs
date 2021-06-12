use crate::model::{self, ModelPtr};

use crate::constants;
use crate::game::Game;
use crate::input::trainbatch::TrainBatch;



use rand::prelude::*;
use rand::thread_rng;
use std::error::Error;
use std::fs::{self};
use std::io::{self};
use std::path::{Path, PathBuf};

/// Performs all disk related options.
pub struct Disk {
    data_dir: PathBuf,
    games_dir: PathBuf,
    archive_dir: PathBuf,
    latest_path: PathBuf,
}

impl Disk {
    /// Returns a new Disk instance.
    /// Initializes the disk directory if it is not already.
    pub fn new(data_dir: &Path) -> Result<Self, io::Error> {
        fs::create_dir_all(&data_dir)?;

        Ok(Disk {
            data_dir: data_dir.to_path_buf(),
            games_dir: data_dir.join("games"),
            latest_path: data_dir.join("latest"),
            archive_dir: data_dir.join("archive"),
        })
    }

    /// Loads the latest generation, if one is available.
    /// Returns Ok(Some(model)) if the model was loaded, Ok(None) if no model is present,
    /// or Err(e) if some other error occurred.
    pub fn load_model(&mut self) -> Result<Option<ModelPtr>, Box<dyn Error>> {
        Ok(model::load(&self.latest_path)?)
    }

    /// Archives the current model and returns the archived generation number.
    pub fn archive_model(&self) -> Result<usize, Box<dyn Error>> {
        if !self.archive_dir.is_dir() {
            fs::create_dir_all(&self.archive_dir)?;
        }

        // Walk through archive generations to find the lowest available slot.
        let mut cur: usize = 0;
        let mut gen_path: PathBuf;

        loop {
            gen_path = self.archive_dir.join(format!("generation_{}", cur));

            if !gen_path.exists() {
                break;
            }

            cur += 1;
        }

        fs_extra::move_items(
            &[&self.latest_path, &self.games_dir],
            gen_path,
            &fs_extra::dir::CopyOptions::new(),
        )?;

        Ok(cur)
    }

    /// Loads a model directly and writes it to the disk.
    pub fn save_model(&mut self, md: ModelPtr) -> Result<(), Box<dyn Error>> {
        model::save(&md, &self.latest_path)?;
        Ok(())
    }

    /// Gets the path to the next game to be played.
    /// Returns the path to any incomplete games as well.
    pub fn next_game_path(&self) -> Result<Option<PathBuf>, Box<dyn Error>> {
        if !self.games_dir.is_dir() {
            fs::create_dir_all(&self.games_dir)?;
        }

        for game_id in 0..constants::TRAINING_SET_SIZE {
            // Check if the game has already been generated
            let game_path = self.games_dir.join(format!("{}.game", game_id));

            if !game_path.exists() {
                return Ok(Some(game_path.clone()));
            }

            // If the game path isn't a file, something is very wrong
            if !game_path.is_file() {
                return Err(format!(
                    "{} exists but is not a game file, refusing to proceed!",
                    game_path.display()
                )
                .into());
            }

            // Parse the game.
            let gm = Game::load(&game_path)?;

            // Check the game was completed.
            if !gm.is_complete() {
                return Ok(Some(game_path.clone()));
            }         
        }

        Ok(None)
    }

    /// Loads training batches from the disk.
    /// Returns Err if the game set is not complete.
    pub fn get_training_batches(&self) -> Result<Vec<TrainBatch>, Box<dyn Error>> {
        let mut saved_games: Vec<Game> = Vec::new();

        for i in 0..constants::TRAINING_SET_SIZE {
            let game_path = self.games_dir.join(format!("{}.game", i));

            // Parse the game.
            let gm = Game::load(&game_path)?;

            // Check the game was completed.
            if !gm.is_complete() {
                return Err("Training set contains incomplete games!".into());
            }

            saved_games.push(gm);
        }

        let mut training_batches: Vec<TrainBatch> = Vec::new();
        let mut rng = thread_rng();

        for _ in 0..constants::TRAINING_BATCH_COUNT {
            let mut tb = TrainBatch::new(constants::TRAINING_BATCH_SIZE);

            for _ in 0..constants::TRAINING_BATCH_SIZE {
                saved_games[rng.next_u32() as usize % constants::TRAINING_SET_SIZE].add_to_batch(&mut tb);
            };

            training_batches.push(tb);
        }

        Ok(training_batches)
    }
}