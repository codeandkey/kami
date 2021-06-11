use crate::model::{
    self,
    Model,
    ModelPtr,
    mock::MockModel,
};

use crate::input::trainbatch::TrainBatch;
use crate::position::Position;

use chess::ChessMove;
use rand::prelude::*;
use rand::thread_rng;
use std::fs::{self, File};
use std::io::{self, BufReader, BufRead};
use std::error::Error;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, RwLock};

/// Manages data directories and their contents.
pub struct Disk {
    loaded: Option<ModelPtr>,
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
            loaded: None,
            data_dir: data_dir.to_path_buf(),
            games_dir: data_dir.join("games"),
            latest_path: data_dir.join("latest"),
            archive_dir: data_dir.join("archive"),
        })
    }

    /// Returns a pointer to the loaded model, if a model is loaded.
    pub fn get_model(&self) -> Option<ModelPtr> {
        self.loaded.clone()
    }

    /// Loads the latest generation, if one is available.
    /// Returns Ok(true) if the model was loaded, Ok(false) if no model is present,
    /// or Err(e) if some other error occurred.
    pub fn load(&mut self) -> Result<bool, Box<dyn Error>> {
        let res = model::load(&self.latest_path)?;
        
        match res {
            Some(m) => {
                self.loaded = Some(m);
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Loads a model directly and writes it to the disk.
    pub fn generate(&mut self, md: ModelPtr) -> Result<(), Box<dyn Error>> {     
        model::save(&md, &self.latest_path)?;
        self.loaded = Some(md);
        Ok(())
    }

    /// Gets the path to the next game to be played.
    /// If any games are incomplete, they are removed from the training set.
    pub fn next_game_path(&self, trainbatch_size: usize) -> Result<Option<PathBuf>, Box<dyn Error>> {
        for game_id in 0..trainbatch_size {
            // Check if the game has already been generated
            let game_path = self.games_dir.join(format!("{}.game", game_id));

            if game_path.exists() {
                // If the game path isn't a file, something is very wrong
                if !game_path.is_file() {
                    return Err(format!(
                        "{} exists but is not a game file, refusing to proceed!",
                        game_path.display()
                    ).into());
                }

                // Check the game has a result at the end
                let fd = File::open(&game_path)?;
                let reader = BufReader::new(fd);

                let last_line = reader.lines().last().unwrap_or(Ok("".to_string())).unwrap();

                if last_line.starts_with("result") {
                    continue;
                } else {
                    fs::remove_file(&game_path)?;
                    return Ok(Some(game_path));
                }
            }
        }

        Ok(None)
    }

    /// Loads a training batch from the disk.
    /// Returns Err if the game set is not complete.
    pub fn get_training_batch(&self, count: usize, size: usize) -> Result<Vec<TrainBatch>, Box<dyn Error>> {
        let mut training_batches: Vec<TrainBatch> = Vec::new();

        for _ in 0..count {
            let mut next_batch = TrainBatch::new(size);

            for _ in 0..size {
                // Load a random position from a random game.
                // Holy declarations batman!
                let mut rng = thread_rng();
                let game_id = rng.next_u32() as usize % size;
                let game_path = self.games_dir.join(format!("{}.game", game_id));
                let game_file = File::open(&game_path).expect("failed to open game file");
                let reader = BufReader::new(game_file);
                let lines = reader.lines().map(Result::unwrap).collect::<Vec<String>>();
                let pos_idx = rng.next_u32() as usize % (lines.len() - 1);
                let pos_line = &lines[pos_idx];
                let result = lines
                    .last()
                    .unwrap()
                    .split(' ')
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>()[1]
                    .parse::<f32>()
                    .unwrap();

                // Fast-forward a game to this position
                let mut pos = Position::new();
                for i in 0..pos_idx {
                    assert!(pos.make_move(
                        ChessMove::from_str(lines[i].split(' ').next().unwrap()).unwrap()
                    ));
                }

                // Grab MCTS data from this line
                let mcts_data = pos_line
                    .split(' ')
                    .skip(1)
                    .map(|x| x.parse::<f32>().unwrap())
                    .collect::<Vec<f32>>();

                next_batch.add(&pos, &mcts_data, result);
            }

            training_batches.push(next_batch);
        }

        Ok(training_batches)
    }

    /// Archives the current model, and then trains the loaded model on a batch.
    /// Returns the new generation number, or an Err if something is wrong.
    pub fn train(&self, training_batches: Vec<TrainBatch>) -> Result<usize, Box<dyn Error>> {
        if self.loaded.is_none() {
            return Err("No model is loaded!".into());
        }

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

        // Make new games dir for new generation.
        fs::create_dir_all(&self.games_dir)?;

        self.loaded.as_ref().unwrap().write().unwrap().train(training_batches);

        model::save(self.loaded.as_ref().unwrap(), &self.latest_path)?;

        return Ok(cur);
    }
}