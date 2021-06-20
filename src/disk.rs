use crate::model::{self, Model};

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
    games_dir: PathBuf,
    archive_dir: PathBuf,
    latest_path: PathBuf,
}

impl Disk {
    /// Returns a new Disk instance.
    /// Initializes the disk directory if it is not already.
    pub fn new(data_dir: &Path) -> Result<Self, io::Error> {
        fs::create_dir_all(&data_dir)?;

        let games_dir = data_dir.join("games");
        let archive_dir = data_dir.join("archive");

        fs::create_dir_all(&games_dir)?;
        fs::create_dir_all(&archive_dir)?;

        Ok(Disk {
            games_dir: games_dir,
            latest_path: data_dir.join("model"),
            archive_dir: archive_dir,
        })
    }

    /// Gets the latest model path.
    pub fn get_model_path(&self) -> PathBuf {
        self.latest_path.clone()
    }

    /// Loads the latest generation, if one is available.
    /// Returns Ok(Some(model)) if the model was loaded, Ok(None) if no model is present,
    /// or Err(e) if some other error occurred.
    pub fn load_model(&self) -> Result<Option<Model>, Box<dyn Error>> {
        Ok(model::load(&self.latest_path)?)
    }

    /// Archives the current model and returns the archived generation number.
    pub fn archive_model(&self) -> Result<usize, Box<dyn Error>> {
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

        // Create target generation dir.
        fs::create_dir_all(&gen_path)?;

        fs_extra::copy_items(
            &[&self.latest_path, &self.games_dir],
            gen_path,
            &fs_extra::dir::CopyOptions::new(),
        )?;

        // Regenerate games dir.
        fs::remove_dir_all(&self.games_dir)?;
        fs::create_dir_all(&self.games_dir)?;

        Ok(cur)
    }

    /// Gets the path to the next game to be played.
    /// Returns the path to any incomplete games as well.
    pub fn next_game_path(&self) -> Result<Option<PathBuf>, Box<dyn Error>> {
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

            let gm = match Game::load(&game_path) {
                Ok(gm) => gm,
                Err(e) => panic!("corrupted game file {}: {}", &game_path.display(), e),
            };

            //let gm = Game::load(&game_path)?;

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

        for _i in 0..constants::TRAINING_BATCH_COUNT {
            let mut tb = TrainBatch::new(constants::TRAINING_BATCH_SIZE);

            for _ in 0..constants::TRAINING_BATCH_SIZE {
                saved_games[rng.next_u32() as usize % constants::TRAINING_SET_SIZE]
                    .add_to_batch(&mut tb);
            }

            training_batches.push(tb);
        }

        Ok(training_batches)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::model::Type;
    use chess::ChessMove;
    use std::str::FromStr;

    /// Returns a tempdir for mocking.
    fn mock_data_dir() -> PathBuf {
        tempfile::tempdir()
            .expect("failed creating data dir")
            .into_path()
    }

    /// Tests the disk can be initialized.
    #[test]
    fn disk_can_initialize() {
        let data_dir = mock_data_dir();
        Disk::new(&data_dir).expect("failed initializing disk");

        assert!(data_dir.join("games").is_dir());
        assert!(data_dir.join("archive").is_dir());
    }

    /// Tests the disk can load the latest model if it exists.
    #[test]
    fn disk_can_find_latest() {
        let data_dir = mock_data_dir();
        let d = Disk::new(&data_dir).expect("failed initializing disk");

        // Write a mock model to the latest path.
        model::generate(&data_dir.join("model"), Type::Mock).expect("model gen failed");

        assert!(matches!(
            d.load_model().expect("failed loading model").unwrap(),
            Model::Mock
        ));
    }

    /// Tests the disk can archive a model.
    #[test]
    fn disk_can_archive_model() {
        let data_dir = mock_data_dir();
        let d = Disk::new(&data_dir).expect("failed initializing disk");

        // Write a mock model to the latest path.
        model::generate(&d.get_model_path(), Type::Mock).expect("model gen failed");

        assert_eq!(d.archive_model().expect("archive model failed"), 0);

        assert!(data_dir.join("archive").join("generation_0").is_dir());
        assert!(data_dir
            .join("archive")
            .join("generation_0")
            .join("model")
            .is_dir());
        assert!(data_dir
            .join("archive")
            .join("generation_0")
            .join("model")
            .join("mock.type")
            .is_file());
        assert!(data_dir
            .join("archive")
            .join("generation_0")
            .join("games")
            .is_dir());
    }

    /// Tests the disk can find the next game path when no games are generated.
    #[test]
    fn disk_can_get_next_game_path_empty() {
        let data_dir = mock_data_dir();
        let d = Disk::new(&data_dir).expect("failed initializing disk");

        assert_eq!(
            d.next_game_path()
                .expect("failed getting next game path")
                .unwrap(),
            data_dir.join("games").join("0.game")
        );
    }

    /// Tests the disk can find the next game path when there is an incomplete game.
    #[test]
    fn disk_can_get_next_game_path_incomplete() {
        let data_dir = mock_data_dir();
        let d = Disk::new(&data_dir).expect("failed initializing disk");

        let mut g = Game::new();

        // Write an incomplete game to 1
        g.make_move(ChessMove::from_str("e2e4").unwrap(), Vec::new());
        g.save(&data_dir.join("games").join("1.game"))
            .expect("game write failed");

        // Write a completed game to 0
        g.finalize(1.0);
        g.save(&data_dir.join("games").join("0.game"))
            .expect("game write failed");

        // Next game should be 1
        assert_eq!(
            d.next_game_path()
                .expect("failed getting next game path")
                .unwrap(),
            data_dir.join("games").join("1.game")
        );
    }

    /// Tests the disk returns no next game path when the set is complete.
    #[test]
    fn disk_can_get_next_game_path_all_complete() {
        let data_dir = mock_data_dir();
        let d = Disk::new(&data_dir).expect("failed initializing disk");
        let mut g = Game::new();

        g.make_move(ChessMove::from_str("e2e4").unwrap(), Vec::new());
        g.finalize(1.0);

        for gid in 0..constants::TRAINING_SET_SIZE {
            // Write a completed game to n
            g.save(&data_dir.join("games").join(format!("{}.game", gid)))
                .expect("game write failed");
        }

        // No next game!
        assert!(d
            .next_game_path()
            .expect("failed getting next path")
            .is_none());
    }

    /// Tests the disk can return a valid training batch set.
    #[test]
    fn disk_can_get_training_batches() {
        let data_dir = mock_data_dir();
        let d = Disk::new(&data_dir).expect("failed initializing disk");

        let mut g = Game::new();

        g.make_move(ChessMove::from_str("e2e4").unwrap(), Vec::new());
        g.finalize(1.0);

        for gid in 0..constants::TRAINING_SET_SIZE {
            // Write a completed game to n
            g.save(&data_dir.join("games").join(format!("{}.game", gid)))
                .expect("game write failed");
        }

        let batches = d
            .get_training_batches()
            .expect("failed getting training batches");

        assert_eq!(batches.len(), constants::TRAINING_BATCH_COUNT);

        for b in batches {
            assert_eq!(b.get_inner().get_size(), constants::TRAINING_BATCH_SIZE);
            assert_eq!(b.get_results(), &[1.0; constants::TRAINING_BATCH_SIZE]);
            assert_eq!(b.get_mcts(), &[0.0; constants::TRAINING_BATCH_SIZE * 4096]);
        }
    }

    /// Tests the disk does not get training batches from incomplete games.
    #[test]
    fn disk_can_get_training_batches_incomplete() {
        let data_dir = mock_data_dir();
        let d = Disk::new(&data_dir).expect("failed initializing disk");

        let mut g = Game::new();

        g.make_move(ChessMove::from_str("e2e4").unwrap(), Vec::new());

        for gid in 0..constants::TRAINING_SET_SIZE {
            // Write a completed game to n
            g.save(&data_dir.join("games").join(format!("{}.game", gid)))
                .expect("game write failed");
        }

        assert!(d.get_training_batches().is_err());
    }

    /// Tests the disk does not get training batches from missing games
    #[test]
    fn disk_can_get_training_batches_missing_games() {
        let data_dir = mock_data_dir();
        let d = Disk::new(&data_dir).expect("failed initializing disk");

        let mut g = Game::new();

        g.make_move(ChessMove::from_str("e2e4").unwrap(), Vec::new());

        for gid in 0..constants::TRAINING_SET_SIZE - 1 {
            // Write a completed game to n
            g.save(&data_dir.join("games").join(format!("{}.game", gid)))
                .expect("game write failed");
        }

        assert!(d.get_training_batches().is_err());
    }
}
