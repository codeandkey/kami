use crate::constants;
use crate::disk::Disk;
use crate::game::Game;
use crate::model::{self, Model, ModelPtr, mock::MockModel};
use crate::position::Position;
use crate::searcher::{Searcher, SearchStatus};


use std::error::Error;
use std::sync::{Arc, Mutex};
use std::thread::{spawn, JoinHandle};

/// Manages the model training lifecycle.
pub struct Trainer {
    latest: ModelPtr,
    diskmgr: Arc<Mutex<Disk>>,
    handle: Option<JoinHandle<()>>,
}

impl Trainer {
    /// Creates a new trainer instance with the latest generation loaded.
    pub fn new(diskmgr: Arc<Mutex<Disk>>) -> Result<Self, Box<dyn Error>> {
        // Load model, generate one if doesn't exist.
        let mut latest = diskmgr.lock().unwrap().load_model()?;

        if latest.is_none() {
            latest = Some(model::make_ptr(MockModel::generate()?));
            diskmgr.lock().unwrap().save_model(latest.as_ref().unwrap().clone())?;
        }
        
        Ok(Trainer {
            latest: latest.unwrap(),
            diskmgr: diskmgr,
            handle: None,
        })
    }

    /// Starts training the model.
    pub fn start_training(&mut self) -> Arc<Mutex<bool>> {
        assert!(self.handle.is_none());

        let thr_diskmgr = self.diskmgr.clone();
        let thr_model = self.latest.clone();
        let stop = Arc::new(Mutex::new(false));
        let thr_stop = stop.clone();

        self.handle = Some(spawn(move || {
            loop {
                println!("Building training set.");

                // Build training set.
                while let Some(game_path) = thr_diskmgr.lock().unwrap().next_game_path().unwrap() {
                    let mut current_game: Game;

                    // Setup a position.
                    let mut current_position = Position::new();

                    println!("Next target game path: {}", game_path.display());

                    if game_path.exists() {
                        println!("Resuming incomplete game {}", game_path.display());
                        current_game = Game::load(&game_path).expect("failed loading incomplete game");

                        for mv in current_game.get_actions() {
                            assert!(current_position.make_move(mv));
                        }
                    } else {
                        current_game = Game::new();
                        println!("Generating game {}", game_path.display());
                    }
        
                    loop {
                        // Check if the game is over
                        if current_position.is_game_over().is_some() {
                            break;
                        }
        
                        // Game is not over, we will search the position and make a move.
        
                        // Initialize searcher
                        let mut search = Searcher::new();
        
                        // Choose temperature for this search
                        let mut temperature = constants::TEMPERATURE;
        
                        if current_position.ply() >= constants::TEMPERATURE_DROPOFF_PLY {
                            temperature = constants::TEMPERATURE_DROPOFF;
                        }
        
                        let search_rx = search
                            .start(
                                Some(constants::SEARCH_TIME),
                                thr_model.clone(),
                                current_position.clone(),
                                temperature,
                                constants::SEARCH_BATCH_SIZE,
                            )
                            .unwrap();
        
                        // Display search status until the search is done.
                        loop {
                            let status = search_rx
                                .recv()
                                .expect("unexpected recv fail from search status rx");
        
                            match status {
                                SearchStatus::Searching(status) => {
                                    println!("==> Game {}, hist {}", game_path.display(), current_game.to_string());
                                    status.print();
                                }
                                SearchStatus::Stopping => println!("Stopping search.."),
                                SearchStatus::Done => {
                                    println!("Stopped search.");
                                    break;
                                }
                            }
                        }
        
                        // Wait for search to stop and collect final tree.
                        let final_tree = search.wait().expect("search did not return tree");
        
                        // Examine tree and perform move selection.
                        let selected_move = final_tree.select();
        
                        // Write tree data to stdout
                        final_tree.get_status().unwrap().print();
        
                        // Make the selected move.
                        current_position.make_move(selected_move);
                        current_game.make_move(selected_move, &final_tree.get_mcts_data());

                        if *thr_stop.lock().unwrap() {
                            break;
                        }
                    }

                    if current_position.is_game_over().is_some() {
                        // Game is finished, write to disk.
                        current_game.finalize(current_position.is_game_over().unwrap());
                    }

                    current_game.save(&game_path).expect("failed saving completed game");
                    println!("Wrote game to {}", game_path.display());

                    if *thr_stop.lock().unwrap() {
                        break;
                    }
                }
                
                if *thr_stop.lock().unwrap() {
                    break;
                }

                // If training set is complete, archive and train the model.
                if thr_diskmgr.lock().unwrap().next_game_path().unwrap().is_none() {
                    // Build training batches.
                    println!("Training set complete, building training batches.");

                    let batches = thr_diskmgr.lock().unwrap().get_training_batches().expect("failed getting training batches");

                    println!("Archiving model.");
                    let gen = thr_diskmgr.lock().unwrap().archive_model().expect("failed archiving model");
                    println!("Archived as generation {}.", gen);

                    println!("Training model.");
                    thr_model.write().unwrap().train(batches);
                    println!("Finished training model! Starting training set for generation {}", gen + 1);
                }

                if *thr_stop.lock().unwrap() {
                    break;
                }
            }
        }));

        stop
    }

    /// Waits for the trainer to stop.
    pub fn wait(&mut self) -> Result<(), Box<dyn Error>> {
        if self.handle.is_none() {
            return Err("Trainer is not running!".into());
        }

        self.handle.take().unwrap().join().expect("failed joining trainer thread");

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use chess::ChessMove;
    use std::str::FromStr;

    /// Returns a Disk initialized on a tempdir for mocking.
    fn mock_disk() -> Arc<Mutex<Disk>> {
        let data_dir = tempfile::tempdir().expect("failed creating data dir").into_path();
        Arc::new(Mutex::new(Disk::new(&data_dir).expect("failed initializing disk")))
    }

    /// Tests the trainer can be initialized.
    #[test]
    fn trainer_can_initialize() {
        Trainer::new(mock_disk()).expect("failed initializing trainer");
    }

    /// Tests the trainer can start and shortly stop training.
    #[test]
    fn trainer_can_start_stop_training() {
        let mut t = Trainer::new(mock_disk()).expect("failed initializing trainer");
        let stopflag = t.start_training();

        std::thread::sleep(std::time::Duration::from_secs(3));

        *stopflag.lock().unwrap() = true;

        t.wait().expect("failed stopping trainer");
    }

    /// Tests the trainer cannot start twice.
    #[test]
    #[should_panic]
    fn trainer_no_start_twice() {
        let mut t = Trainer::new(mock_disk()).expect("failed initializing trainer");

        let sf1 = t.start_training();
        let sf2 = t.start_training();

        // this should never be reached anyway, but try to avoid hanging the tests
        // if things get really broken

        *sf1.lock().unwrap() = true;
        *sf2.lock().unwrap() = true;

        t.wait().expect("did not join trainer thread ??");
    }

    /// Tests the trainer cannot stop without starting.
    #[test]
    fn trainer_stop_without_start() {
        let mut t = Trainer::new(mock_disk()).expect("failed initializing trainer");

        assert!(t.wait().is_err());
    }

    /// Tests the trainer can train a model when the game set is complete.
    #[test]
    fn trainer_can_train_model() {
        let data_dir = tempfile::tempdir().expect("failed creating data dir").into_path();
        let d = Arc::new(Mutex::new(Disk::new(&data_dir).expect("failed initializing disk")));
        let mut t = Trainer::new(d.clone()).expect("trainer init failed");

        let mut g = Game::new();

        g.make_move(ChessMove::from_str("e2e4").unwrap(), &[0.0; 4096]);
        g.finalize(1.0);

        // Build a completed set of "games" so the trainer immediately trains the model.
        for gid in 0..constants::TRAINING_SET_SIZE {
            // Write a completed game to n
            g.save(&data_dir.join("games").join(format!("{}.game", gid))).expect("game write failed");
        }

        let stopflag = t.start_training();

        std::thread::sleep(std::time::Duration::from_secs(3));
        *stopflag.lock().unwrap() = true;
        t.wait().expect("failed stopping trainer");

        // Check the model was archived
        assert!(data_dir.join("archive").join("generation_0").is_dir());
    }
}