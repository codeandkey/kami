use crate::constants;
use crate::disk::Disk;
use crate::game::Game;
use crate::model::{self, Model, Type};
use crate::position::Position;
use crate::searcher::{SearchStatus, Searcher};
use crate::tui::Tui;

use std::error::Error;
use std::sync::{Arc, Mutex};
use std::thread::{spawn, JoinHandle};

/// Manages the model training lifecycle.
/// Displays training status in a TUI on stdout.
pub struct Trainer {
    latest: Arc<Model>,
    diskmgr: Arc<Mutex<Disk>>,
    handle: Option<JoinHandle<()>>,
    stopflag: Arc<Mutex<bool>>,
}

impl Trainer {
    /// Creates a new trainer instance with the latest generation loaded.
    pub fn new(
        diskmgr: Arc<Mutex<Disk>>,
        mut latest: Option<Model>,
    ) -> Result<Self, Box<dyn Error>> {
        // Load model, generate one if doesn't exist.
        let model_path = diskmgr.lock().unwrap().get_model_path();

        if latest.is_none() {
            latest = model::load(&model_path)?;

            if latest.is_none() {
                let model_path = diskmgr.lock().unwrap().get_model_path();
                model::generate(&model_path, Type::Torch)?;
                latest = Some(model::load(&model_path)?.unwrap());
            }
        }

        Ok(Trainer {
            latest: Arc::new(latest.unwrap()),
            diskmgr: diskmgr,
            handle: None,
            stopflag: Arc::new(Mutex::new(false)),
        })
    }

    /// Starts training the model.
    pub fn start(&mut self) {
        assert!(self.handle.is_none());

        let thr_diskmgr = self.diskmgr.clone();
        let thr_model = self.latest.clone();
        let thr_stop = self.stopflag.clone();

        self.handle = Some(spawn(move || {
            let mut tui = Tui::new();
            tui.start();

            loop {
                tui.log(format!(
                    "Building training set of {} games.",
                    constants::TRAINING_SET_SIZE
                ));

                // Build training set.
                while let Some(game_path) = thr_diskmgr.lock().unwrap().next_game_path().unwrap() {
                    let mut current_game: Game;

                    // Setup a position.
                    let mut current_position = Position::new();

                    if game_path.exists() {
                        tui.log(format!("Resuming incomplete game {}", game_path.display()));

                        current_game =
                            Game::load(&game_path).expect("failed loading incomplete game");

                        for mv in current_game.get_actions() {
                            assert!(current_position.make_move(mv));
                        }
                    } else {
                        current_game = Game::new();
                        tui.log(format!("Generating game {}", game_path.display()));
                    }

                    tui.reset_game();

                    loop {
                        // Load position into tui
                        tui.set_position(current_position.clone());

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

                            let will_stop = matches!(status, SearchStatus::Done);

                            if let SearchStatus::Searching(stat) = &status {
                                let nodes: usize = stat.workers.iter().map(|w| w.total_nodes).sum();
                                tui.push_nps((nodes as f64 / stat.elapsed_ms as f64) * 1000.0);
                            }

                            tui.push_status(status);

                            if will_stop {
                                break;
                            }
                        }

                        // Wait for search to stop and collect final tree.
                        let final_tree = search.wait().expect("search did not return tree");

                        // Examine tree and perform move selection.
                        let selected_move = final_tree.select();

                        // Make the selected move.
                        current_position.make_move(selected_move);
                        current_game.make_move(selected_move, final_tree.get_mcts_data());

                        // Find move with best n
                        let mut best_n = 0;
                        let mut best_score: Option<f64> = None;

                        if let Some(children) = &final_tree[0].children {
                            for &nd in children {
                                if final_tree[nd].n > best_n {
                                    best_n = final_tree[nd].n;
                                    best_score = Some(final_tree[nd].q());
                                }
                            }
                        }

                        if let Some(s) = best_score {
                            let score_mul = match current_position.side_to_move() {
                                chess::Color::White => 1.0,
                                chess::Color::Black => -1.0,
                            };

                            tui.push_score(score_mul * s);
                        }

                        // Stop training if the user has requested a stop.
                        if tui.exit_requested() {
                            *thr_stop.lock().unwrap() = true;
                        }

                        if *thr_stop.lock().unwrap() {
                            break;
                        }
                    }

                    if current_position.is_game_over().is_some() {
                        // Game is finished, write to disk.
                        current_game.finalize(current_position.is_game_over().unwrap());
                    }

                    current_game
                        .save(&game_path)
                        .expect("failed saving completed game");
                    tui.log(format!("Wrote game to {}", game_path.display()));

                    if *thr_stop.lock().unwrap() {
                        break;
                    }
                }

                if *thr_stop.lock().unwrap() {
                    break;
                }

                // If training set is complete, archive and train the model.
                if thr_diskmgr
                    .lock()
                    .unwrap()
                    .next_game_path()
                    .unwrap()
                    .is_none()
                {
                    // Build training batches.
                    tui.log("Training set complete, building training batches.");

                    let tbatches = thr_diskmgr
                        .lock()
                        .unwrap()
                        .get_training_batches()
                        .expect("failed getting training batch");

                    tui.log("Archiving model.");
                    let gen = thr_diskmgr
                        .lock()
                        .unwrap()
                        .archive_model()
                        .expect("failed archiving model");
                    tui.log(format!("Archived as generation {}.", gen));

                    tui.log("Training model.");
                    model::train(
                        &thr_diskmgr.lock().unwrap().get_model_path(),
                        tbatches,
                        model::get_type(&thr_model),
                    )
                    .expect("failed training");

                    tui.log(format!(
                        "Finished training model! Starting training set for generation {}",
                        gen + 1
                    ));
                }

                if *thr_stop.lock().unwrap() {
                    break;
                }
            }

            tui.stop();
        }));
    }

    /// Waits for the trainer to stop.
    pub fn wait(&mut self) -> Result<(), Box<dyn Error>> {
        if self.handle.is_none() {
            return Err("Trainer is not running!".into());
        }

        self.handle
            .take()
            .unwrap()
            .join()
            .expect("failed joining trainer thread");

        Ok(())
    }

    /// Stops and joins the trainer.
    pub fn stop(&mut self) -> Result<(), Box<dyn Error>> {
        *self.stopflag.lock().unwrap() = true;
        self.wait()?;
        *self.stopflag.lock().unwrap() = false;

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
        let data_dir = tempfile::tempdir()
            .expect("failed creating data dir")
            .into_path();
        Arc::new(Mutex::new(
            Disk::new(&data_dir).expect("failed initializing disk"),
        ))
    }

    /// Tests the trainer can be initialized.
    #[test]
    fn trainer_can_initialize() {
        Trainer::new(mock_disk(), Some(Model::Mock)).expect("failed initializing trainer");
    }

    /// Tests the trainer can start and shortly stop training.
    #[test]
    fn trainer_can_start_stop_training() {
        let mut t =
            Trainer::new(mock_disk(), Some(Model::Mock)).expect("failed initializing trainer");

        t.start();
        std::thread::sleep(std::time::Duration::from_secs(3));
        t.stop().expect("failed stopping trainer");
    }

    /// Tests the trainer cannot start twice.
    #[test]
    #[should_panic]
    fn trainer_no_start_twice() {
        let mut t =
            Trainer::new(mock_disk(), Some(Model::Mock)).expect("failed initializing trainer");

        t.start();
        t.start();

        // this should never be reached anyway, but try to avoid hanging the tests
        // if things get really broken

        t.stop().expect("trainer stop failed..");
    }

    /// Tests the trainer cannot stop without starting.
    #[test]
    fn trainer_stop_without_start() {
        let mut t =
            Trainer::new(mock_disk(), Some(Model::Mock)).expect("failed initializing trainer");

        assert!(t.wait().is_err());
    }

    /// Tests the trainer can train a model when the game set is complete.
    #[test]
    fn trainer_can_train_model() {
        let data_dir = tempfile::tempdir()
            .expect("failed creating data dir")
            .into_path();
        let d = Arc::new(Mutex::new(
            Disk::new(&data_dir).expect("failed initializing disk"),
        ));

        model::generate(&d.lock().unwrap().get_model_path(), Type::Mock).expect("model gen failed");
        let mut t = Trainer::new(d.clone(), None).expect("trainer init failed");

        let mut g = Game::new();

        g.make_move(ChessMove::from_str("e2e4").unwrap(), Vec::new());
        g.finalize(1.0);

        // Build a completed set of "games" so the trainer immediately trains the model.
        for gid in 0..constants::TRAINING_SET_SIZE {
            // Write a completed game to n
            g.save(&data_dir.join("games").join(format!("{}.game", gid)))
                .expect("game write failed");
        }

        t.start();

        std::thread::sleep(std::time::Duration::from_secs(3));

        t.stop().expect("failed stopping trainer");

        // Check the model was archived
        assert!(data_dir.join("archive").join("generation_0").is_dir());
    }
}
