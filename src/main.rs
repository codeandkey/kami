extern crate dirs;
extern crate serde;

#[cfg(feature = "tch")]
extern crate tch;

mod constants;
mod game;
mod input;
mod model;
mod node;
mod position;
mod searcher;
mod tree;
mod ui;
mod worker;

use crossterm::{
    execute,
    terminal::{enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    event::{self, Event, KeyCode},
};

use std::error::Error;
use std::fs;
use std::io::stdout;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread::{sleep, spawn};
use std::time::Duration;
use tui::backend::CrosstermBackend;

use game::Game;
use input::trainbatch::TrainBatch;
use position::Position;
use searcher::{SearchStatus, Searcher};
use ui::Ui;

/**
 * Server entry point
 */
fn main() -> Result<(), Box<dyn Error>> {
    // Print program information
    println!(
        "========================== kami {} ==========================",
        env!("CARGO_PKG_VERSION")
    );
    println!("\tA chess engine powered by reinforcement learning");
    println!("\tJustin Stanley <jtst@iastate.edu>");
    println!("================================================================");

    sleep(Duration::from_secs(1));

    // Set data dir
    let data_dir = dirs::data_dir().unwrap().join("kami");

    train(&data_dir)?;

    println!("Shutdown complete, goodbye :)");
    Ok(())
}

/// Runs the training loop.
/// Starts a TUI and feeds the current status to it.
fn train(data_dir: &Path) -> Result<(), Box<dyn Error>> {
    // Set up directories.
    const MODEL_TYPE: model::Type = model::Type::Torch;

    let games_dir = data_dir.join("games");
    let model_dir = data_dir.join("model");
    let archive_dir = data_dir.join("archive");

    // Generate new model if needed.
    if !model_dir.exists() {
        println!("Generating new model.");
        model::generate(&model_dir, MODEL_TYPE)?;
    }

    // Switch to TUI screen and start rendering TUI.
    enable_raw_mode().expect("failed setting raw mode");

    execute!(stdout(), EnterAlternateScreen)
        .expect("failed starting alternate screen");

    let mut ui = Ui::new();
    ui.start(CrosstermBackend::new(stdout()));

    let ui_tx = ui.tx();

    // Start event thread.

    let should_stop = Arc::new(Mutex::new(false));
            
    let inp_ui_tx = ui_tx.clone();
    let inp_should_stop = should_stop.clone();
    let inp_thread = spawn(move || {
        while !*inp_should_stop.lock().unwrap() {
            match event::read().unwrap() {
                Event::Key(e) => match e.code {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        *inp_should_stop.lock().unwrap() = true;
                        inp_ui_tx.send(ui::Event::Log("Shutting down..".to_string())).unwrap()
                    },
                    KeyCode::Char('p') => inp_ui_tx.send(ui::Event::Pause).unwrap(),
                    _ => (),
                },
                _ => (),
            };
        }
    });

    // Start training loop.
    loop {
        let current_model = Arc::new(model::load(&model_dir, true)?);

        // Make games destination.
        if !games_dir.exists() {
            fs::create_dir_all(&games_dir)?;
        }

        // Generate training set.
        for game_id in 0..constants::TRAINING_SET_SIZE {
            let game_path = games_dir.join(format!("{}.game", game_id));

            let mut should_generate = false;

            if !game_path.exists() {
                should_generate = true;
                ui_tx.send(ui::Event::Log(format!("Starting game {}", game_path.display()))).unwrap();
            } else {
                let g = game::Game::load(&game_path)?;

                if !g.is_complete() {
                    should_generate = true;
                    ui_tx.send(ui::Event::Log(format!("Resuming incomplete game {}", game_path.display()))).unwrap();
                }
            }

            if !should_generate {
                continue;
            }

            // Play the game and save to disk.
            let mut current_game = match Game::load(&game_path) {
                Ok(g) => g,
                Err(_) => Game::new(),
            };

            // Setup a position.
            let mut current_position = Position::new();

            for mv in current_game.get_actions() {
                assert!(current_position.make_move(mv));
            }

            ui_tx.send(ui::Event::Reset).unwrap();

            loop {
                // Check if the game is over
                if current_position.is_game_over().is_some() {
                    break;
                }

                ui_tx.send(ui::Event::Position(current_position.clone())).unwrap();

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
                        current_model.clone(),
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

                    ui_tx.send(ui::Event::Status(status.clone())).unwrap();

                    if matches!(status, SearchStatus::Done) {
                        break;
                    }
                }

                // Wait for search to stop and collect final tree.
                let final_tree = search.wait().expect("search did not return tree");

                // Examine tree and perform move selection.
                let selected_move = final_tree.select();

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

                    ui_tx.send(ui::Event::Score(score_mul * s)).unwrap();
                }

                // Make the selected move.
                current_position.make_move(selected_move);
                current_game.make_move(selected_move, final_tree.get_mcts_data());

                // Stop training if the user has requested a stop.
                if *should_stop.lock().unwrap() {
                    break;
                }
            }

            if current_position.is_game_over().is_some() {
                // Game is finished, write to disk.
                current_game.finalize(current_position.is_game_over().unwrap());

                ui_tx.send(ui::Event::Log(format!(
                    "Game over, result {}",
                    current_position.is_game_over().unwrap()
                ))).unwrap();
            }

            current_game
                .save(&game_path)
                .expect("failed saving completed game");

            ui_tx.send(ui::Event::Log(format!("Wrote game to {}", game_path.display()))).unwrap();

            if *should_stop.lock().unwrap() {
                break;
            }
        }

        if *should_stop.lock().unwrap() {
            break;
        }

        // Generate training batches.
        let training_batches: Vec<TrainBatch> = (0..constants::TRAINING_BATCH_COUNT)
            .map(|_| TrainBatch::generate(&games_dir))
            .collect::<Result<Vec<TrainBatch>, _>>()?;

        // Train model.
        ui_tx.send(ui::Event::Log("Training model".to_string())).unwrap();
        model::train(
            &model_dir,
            training_batches,
            model::get_type(&current_model),
        )?;

        // Archive games.
        // Walk through archive generations to find the lowest available slot.
        let mut cur: usize = 0;
        let mut gen_path = archive_dir.join(format!("generation_{}", cur));

        if !archive_dir.exists() {
            fs::create_dir_all(&archive_dir)?;
        }

        while gen_path.exists() {
            cur += 1;
            gen_path = archive_dir.join(format!("generation_{}", cur));
        }

        // Create target generation dir.
        fs::create_dir_all(&gen_path)?;

        fs_extra::copy_items(&[&games_dir], gen_path, &fs_extra::dir::CopyOptions::new())?;

        // Regenerate games dir.
        fs::remove_dir_all(&games_dir)?;
        fs::create_dir_all(&games_dir)?;

        ui_tx.send(ui::Event::Log(format!("Archived games for generation {}", cur))).unwrap();

        if *should_stop.lock().unwrap() {
            break;
        }
    }

    ui_tx.send(ui::Event::Stop).unwrap();

    inp_thread.join().expect("failed joining input thread");
    ui.join();

    // Leave TUI screen and reset terminal.
    execute!(stdout(), LeaveAlternateScreen)
        .expect("failed leaving alternate screen");

    Ok(())
}
