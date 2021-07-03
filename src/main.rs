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
mod tui;
mod worker;

use std::error::Error;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::thread::sleep;
use std::time::Duration;

use crate::tui::Tui;
use game::Game;
use input::trainbatch::TrainBatch;
use position::Position;
use searcher::{SearchStatus, Searcher};

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

    let mut tui = Tui::new();

    tui.start();

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
                tui.log(format!("Starting game {}", game_path.display()));
            } else {
                let g = game::Game::load(&game_path)?;

                if !g.is_complete() {
                    should_generate = true;
                    tui.log(format!("Resuming incomplete game {}", game_path.display()));
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

            tui.reset_game();

            loop {
                // Check if the game is over
                if current_position.is_game_over().is_some() {
                    break;
                }

                tui.set_position(current_position.clone());

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

                    tui.push_status(status.clone());

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

                    tui.push_score(score_mul * s);
                }

                // Make the selected move.
                current_position.make_move(selected_move);
                current_game.make_move(selected_move, final_tree.get_mcts_data());

                // Stop training if the user has requested a stop.
                if tui.exit_requested() {
                    break;
                }
            }

            if current_position.is_game_over().is_some() {
                // Game is finished, write to disk.
                current_game.finalize(current_position.is_game_over().unwrap());
                tui.log(format!(
                    "Game over, result {}",
                    current_position.is_game_over().unwrap()
                ));
            }

            current_game
                .save(&game_path)
                .expect("failed saving completed game");
            tui.log(format!("Wrote game to {}", game_path.display()));

            if tui.exit_requested() {
                break;
            }
        }

        if tui.exit_requested() {
            break;
        }

        // Generate training batches.
        let training_batches: Vec<TrainBatch> = (0..constants::TRAINING_BATCH_COUNT)
            .map(|_| TrainBatch::generate(&games_dir))
            .collect::<Result<Vec<TrainBatch>, _>>()?;

        // Train model.
        tui.log("Training model.");
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

        tui.log(format!("Archived games for generation {}", cur));

        if tui.exit_requested() {
            break;
        }
    }

    tui.stop();
    Ok(())
}
