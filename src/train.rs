/// Routines for evaluating model ELO

use crate::constants;
use crate::dir;
use crate::input::trainbatch::TrainBatch;
use crate::game::Game;
use crate::model::{self, Model};
use crate::position::Position;
use crate::ui::{self, Ui};
use crate::searcher::{Searcher, SearchStatus};

use chess::{ChessMove, Color};
use crossterm::{
    execute,
    terminal::{enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    event::{self, Event, KeyCode},
};

use std::path::Path;
use std::process::{Command, Stdio};

use std::error::Error;
use std::fs;
use std::io::{stdout, stdin, Read, Write};
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::thread::{sleep, spawn};
use std::time::Duration;
use tui::backend::CrosstermBackend;

/// Starts the training loop.
/// Initializes a TUI, starts playing games and evaluating ELO progress.
pub fn train() -> Result<(), Box<dyn Error>> {
    // Generate model if needed.
    let model_dir = dir::model_dir()?;
    if !model_dir.exists() {
        model::generate(&model_dir, constants::DEFAULT_MODEL_TYPE)?;
    }

    // Switch to TUI screen.
    enable_raw_mode().expect("failed setting raw mode");

    execute!(stdout(), EnterAlternateScreen)
        .expect("failed starting alternate screen");

    let ui = Arc::new(Mutex::new(Ui::new()));
    ui.lock().unwrap().start(CrosstermBackend::new(stdout()));

    // Start event thread.   
    let inp_ui = ui.clone();
    let inp_thread = spawn(move || {
        while !inp_ui.lock().unwrap().should_exit() {
            if !event::poll(Duration::from_millis(100)).expect("event poll failed") {
                continue;
            }

            match event::read().unwrap() {
                Event::Key(e) => match e.code {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        inp_ui.lock().unwrap().request_exit();
                        inp_ui.lock().unwrap().log("Requested exit.");
                    },
                    KeyCode::Char('p') => inp_ui.lock().unwrap().pause(),
                    _ => (),
                },
                _ => (),
            };
        }
    });

    // Start training loop.
    for generation in dir::current_generation()?..usize::MAX {
        // Generate training set.
        if !generate_training_set(ui.clone())? {
            break;
        }

        // Archive and train model.
        ui.lock().unwrap().log("Training model.");
        advance_model(ui.clone())?;

        // Perform ELO evaluation if ready.
        if generation % constants::ELO_EVALUATION_INTERVAL == 0 {
            ui.lock().unwrap().log(format!("Evaluating engine ELO after generation {}.", generation + 1));

            if !evaluate_elo(ui.clone())? {
                break;
            }
        }
    }

    // Stop input thread.
    inp_thread.join().unwrap();

    // Stop ui thread.
    ui.lock().unwrap().join();

    // Leave TUI screen and reset terminal.
    execute!(stdout(), LeaveAlternateScreen)
        .expect("failed leaving alternate screen");

    return Ok(());
}

/// Generates a training set. Returns true if the training set is complete, or false if the user requested an exit.
/// Returns Err() if something fails.
fn generate_training_set(ui: Arc<Mutex<ui::Ui>>) -> Result<bool, Box<dyn Error>> {
    ui.lock().unwrap().log("Generating training set.");

    // Load the current model.
    let model = Arc::new(model::load(&dir::model_dir()?, true)?);

    // Walk through needed games and generate them.
    while let Some(next_game) = dir::next_game()? {
        // Set initial position
        let mut position = Position::new();

        let mut game = if next_game.exists() {
            ui.lock().unwrap().log(format!("Resuming game {}", next_game.display()));
            Game::load(&next_game)?
        } else {
            ui.lock().unwrap().log(format!("Generating game {}", next_game.display()));
            Game::new()
        };

        // Get game back up to current position
        for mv in &game.get_actions() {
            assert!(position.make_move(*mv), "Invalid move in resumed game!");
        }

        ui.lock().unwrap().reset();

        // Start searching until game is over.
        while position.is_game_over().is_none() {
            if ui.lock().unwrap().should_exit() {
                break;
            }

            ui.lock().unwrap().position(position.clone());

            let (selected_move, mcts) = do_search(model.clone(), ui.clone(), &position)?;
            assert!(position.make_move(selected_move));
            game.make_move(selected_move, mcts);
        }

        // Finalize game if it is over.
        if let Some(result) = position.is_game_over() {
            game.finalize(result);

            ui.lock().unwrap().log(format!("Game over, {}",
                if result > 0.0 {
                    "1-0"
                } else {
                    if result < 0.0 {
                        "0-1"
                    } else {
                        "1/2-1/2"
                    }
                }
            ));
        }

        // Write game to disk.
        game.save(&next_game)?;

        ui.lock().unwrap().log(format!("Wrote game to {}", next_game.display()));

        // If user wants to quit, stop here.
        if ui.lock().unwrap().should_exit() {
            return Ok(false);
        }
    }

    ui.lock().unwrap().log(format!("Training set complete, {} games total", constants::TRAINING_SET_SIZE));

    return Ok(true);
}

fn evaluate_elo(ui: Arc<Mutex<ui::Ui>>) -> Result<bool, Box<dyn Error>> {
    ui.lock().unwrap().log("Starting ELO evaluation.");

    // Start stockfish process.
    let stockfish_path = "scripts/stockfish_14_x64_avx2";

    let mut stockfish = Command::new(stockfish_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;

    ui.lock().unwrap().log(format!("Started stockfish: {}", stockfish_path));

    let sf_stdin = stockfish.stdin.take().unwrap();
    let sf_stdout = stockfish.stdin.take().unwrap();



    return Ok(true);
}

fn do_search(model: Arc<Model>, ui: Arc<Mutex<ui::Ui>>, position: &Position) -> Result<(ChessMove, Vec<(ChessMove, f64)>), Box<dyn Error>> {
    // Run search.
    let search_ui = ui.clone();
    let tree = Searcher::new()
        .model(model)
        .position(position.clone())
        .run(move |status| {
            search_ui.lock().unwrap().status(status);
        })?;

    let score_mul = match position.side_to_move() {
        Color::White => 1.0,
        Color::Black => -1.0,
    };

    ui.lock().unwrap().score(tree.score() * score_mul);
    return Ok((tree.select(), tree.get_mcts_data()));
}

/// Advances the model generation.
/// Expects a complete training batch to be ready.
fn advance_model(ui: Arc<Mutex<ui::Ui>>) -> Result<(), Box<dyn Error>> {
    // Archive current generation.
    ui.lock().unwrap().log("Archiving model.");
    ui.lock().unwrap().log(format!("Archived as generation {}.", dir::archive()?));
    
    // Generate training batches.
    let training_batches: Vec<TrainBatch> = (0..constants::TRAINING_BATCH_COUNT)
        .map(|_| TrainBatch::generate(&dir::games_dir()?))
        .collect::<Result<Vec<TrainBatch>, _>>()?;

    // Train the model in place.
    ui.lock().unwrap().log("Training model.");
    let model = model::load(&dir::model_dir()?, true)?;
    model::train(&dir::model_dir()?, training_batches, model::get_type(&model))?;
    ui.lock().unwrap().log("Finished training model.");

    return Ok(());
}