/// Routines for evaluating model ELO
use crate::constants;
use crate::dir;
use crate::game::Game;
use crate::input::trainbatch::TrainBatch;
use crate::model::{self, Model};
use crate::position::Position;
use crate::searcher::Searcher;
use crate::ui::{self, Ui};

use chess::{ChessMove, Color};
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

use rand::{prelude::*, thread_rng};

use std::process::{Command, Stdio};

use std::error::Error;
use std::fs;
use std::io::{stdout, BufRead, BufReader, Read, Write};
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::thread::spawn;
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

    execute!(stdout(), EnterAlternateScreen).expect("failed starting alternate screen");

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
                    }
                    KeyCode::Char('p') => inp_ui.lock().unwrap().pause(),
                    _ => (),
                },
                _ => (),
            };
        }
    });

    // Start training loop.
    for generation in dir::get_generation()?..usize::MAX {
        ui.lock().unwrap().log(format!("Resuming generation {}", generation));

        // Perform ELO evaluation if ready.
        if generation % constants::ELO_EVALUATION_INTERVAL == 0 {
            if !evaluate_elo(ui.clone())? {
                break;
            }
        }

        // Generate training set.
        if !generate_training_set(ui.clone())? {
            break;
        }

        // Archive and train model.
        advance_model(ui.clone())?;
    }

    // Stop input thread.
    inp_thread.join().unwrap();

    // Stop ui thread.
    ui.lock().unwrap().join();

    // Leave TUI screen and reset terminal.
    execute!(stdout(), LeaveAlternateScreen).expect("failed leaving alternate screen");

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
            ui.lock()
                .unwrap()
                .log(format!("Resuming game {}", next_game.display()));
            Game::load(&next_game)?
        } else {
            ui.lock()
                .unwrap()
                .log(format!("Generating game {}", next_game.display()));
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

            ui.lock().unwrap().log(format!(
                "Game over, {}",
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

        ui.lock()
            .unwrap()
            .log(format!("Wrote game to {}", next_game.display()));

        // If user wants to quit, stop here.
        if ui.lock().unwrap().should_exit() {
            return Ok(false);
        }
    }

    ui.lock().unwrap().log(format!(
        "Training set complete, {} games total",
        constants::TRAINING_SET_SIZE
    ));

    return Ok(true);
}

fn evaluate_elo(ui: Arc<Mutex<ui::Ui>>) -> Result<bool, Box<dyn Error>> {
    ui.lock().unwrap().log(format!(
        "Starting ELO evaluation on generation {}",
        dir::get_generation()?
    ));

    // Load the current model.
    let model = Arc::new(model::load(&dir::model_dir()?, true)?);

    // Start stockfish process.
    let stockfish_path = "scripts/stockfish_14_x64_avx2";

    let mut stockfish = Command::new(stockfish_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;

    ui.lock()
        .unwrap()
        .log(format!("Started stockfish: {}", stockfish_path));

    let mut sf_stdin = stockfish.stdin.take().unwrap();
    let mut sf_stdout = BufReader::new(stockfish.stdout.take().unwrap());

    let mut sf_input;

    let sf_read = |sf: &mut BufReader<_>| -> Result<String, Box<dyn Error>> {
        let mut input = String::new();
        sf.read_line(&mut input)?;
        Ok(input.trim().to_string())
    };

    // Perform UCI handshake
    sf_stdin.write(b"uci\n")?;

    // Wait for uciok
    loop {
        sf_input = sf_read(&mut sf_stdout)?;

        if sf_input == "uciok" {
            ui.lock().unwrap().log("Stockfish ready.");
            break;
        }
    }

    let mut rng = thread_rng();
    let mut kami_move = rng.next_u32() % 2 == 0;

    // Enable stockfish strength tuning
    sf_stdin.write(b"setoption UCI_LimitStrength value true\n")?;

    // Start playing games.
    while let Some((next_game, game_id)) = dir::next_elo_game()? {
        // Set initial position
        let mut position = Position::new();

        let mut game = if next_game.exists() {
            ui.lock()
                .unwrap()
                .log(format!("Resuming game {}", next_game.display()));
            Game::load(&next_game)?
        } else {
            ui.lock()
                .unwrap()
                .log(format!("Generating game {}", next_game.display()));
            Game::new()
        };

        // Get game back up to current position
        let actions = game.get_actions();

        for mv in &actions {
            assert!(position.make_move(*mv), "Invalid move in resumed game!");
        }

        if actions.len() > 0 {
            // Resuming game, figure out whose turn it is by the MCTS counts.
            kami_move = game.get_mcts().last().unwrap().len() == 0;
        }

        // Set stockfish ELO rating
        let sf_elo = constants::STOCKFISH_ELO[game_id];

        sf_stdin.write(format!("setoption UCI_Elo value {}\n", sf_elo).as_bytes())?;

        if kami_move ^ (position.side_to_move() == Color::Black) {
            ui.lock().unwrap().log(format!(
                "Kami generation {} VS. Stockfish [{}]",
                dir::get_generation()?,
                sf_elo
            ));
        } else {
            ui.lock().unwrap().log(format!(
                "Stockfish [{}] VS. Kami generation {}",
                sf_elo,
                dir::get_generation()?
            ));
        }

        if kami_move {
            ui.lock().unwrap().log("Kami to move. GLHF!");
        } else {
            ui.lock().unwrap().log("Stockfish to move. GLHF!");
        }

        ui.lock().unwrap().reset();

        while position.is_game_over().is_none() {
            // Make a move!
            ui.lock().unwrap().position(position.clone());

            if kami_move {
                // Kami's turn.

                if ui.lock().unwrap().should_exit() {
                    break;
                }

                let (selected_move, mcts) = do_search(model.clone(), ui.clone(), &position)?;
                assert!(position.make_move(selected_move));
                game.make_move(selected_move, mcts);
            } else {
                // Stockfish's turn.
                // Load position into uci

                sf_stdin.write(
                    format!(
                        "position startpos moves {}\n",
                        game.get_actions()
                            .iter()
                            .map(|x| x.to_string())
                            .collect::<Vec<String>>()
                            .join(" ")
                    )
                    .as_bytes(),
                )?;
                sf_stdin.write(format!("go movetime {}\n", constants::MOVETIME_ELO).as_bytes())?;

                // Wait for selected move

                loop {
                    sf_input = sf_read(&mut sf_stdout)?;

                    let parts: Vec<&str> = sf_input.split(" ").collect();

                    if parts.len() > 0 {
                        if parts[0] == "bestmove" {
                            let selected_move =
                                ChessMove::from_str(parts[1]).expect("bad move from stockfish?");

                            assert!(position.make_move(selected_move));
                            game.make_move(selected_move, Vec::new());

                            sf_stdin.write(b"stop\n")?;
                            break;
                        }
                    }
                }
            }

            // Switch sides.
            kami_move = !kami_move;
        }

        // Finalize game if it is over.
        if let Some(result) = position.is_game_over() {
            game.finalize(result);

            ui.lock().unwrap().log(format!(
                "Game over, {} ({})",
                if result > 0.0 {
                    "1-0"
                } else {
                    if result < 0.0 {
                        "0-1"
                    } else {
                        "1/2-1/2"
                    }
                },
                if result != 0.0 {
                    if kami_move {
                        "Stockfish wins"
                    } else {
                        "Kami wins"
                    }
                } else {
                    "draw"
                }
            ));
        }

        // Write game to disk.
        game.save(&next_game)?;

        ui.lock()
            .unwrap()
            .log(format!("Wrote game to {}", next_game.display()));

        // If user wants to quit, stop here.
        if ui.lock().unwrap().should_exit() {
            return Ok(false);
        }
    }

    // All games generated, compute final ELO result.
    let game_set = dir::elo_game_set()?;
    let mut results = Vec::new();

    for game in &game_set {
        if game.get_mcts()[0].len() > 0 {
            results.push(game.get_result().unwrap());
        } else {
            results.push(-game.get_result().unwrap());
        }
    }

    // Compute ELO
    let score: f32 = results.iter().sum();
    let elo = (constants::STOCKFISH_ELO.iter().sum::<usize>() as f32 + 400.0 * score)
        / constants::ELO_EVALUATION_NUM_GAMES as f32;

    ui.lock()
        .unwrap()
        .log(format!("Finished ELO evaluation. Total score: {}", score));
    ui.lock().unwrap().log(format!(
        "ELO estimate for generation {}: {}",
        dir::get_generation()?,
        elo
    ));

    // Write ELO results.
    fs::write(dir::games_dir()?.join("elo"), format!("{}", elo).as_bytes())?;

    return Ok(true);
}

/// Searches a position.
/// Returns the selected move along with the search MCTS counts.
fn do_search(
    model: Arc<Model>,
    ui: Arc<Mutex<ui::Ui>>,
    position: &Position,
) -> Result<(ChessMove, Vec<(ChessMove, f64)>), Box<dyn Error>> {
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
    // Generate training batches.
    let training_batches: Vec<TrainBatch> = (0..constants::TRAINING_BATCH_COUNT)
        .map(|_| TrainBatch::generate(&dir::games_dir()?))
        .collect::<Result<Vec<TrainBatch>, _>>()?;

    // Train the model in place.
    ui.lock().unwrap().log("Training model.");
    let model = model::load(&dir::model_dir()?, true)?;
    model::train(
        &dir::model_dir()?,
        training_batches,
        model::get_type(&model),
    )?;
    ui.lock().unwrap().log("Finished training model.");

    // Advance generation number.
    dir::new_generation()?;
    ui.lock().unwrap().log("Incremented generation.");

    return Ok(());
}
