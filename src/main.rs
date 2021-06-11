extern crate clap;
extern crate dirs;
extern crate serde;
extern crate tch;

mod input;
mod model;
mod node;
mod position;
mod searcher;
mod tree;
mod worker;

use chess::ChessMove;
use clap::App;

use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::str::FromStr;

use input::trainbatch::TrainBatch;
use model::mock::MockModel;
use model::Model;
use position::Position;
use rand::prelude::*;
use rand::thread_rng;
use searcher::{SearchStatus, Searcher};

const TRAINING_SET_SIZE: u8 = 1; // number of games per generation
const SEARCH_TIME: usize = 2500; // milliseconds per move
const TEMPERATURE: f32 = 1.0; // MCTS initial temperature
const TEMPERATURE_DROPOFF: f32 = 0.1; // MCTS final temperature
const TEMPERATURE_DROPOFF_PLY: usize = 25; // ply to switch from initial to dropoff temperature
const SEARCH_BATCH_SIZE: usize = 16; // number of nodes to expand at once on a single thread
const TRAINING_BATCH_SIZE: usize = 16;
const TRAINING_BATCH_COUNT: usize = 32;

/**
 * Server entry point
 */

fn main() -> Result<(), Box<dyn Error>> {
    // Show program information
    App::new("kami")
        .version(env!("CARGO_PKG_VERSION"))
        .author("Justin Stanley <jtst@iastate.edu>")
        .about("A chess engine powered by reinforcement learning.")
        .get_matches();

    // Initialize data dir
    let data_dir = dirs::data_dir().unwrap().join("kami");

    if !data_dir.is_dir() {
        println!("Initializing data dir at {}", data_dir.display());
        fs::create_dir_all(&data_dir).expect("failed initializing data dir");
    }

    // Initialize model
    let model_path = data_dir.join("model");
    let model = MockModel::new(&model_path);

    // Initialize games dir
    let games_dir = data_dir.join("games");

    if !games_dir.is_dir() {
        println!("Initializing games dir at {}", games_dir.display());
        fs::create_dir_all(&games_dir).expect("failed initializing games dir");
    }

    // Start playing games!
    println!("Starting games!");

    loop {
        // Play n games to complete training set.
        for game_id in 0..TRAINING_SET_SIZE {
            // Check if the game has already been generated
            let game_path = games_dir.join(format!("{}.game", game_id));

            if game_path.exists() {
                // If the game path isn't a file, something is very wrong
                if !game_path.is_file() {
                    panic!(
                        "{} exists but is not a game file, refusing to proceed!",
                        game_path.display()
                    );
                }

                // Check the game has a result at the end
                let fd = File::open(&game_path).expect("failed opening existing game");
                let reader = BufReader::new(fd);

                let last_line = reader.lines().last().unwrap_or(Ok("".to_string())).unwrap();

                if last_line.starts_with("result") {
                    println!("{} already generated, skipping", game_path.display());
                    continue;
                } else {
                    println!("{} incomplete, starting over..", game_path.display());
                    fs::remove_file(&game_path).expect("failed to delete incomplete game");
                }
            }

            // Game will be generated!
            println!(
                "Generating game {} of {} ({})",
                game_id + 1,
                TRAINING_SET_SIZE,
                game_path.display()
            );

            let mut fd = File::create(game_path).expect("failed to open game output");

            // Setup a position and start making moves.
            let mut current_position = Position::new();
            let result: f32;
            let mut ply = 0;

            let mut hist_moves: Vec<ChessMove> = Vec::new();

            loop {
                // Check if the game is over
                if let Some(res) = current_position.is_game_over() {
                    result = res;
                    break;
                }

                // Game is not over, we will search the position and make a move.

                // Initialize searcher
                let mut search = Searcher::new();

                // Choose temperature for this search
                let mut temperature = TEMPERATURE;

                if ply >= TEMPERATURE_DROPOFF_PLY {
                    temperature = TEMPERATURE_DROPOFF;
                }

                let search_rx = search
                    .start(
                        Some(SEARCH_TIME),
                        model.clone(),
                        current_position.clone(),
                        temperature,
                        SEARCH_BATCH_SIZE,
                    )
                    .unwrap();

                // Display search status until the search is done.
                loop {
                    let status = search_rx
                        .recv()
                        .expect("unexpected recv fail from search status rx");

                    match status {
                        SearchStatus::Searching(status) => {
                            let hist_string = hist_moves
                                .iter()
                                .map(|x| x.to_string())
                                .collect::<Vec<String>>()
                                .join(" ");

                            println!(
                                "==> Game {} of {}, hist {}",
                                game_id + 1,
                                TRAINING_SET_SIZE,
                                hist_string
                            );
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

                // Increment ply
                ply += 1;

                // Examine tree and perform move selection.
                let selected_move = final_tree.select();

                // Write some tree data to stdout
                final_tree.get_status().unwrap().print();

                // Before making the move, write the MCTS snapshot and decision to the training output.
                let mcts_data = final_tree.get_mcts_data();

                // Write selected move
                fd.write(format!("{}", selected_move.to_string()).as_bytes())
                    .expect("file write fail");

                // Write MCTS
                for i in 0..4096 {
                    fd.write(format!(" {}", mcts_data[i]).as_bytes())
                        .expect("file write fail");
                }

                fd.write(b"\n").expect("file write fail");

                // Make the selected move.
                current_position.make_move(selected_move);
                hist_moves.push(selected_move);
            }

            // Game is over! Write final result to file.
            fd.write(format!("result {}\n", result).as_bytes())
                .expect("file write fail");
        }

        println!("All games have been generated. Starting training phase.");
        println!(
            "Selecting {} batches of {} positions each from training set..",
            TRAINING_BATCH_COUNT, TRAINING_BATCH_SIZE
        );

        let mut training_batches: Vec<TrainBatch> = Vec::new();

        for _ in 0..TRAINING_BATCH_COUNT {
            let mut next_batch = TrainBatch::new(TRAINING_BATCH_SIZE);

            for _ in 0..TRAINING_BATCH_SIZE {
                // Load a random position from a random game.
                // Holy declarations batman!
                let mut rng = thread_rng();
                let game_id = rng.next_u32() as u8 % TRAINING_SET_SIZE;
                let game_path = games_dir.join(format!("{}.game", game_id));
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

        println!("Finished building training batches. Archiving previous model and games.");

        let archive_path = data_dir.join("archive");

        if !archive_path.is_dir() {
            fs::create_dir_all(&archive_path).expect("failed to initialize archive dir");
        }

        // Walk through archive generations to find the lowest available slot.
        let mut cur: usize = 0;

        loop {
            let gen_path = archive_path.join(format!("generation_{}", cur));

            if gen_path.exists() {
                cur += 1;
                continue;
            }

            println!("Archiving model as generation {}.", cur);
            fs_extra::move_items(
                &[&model_path, &games_dir],
                gen_path,
                &fs_extra::dir::CopyOptions::new(),
            )
            .expect("failed to archive generation");
            println!("Archived model. ");

            // Make new games dir for new generation.
            fs::create_dir_all(&games_dir).expect("failed initializing games dir");
            break;
        }

        println!("Training current model.");
        model.write().unwrap().train(training_batches);
        println!("Finished training model. ");

        println!("Writing new model to disk.");
        model
            .read()
            .unwrap()
            .write(&model_path)
            .expect("failed writing new model");
        println!("Finished training iteration! Building new training set.");
    }
}
