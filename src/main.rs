extern crate clap;
extern crate dirs;
extern crate serde;
extern crate tch;

mod disk;
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

use disk::Disk;
use input::trainbatch::TrainBatch;
use model::mock::MockModel;
use model::Model;
use position::Position;
use rand::prelude::*;
use rand::thread_rng;
use searcher::{SearchStatus, Searcher};

const TRAINING_SET_SIZE: usize = 1; // number of games per generation
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

    // Initialize disk manager
    let mut diskmgr = Disk::new(&data_dir)?;

    // Load model, generate one if doesn't exist.
    if !diskmgr.load()? {
        diskmgr.generate(model::make_ptr(MockModel::generate()?))?;
    }

    // Initialize games dir
    let games_dir = data_dir.join("games");

    if !games_dir.is_dir() {
        println!("Initializing games dir at {}", games_dir.display());
        fs::create_dir_all(&games_dir).expect("failed initializing games dir");
    }

    // Start playing games!
    println!("Starting games!");

    loop {
        while let Some(game_path) = diskmgr.next_game_path(TRAINING_SET_SIZE)? {
            println!(
                "Generating game {}",
                game_path.display()
            );

            let mut fd = File::create(&game_path).expect("failed to open game output");

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
                        diskmgr.get_model().unwrap(),
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
                                "==> Game {}, hist {}",
                                game_path.display(),
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

        let batches = diskmgr.get_training_batch(TRAINING_BATCH_COUNT, TRAINING_BATCH_SIZE)?;

        println!("Finished building training batches.");
        println!("Archived generation {}", diskmgr.train(batches)?);
    }
}
