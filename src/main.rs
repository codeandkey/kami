extern crate clap;
extern crate dirs;
extern crate serde;
extern crate tch;

mod batch;
mod listener;
mod model;
mod node;
mod perft;
mod position;
mod searcher;
mod train;
mod tree;
mod worker;

use clap::{App, Arg};

use std::error::Error;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread::{spawn, JoinHandle};

use model::mock::MockModel;
use model::Model;
use position::Position;

const PORT: u16 = 2961; // port for status clients
const TRAINING_SET_SIZE: u8 = 8; // number of games per generation
const SEARCH_TIME: usize = 2500; // milliseconds per move
const TEMPERATURE: f32   = 1.0; // MCTS initial temperature
const TEMPERATURE_DROPOFF: f32 = 0.1; // MCTS final temperature
const TEMPERATURE_DROPOFF_PLY: usize = 25; // ply to switch from initial to dropoff temperature

/**
 * Server entry point
 */

fn main() -> Result<(), Box<dyn Error>> {
    // Parse arguments
    let matches = App::new("kami")
        .version(env!("CARGO_PKG_VERSION"))
        .author("Justin Stanley <jtst@iastate.edu>")
        .about("A chess engine powered by reinforcement learning.")
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .long("verbose")
                .help("Enables verbose logging"),
        )
        .get_matches();

    // Initialize data dir
    let data_dir = dirs::data_dir().unwrap().join("kami");

    if !data_dir.is_dir() {
        println!("Initializing data dir at {}", data_dir.display());
        std::fs::create_dir_all(&data_dir).expect("failed initializing data dir");
    }

    // Initialize model
    let model_path = data_dir.join("model");
    let mut model = MockModel::new(&model_path);

    // Start status listener
    let mut listener = Arc::new(Mutex::new(listener::Listener::new(PORT)));
    listener
        .lock()
        .unwrap()
        .start()
        .expect("failed to start status listener");

    // Initialize games dir
    let games_dir = data_dir.join("games");

    if !games_dir.is_dir() {
        println!("Initializing games dir at {}", games_dir.display());
        std::fs::create_dir_all(&games_dir).expect("failed initializing games dir");
    }

    // Start playing games!
    println!("Starting games!");

    loop {
        // Play n games to complete training set.
        for game_id in 0..TRAINING_SET_SIZE {
            // Check if the game has already been generated
            let game_path = games_dir.join(format!("{}.game", game_id));

            if game_path.exists() {
                if !game_path.is_file() {
                    panic!(
                        "{} exists but is not a game file, refusing to proceed!",
                        game_path.display()
                    );
                }

                println!("{} already generated, skipping", game_path.display());
                continue;
            }

            // Game will be generated!
            println!(
                "Generating game {} of {} ({})",
                game_id + 1,
                TRAINING_SET_SIZE,
                game_path.display()
            );

            let mut fd = std::fs::File::create(game_path).expect("failed to open game output");

            // Setup a position and start making moves.
            let mut current_position = Position::new();
            let result: f32;
            let mut ply = 0;

            loop {
                // Check if the game is over
                if let Some(res) = current_position.is_game_over() {
                    result = res;
                    break;
                }

                // Game is not over, we will search the position and make a move.

                // Initialize searcher
                let mut search = searcher::Searcher::new();

                search.start(
                    Some(SEARCH_TIME),
                    model.clone(),
                    current_position.clone(),
                    listener.clone(),
                );

                // Wait for search to stop and collect final tree.
                let final_tree = search.wait().expect("search did not return tree");
                let mut temperature = TEMPERATURE;

                if ply >= TEMPERATURE_DROPOFF_PLY {
                    temperature = TEMPERATURE_DROPOFF;
                }

                // Increment ply
                ply += 1;

                // Examine tree and perform move selection.
                let selected_move = final_tree.select(temperature);

                // Write some tree data to stdout
                let children = final_tree[0].children.as_ref().unwrap().clone();
                let p_total = final_tree[0].p_total;
                let mut child_data = children
                    .iter()
                    .map(|&c| (
                        final_tree[c].action.unwrap().to_string(), // action string
                        final_tree[c].n, // visit count
                        (final_tree[c].n as f32).powf(1.0 / temperature), // normalized visit count
                        final_tree[c].p / p_total, // normalized policy
                        final_tree[c].q(), // average value
                    ))
                    .collect::<Vec<(String, u32, f32, f32, f32)>>();

                let total_nn: f32 = child_data.iter().map(|x| x.2).sum();
                let total_n: u32 = child_data.iter().map(|x| x.1).sum();

                // Reorder children by N
                child_data.sort_unstable_by(|a, b| b.1.cmp(&a.1));

                // Print out children
                println!(
                    "==> Making decision.. {} nodes considered in {}ms ({} nps)",
                    total_n,
                    SEARCH_TIME,
                    (total_n as f32) / (SEARCH_TIME as f32 / 1000.0)
                );

                for (action, n, nn, p, q) in child_data {
                    println!(
                        "{}> {:>5} | N: {:3.1}% ({:>4}) | P: {:3.1}% | Q: {:3.2}",
                        if action == selected_move.to_string() { "####" } else { "    " },
                        action,
                        nn / total_nn,
                        n,
                        p * 100.0,
                        q
                    );
                }

                // Before making the move, write the input and search snapshots to the training output.
                let frames_data = current_position.get_frames();
                let headers_data = current_position.get_headers();
                let lmm_data = current_position.get_lmm();
                let mcts_data = final_tree.get_mcts_data();
                
                // Write selected move
                fd.write(format!("{}", selected_move.to_string()).as_bytes());

                // Write LMM
                for i in 0..4096 {
                    fd.write(format!(" {}", lmm_data[i]).as_bytes());
                }

                // Write MCTS
                for i in 0..4096 {
                    fd.write(format!(" {}", mcts_data[i]).as_bytes());
                }

                // Write frames
                let frames_size = model::PLY_FRAME_COUNT * model::PLY_FRAME_SIZE * 64;
                assert_eq!(frames_size, frames_data.len());

                for i in 0..frames_size {
                    fd.write(format!(" {}", frames_data[i]).as_bytes());
                }

                // Write header
                let headers_size = model::SQUARE_HEADER_SIZE;
                assert_eq!(headers_size, headers_data.len());

                for i in 0..headers_size {
                    fd.write(format!(" {}", headers_data[i]).as_bytes());
                }

                fd.write(b"\n");

                // Make the selected move.
                current_position.make_move(selected_move);
            }

            // Game is over! Write final result to file.
            fd.write(format!("result {}\n", result).as_bytes());
        }
    }

    Ok(())
}
