extern crate dirs;
extern crate serde;

#[cfg(feature = "tch")]
extern crate tch;

mod constants;
mod dir;
mod game;
mod input;
mod model;
mod node;
mod position;
mod searcher;
mod train;
mod tree;
mod ui;
mod worker;

use std::error::Error;
use std::fs;
use std::io::{stdout, stdin, Read, Write};
use std::path::Path;
use std::str::FromStr;
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

    // Set data dir
    let data_dir = dirs::data_dir().expect("Error getting data directory!").join("kami");

    sleep(Duration::from_secs(1));
    
    train::train()?;

    println!("Shutdown complete, goodbye :)");
    Ok(())
}
