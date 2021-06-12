extern crate clap;
extern crate dirs;
extern crate serde;
extern crate tch;

mod constants;
mod disk;
mod game;
mod input;
mod model;
mod node;
mod position;
mod searcher;
mod trainer;
mod tree;
mod worker;

use clap::App;

use std::error::Error;

use std::sync::{Arc, Mutex};

use disk::Disk;

use trainer::Trainer;

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

    // Set data dir
    let data_dir = dirs::data_dir().unwrap().join("kami");

    // Initialize disk manager
    let diskmgr = Arc::new(Mutex::new(Disk::new(&data_dir)?));

    // Initialize trainer
    let mut trainer = Trainer::new(diskmgr.clone())?;

    let stopflag = trainer.start_training();

    // Set shutdown handler
    let ctrlc_stopflag = stopflag.clone();

    ctrlc::set_handler(move || {
        *ctrlc_stopflag.lock().unwrap() = true;
        println!("Received ctrlc, starting shutdown.");
    })?;

    loop {
        if *stopflag.lock().unwrap() {
            break;
        }

        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    trainer.wait()?;

    println!("Shutdown complete, goodbye :)");
    Ok(())
}
