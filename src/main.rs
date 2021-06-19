extern crate dirs;
extern crate serde;

#[cfg(feature = "tch")]
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
mod tui;
mod worker;

use disk::Disk;
use std::error::Error;
use std::sync::{Arc, Mutex};
use trainer::Trainer;

/**
 * Server entry point
 */

fn main() -> Result<(), Box<dyn Error>> {
    // Set data dir
    let data_dir = dirs::data_dir().unwrap().join("kami");

    // Initialize disk manager
    let diskmgr = Arc::new(Mutex::new(Disk::new(&data_dir)?));

    // Initialize trainer
    let mut trainer = Trainer::new(diskmgr.clone(), None)?;

    trainer.start();
    trainer.wait()?;

    println!("Shutdown complete, goodbye :)");
    Ok(())
}
