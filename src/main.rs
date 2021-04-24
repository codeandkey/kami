extern crate dirs;
#[macro_use] extern crate log;
extern crate pretty_env_logger;
extern crate tensorflow;

use std::fs;
use tensorflow::SessionOptions;

mod net;

fn main() {
    pretty_env_logger::init();

    let data_dir = dirs::data_dir().unwrap().join("kami");

    // Setup data directories
    fs::create_dir_all(&data_dir).expect("Failed to initialize data directory");

    // Setup tensorflow
    info!("Loading latest model..");

    let latest = net::Model::load(&data_dir.join("model"), SessionOptions::new()).expect("Failed to load model.");
}
